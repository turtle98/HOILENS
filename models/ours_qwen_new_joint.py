"""
Joint single-pass HOI detection with Qwen2.5-VL-3B.

Instead of the encoder-decoder cross-attention structure in ours_qwen_new_old.py,
region tokens are injected directly into Qwen's sequence and processed in one
forward pass through Qwen's actual language model layers (self-attention + LoRA).

Architecture change vs. ours_qwen_new_old.py:
  OLD  : frozen Qwen → 37 hidden states → ROI align on hs[0] (image tokens only)
           → CrossAttendWithLoRA(region queries, layer-aligned context list)
  NEW  : frozen Qwen hook → full-sequence inputs_embeds → ROI align on image slice
           → concat [inputs_embeds, region_embeds] → QwenSelfAttendLoRA (one pass)
           → extract [:, -N_region:, :] → lm_head projection

Everything outside of __init__ / compute_sim_scores is identical to
ours_qwen_new_old.py.  Sections marked # [JOINT] highlight the differences.
"""

import os
import sys
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou
import json
import numpy as np
import torchvision
import math
import copy

from utils.hico_list import hico_verbs_sentence, hico_verb_object_list, hico_verbs, hico_objects
from utils.vcoco_list import vcoco_verbs_sentence
from utils.hico_utils import reserve_indices, HOI_IDX_TO_ACT_IDX
from utils.postprocessor import PostProcess
from utils.ops import (
    binary_focal_loss_with_logits, binary_focal_loss,
    vectorized_bboxes_and_indices, bbox_to_token, compute_spatial_encodings,
)
from utils import hico_text_label
from utils.builder import build_vision_projector
from utils.position_encoding import compute_sinusoidal_pe

from CLIP.clip import build_model
from CLIP.customCLIP import CustomCLIP, tokenize

from detr.models.backbone import build_backbone
from detr.models.transformer import build_transformer
from detr.models.detr import DETR
from detr.util import box_ops

from transformers import CLIPVisionModel, CLIPImageProcessor, AutoConfig

# [JOINT] import from qwen_utils_joint instead of qwen_utils
from methods.qwen_utils_joint import (
    load_qwen_state,
    run_qwen_model,
    generate_qwen_model,
    compute_conditional_likelihood_qwen,
    retrieve_logit_lens_qwen,
    get_img_idx_qwen,
    tensor_to_pil,
    get_lm_head_embeddings_qwen,
    _get_qwen_grid,
    QWEN_HIDDEN_SIZE,
    QWEN_NUM_LAYERS,
    QWEN_GRID_H,
    QWEN_GRID_W,
    QWEN_NUM_IMAGE_TOKENS,
    QWEN_IMAGE_SIZE,
    QWEN_PATCH_SIZE,
    QWEN_MERGE_SIZE,
    QWEN_START_LAYER,
    QWEN_NUM_HEADS,
    QWEN_NUM_KV_HEADS,
    QWEN_HEAD_DIM,
    QWEN_KV_HIDDEN_SIZE,
    # [JOINT] new helper
    get_qwen_inputs_embeds,
)
from methods.attention import llama_modify

from detr.util.misc import nested_tensor_from_tensor_list


# ---------------------------------------------------------------------------
# Shared small modules
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LoRALinear(nn.Module):
    """LoRA adapter for a frozen linear layer."""

    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.rank    = rank
        self.alpha   = alpha
        self.scaling = alpha / rank
        self.lora_A  = nn.Linear(in_features, rank, bias=False)
        self.lora_B  = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x, frozen_weight):
        frozen_dtype = frozen_weight.weight.dtype
        frozen_out   = frozen_weight(x.to(frozen_dtype))
        lora_out     = self.lora_B(self.lora_A(x.float())).to(frozen_dtype) * self.scaling
        return frozen_out + lora_out


# ---------------------------------------------------------------------------
# [JOINT] QwenSelfAttendLoRA — replaces CrossAttendWithLoRA
# ---------------------------------------------------------------------------

class QwenSelfAttendLoRA(nn.Module):
    """
    Runs Qwen's actual language model layers as self-attention with LoRA.

    Unlike CrossAttendWithLoRA (which clones layers and does cross-attention
    between region queries and layer-aligned encoder context), this module:
      - Holds a reference to Qwen's original 36 LM layers (frozen)
      - Adds LoRA adapters to each layer's self-attention projections
      - forward() runs the full combined sequence [image+text tokens, region tokens]
        through all LM layers as self-attention — region tokens naturally attend
        to image tokens at every layer
      - Applies the final layer norm and returns the full output sequence

    Caller extracts output[:, -N_region:, :] to get region token outputs.
    """

    def __init__(self, qwen_model, lora_rank=8, lora_alpha=8):
        super().__init__()
        self.lm   = qwen_model.model.language_model
        self.norm = self.lm.norm

        # Freeze all LM parameters
        for param in self.lm.parameters():
            param.requires_grad = False

        # LoRA adapters — one set per layer, same GQA dims as CrossAttendWithLoRA
        self.lora_q = nn.ModuleList()
        self.lora_k = nn.ModuleList()
        self.lora_v = nn.ModuleList()
        self.lora_o = nn.ModuleList()

        for layer in self.lm.layers:
            attn   = layer.self_attn
            q_dim  = attn.q_proj.weight.shape[0]   # 2048
            kv_dim = attn.k_proj.weight.shape[0]   # 1024  (GQA)
            self.lora_q.append(LoRALinear(q_dim, q_dim,  lora_rank, lora_alpha))
            self.lora_k.append(LoRALinear(q_dim, kv_dim, lora_rank, lora_alpha))
            self.lora_v.append(LoRALinear(q_dim, kv_dim, lora_rank, lora_alpha))
            self.lora_o.append(LoRALinear(q_dim, q_dim,  lora_rank, lora_alpha))

    def forward(self, combined_embeds):
        """
        Args:
            combined_embeds : [1, seq_len + N_region, QWEN_HIDDEN_SIZE]
                              Full sequence with region tokens appended.
        Returns:
            hidden_states   : [1, seq_len + N_region, QWEN_HIDDEN_SIZE]
                              After all LM layers + final norm.
                              Caller slices [:, -N_region:, :] for region outputs.
        """
        hidden_states = combined_embeds
        bsz, full_len, _ = hidden_states.shape

        for idx, layer in enumerate(self.lm.layers):
            attn         = layer.self_attn
            head_dim     = attn.head_dim          # 128
            num_heads    = attn.num_heads          # 16
            num_kv_heads = attn.num_key_value_heads  # 8
            groups       = num_heads // num_kv_heads  # 2

            residual = hidden_states
            normed   = layer.input_layernorm(hidden_states)

            # Self-attention: Q, K, V all from the same sequence
            q = self.lora_q[idx](normed, attn.q_proj)  # [bsz, full_len, 2048]
            k = self.lora_k[idx](normed, attn.k_proj)  # [bsz, full_len, 1024]
            v = self.lora_v[idx](normed, attn.v_proj)  # [bsz, full_len, 1024]

            q = q.view(bsz, full_len, num_heads,    head_dim).transpose(1, 2)
            k = k.view(bsz, full_len, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(bsz, full_len, num_kv_heads, head_dim).transpose(1, 2)

            # GQA: expand K, V to match Q head count
            k = k.repeat_interleave(groups, dim=1)  # [bsz, 16, full_len, 128]
            v = v.repeat_interleave(groups, dim=1)

            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None)

            attn_out      = attn_out.transpose(1, 2).contiguous().view(bsz, full_len, -1)
            attn_out      = self.lora_o[idx](attn_out, attn.o_proj)

            hidden_states = residual + attn_out
            residual      = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# HOIQWEN_Joint
# ---------------------------------------------------------------------------

class HOIQWEN_Joint(nn.Module):
    def __init__(
        self,
        args,
        detector:    nn.Module,
        postprocessor: nn.Module,
        model:       nn.Module,       # Qwen state dict
        object_embedding: torch.Tensor,
        human_idx:   int,
        num_classes: int,
        alpha: float = 0.5,
        gamma: float = 2.0,
        box_score_thresh: float = 0.2,
        fg_iou_thresh:    float = 0.5,
        min_instances:    int   = 3,
        max_instances:    int   = 15,
        object_class_to_target_class: List[list] = None,
        object_n_verb_to_interaction: List[list] = None,
    ) -> None:
        super().__init__()

        self.detector      = detector
        self.postprocessor = postprocessor
        self.clip_head     = model          # Qwen state dict

        self.register_buffer("object_embedding", object_embedding)

        self.visual_output_dim = QWEN_HIDDEN_SIZE

        self.object_n_verb_to_interaction = np.asarray(
            object_n_verb_to_interaction, dtype=float
        )
        self.args        = args
        self.human_idx   = human_idx
        self.num_classes = num_classes
        self.hyper_lambda = args.hyper_lambda
        self.alpha       = alpha
        self.gamma       = gamma
        self.box_score_thresh       = box_score_thresh
        self.fg_iou_thresh          = fg_iou_thresh
        self.min_instances          = min_instances
        self.max_instances          = max_instances
        self.object_class_to_target_class = object_class_to_target_class
        self.dataset       = args.dataset
        self.reserve_indices = reserve_indices

        qwen_model = self.clip_head["model"]
        self.lm_head_embeddings = qwen_model.lm_head.weight.detach().cpu()

        tokenizer = self.clip_head['tokenizer']

        # ------------------------------------------------------------------
        # Verb token ids (117-class path)
        # ------------------------------------------------------------------
        if self.num_classes == 117:
            verb_forms_3 = [
                ('adjust',         'adjusting',      'adjusted'),
                ('assemble',       'assembling',     'assembled'),
                ('block',          'blocking',       'blocked'),
                ('blow',           'blowing',        'blew'),
                ('board',          'boarding',       'boarded'),
                ('break',          'breaking',       'broke'),
                ('brush_with',     'brushing with',  'brushed with'),
                ('buy',            'buying',         'bought'),
                ('carry',          'carrying',       'carried'),
                ('catch',          'catching',       'caught'),
                ('chase',          'chasing',        'chased'),
                ('check',          'checking',       'checked'),
                ('clean',          'cleaning',       'cleaned'),
                ('control',        'controlling',    'controlled'),
                ('cook',           'cooking',        'cooked'),
                ('cut',            'cutting',        'cut'),
                ('cut_with',       'cutting with',   'cut with'),
                ('direct',         'directing',      'directed'),
                ('drag',           'dragging',       'dragged'),
                ('dribble',        'dribbling',      'dribbled'),
                ('drink_with',     'drinking with',  'drank with'),
                ('drive',          'driving',        'drove'),
                ('dry',            'drying',         'dried'),
                ('eat',            'eating',         'ate'),
                ('eat_at',         'eating at',      'ate at'),
                ('exit',           'exiting',        'exited'),
                ('feed',           'feeding',        'fed'),
                ('fill',           'filling',        'filled'),
                ('flip',           'flipping',       'flipped'),
                ('flush',          'flushing',       'flushed'),
                ('fly',            'flying',         'flew'),
                ('greet',          'greeting',       'greeted'),
                ('grind',          'grinding',       'ground'),
                ('groom',          'grooming',       'groomed'),
                ('herd',           'herding',        'herded'),
                ('hit',            'hitting',        'hit'),
                ('hold',           'holding',        'held'),
                ('hop_on',         'hopping on',     'hopped on'),
                ('hose',           'hosing',         'hosed'),
                ('hug',            'hugging',        'hugged'),
                ('hunt',           'hunting',        'hunted'),
                ('inspect',        'inspecting',     'inspected'),
                ('install',        'installing',     'installed'),
                ('jump',           'jumping',        'jumped'),
                ('kick',           'kicking',        'kicked'),
                ('kiss',           'kissing',        'kissed'),
                ('lasso',          'lassoing',       'lassoed'),
                ('launch',         'launching',      'launched'),
                ('lick',           'licking',        'licked'),
                ('lie_on',         'lying on',       'lied on'),
                ('lift',           'lifting',        'lifted'),
                ('light',          'lighting',       'lit'),
                ('load',           'loading',        'loaded'),
                ('lose',           'losing',         'lost'),
                ('make',           'making',         'made'),
                ('milk',           'milking',        'milked'),
                ('move',           'moving',         'moved'),
                ('no_interaction', 'no interaction', 'no interaction'),
                ('open',           'opening',        'opened'),
                ('operate',        'operating',      'operated'),
                ('pack',           'packing',        'packed'),
                ('paint',          'painting',       'painted'),
                ('park',           'parking',        'parked'),
                ('pay',            'paying',         'paid'),
                ('pet',            'petting',        'petted'),
                ('pick',           'picking',        'picked'),
                ('pick_up',        'picking up',     'picked up'),
                ('point',          'pointing',       'pointed'),
                ('pour',           'pouring',        'poured'),
                ('pull',           'pulling',        'pulled'),
                ('push',           'pushing',        'pushed'),
                ('race',           'racing',         'raced'),
                ('read',           'reading',        'read'),
                ('release',        'releasing',      'released'),
                ('repair',         'repairing',      'repaired'),
                ('ride',           'riding',         'rode'),
                ('row',            'rowing',         'rowed'),
                ('run',            'running',        'ran'),
                ('sail',           'sailing',        'sailed'),
                ('scratch',        'scratching',     'scratched'),
                ('serve',          'serving',        'served'),
                ('set',            'setting',        'set'),
                ('shear',          'shearing',       'sheared'),
                ('sign',           'signing',        'signed'),
                ('sip',            'sipping',        'sipped'),
                ('sit_at',         'sitting at',     'sat at'),
                ('sit_on',         'sitting on',     'sat on'),
                ('slide',          'sliding',        'slid'),
                ('smell',          'smelling',       'smelled'),
                ('spin',           'spinning',       'spun'),
                ('squeeze',        'squeezing',      'squeezed'),
                ('stand_on',       'standing on',    'stood on'),
                ('stand_under',    'standing under', 'stood under'),
                ('stick',          'sticking',       'stuck'),
                ('stir',           'stirring',       'stirred'),
                ('stop_at',        'stopping at',    'stopped at'),
                ('straddle',       'straddling',     'straddled'),
                ('swing',          'swinging',       'swung'),
                ('tag',            'tagging',        'tagged'),
                ('talk_on',        'talking on',     'talked on'),
                ('teach',          'teaching',       'taught'),
                ('text_on',        'texting on',     'texted on'),
                ('throw',          'throwing',       'threw'),
                ('tie',            'tying',          'tied'),
                ('toast',          'toasting',       'toasted'),
                ('train',          'training',       'trained'),
                ('type_on',        'typing on',      'typed on'),
                ('walk',           'walking',        'walked'),
                ('wash',           'washing',        'washed'),
                ('watch',          'watching',       'watched'),
                ('wave',           'waving',         'waved'),
                ('wear',           'wearing',        'wore'),
                ('wield',          'wielding',       'wielded'),
                ('zip',            'zipping',        'zipped'),
            ]

            max_tokens  = 6
            num_verbs   = len(verb_forms_3)
            num_forms   = 3

            verb_token_ids  = torch.full(
                (num_verbs, num_forms, max_tokens), -1, dtype=torch.long
            )
            verb_token_mask = torch.zeros(
                (num_verbs, num_forms, max_tokens), dtype=torch.bool
            )

            for v_idx, forms in enumerate(verb_forms_3):
                for f_idx, form in enumerate(forms):
                    ids = tokenizer.encode(form, add_special_tokens=False)
                    ids = ids[:max_tokens]
                    verb_token_ids [v_idx, f_idx, :len(ids)] = torch.tensor(ids)
                    verb_token_mask[v_idx, f_idx, :len(ids)] = True

            self.register_buffer('verb_token_ids',  verb_token_ids)
            self.register_buffer('verb_token_mask', verb_token_mask)

        # ------------------------------------------------------------------
        # HOI token ids (600-class path)
        # ------------------------------------------------------------------
        elif self.num_classes == 600:
            hoi_labels   = list(hico_text_label.hico_text_label.values())
            max_hoi_tok  = 12
            num_hoi      = len(hoi_labels)

            hoi_token_ids  = torch.full((num_hoi, max_hoi_tok), -1, dtype=torch.long)
            hoi_token_mask = torch.zeros((num_hoi, max_hoi_tok), dtype=torch.bool)

            for h_idx, label in enumerate(hoi_labels):
                ids = tokenizer.encode(label, add_special_tokens=False)
                ids = ids[:max_hoi_tok]
                hoi_token_ids [h_idx, :len(ids)] = torch.tensor(ids)
                hoi_token_mask[h_idx, :len(ids)] = True

            self.register_buffer('hoi_token_ids',  hoi_token_ids)
            self.register_buffer('hoi_token_mask', hoi_token_mask)

            self.filtered_hoi_idx = [
                i for i, label in enumerate(hoi_labels)
                if label in [
                    "no_interaction bicycle", "no_interaction car",
                    "no_interaction motorcycle", "no_interaction airplane",
                    "no_interaction bus", "no_interaction train",
                    "no_interaction truck", "no_interaction boat",
                    "no_interaction bench", "no_interaction bottle",
                    "no_interaction chair", "no_interaction couch",
                    "no_interaction table", "no_interaction door",
                    "no_interaction tv", "no_interaction laptop",
                    "no_interaction mouse", "no_interaction remote",
                    "no_interaction keyboard", "no_interaction microwave",
                    "no_interaction oven", "no_interaction toaster",
                    "no_interaction sink", "no_interaction refrigerator",
                    "no_interaction book", "no_interaction clock",
                    "no_interaction vase", "no_interaction scissors",
                    "no_interaction hair_drier", "no_interaction toothbrush",
                ]
            ]

            self.seen_verb_idxs = list(range(117))

        elif self.num_classes == 24:
            vcoco_verb_forms = [
                ('answer',     'answering',    'answered'),
                ('cut',        'cutting',      'cut'),
                ('drink',      'drinking',     'drank'),
                ('eat',        'eating',       'ate'),
                ('hit',        'hitting',      'hit'),
                ('hold',       'holding',      'held'),
                ('jump',       'jumping',      'jumped'),
                ('kick',       'kicking',      'kicked'),
                ('lay',        'laying',       'laid'),
                ('look',       'looking',      'looked'),
                ('point',      'pointing',     'pointed'),
                ('read',       'reading',      'read'),
                ('ride',       'riding',       'rode'),
                ('run',        'running',      'ran'),
                ('sit',        'sitting',      'sat'),
                ('skateboard', 'skateboarding','skateboarded'),
                ('ski',        'skiing',       'skied'),
                ('smile',      'smiling',      'smiled'),
                ('snowboard',  'snowboarding', 'snowboarded'),
                ('stand',      'standing',     'stood'),
                ('surf',       'surfing',      'surfed'),
                ('talk',       'talking',      'talked'),
                ('throw',      'throwing',     'threw'),
                ('work_on',    'working on',   'worked on'),
            ]
            max_tokens  = 6
            num_verbs   = len(vcoco_verb_forms)
            num_forms   = 3

            verb_token_ids  = torch.full(
                (num_verbs, num_forms, max_tokens), -1, dtype=torch.long
            )
            verb_token_mask = torch.zeros(
                (num_verbs, num_forms, max_tokens), dtype=torch.bool
            )
            for v_idx, forms in enumerate(vcoco_verb_forms):
                for f_idx, form in enumerate(forms):
                    ids = tokenizer.encode(form, add_special_tokens=False)
                    ids = ids[:max_tokens]
                    verb_token_ids [v_idx, f_idx, :len(ids)] = torch.tensor(ids)
                    verb_token_mask[v_idx, f_idx, :len(ids)] = True

            self.register_buffer('verb_token_ids',  verb_token_ids)
            self.register_buffer('verb_token_mask', verb_token_mask)

        if self.num_classes == 117:
            self.unseen_verb_idxs = [
                i for i in range(self.num_classes)
                if i not in self.seen_verb_idxs
            ]
        elif self.num_classes == 600:
            self.seen_hoi_idxs = [
                i for i in range(self.num_classes)
                if i not in self.filtered_hoi_idx
            ]

        # ------------------------------------------------------------------
        # [JOINT] Single self-attention module on Qwen's actual LM layers
        # ------------------------------------------------------------------
        self.joint_lora = QwenSelfAttendLoRA(qwen_model, lora_rank=8, lora_alpha=8)

        # [JOINT] Spatial head: output dim = QWEN_HIDDEN_SIZE (2048)
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),  nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, QWEN_HIDDEN_SIZE), nn.ReLU(),
        ).to(torch.bfloat16)

    # -----------------------------------------------------------------------
    def compute_prior_scores(
        self, x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
    ) -> Tensor:
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        p   = 1.0 if self.training else self.hyper_lambda
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        target_cls_idx = [
            self.object_class_to_target_class[obj.item()] for obj in object_class[y]
        ]
        pair_idx       = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        flat_target    = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target] = s_h[pair_idx]
        prior_o[pair_idx, flat_target] = s_o[pair_idx]
        return torch.stack([prior_h, prior_o])

    # -----------------------------------------------------------------------
    def compute_sim_scores(
        self,
        region_props: List[dict],
        image,
        targets,
        pil_images: list = None,
        priors=None
    ):
        device = image.tensors.device

        boxes_h_collated      = []
        boxes_o_collated      = []
        prior_collated        = []
        object_class_collated = []
        all_logits_collated   = []
        ho_vocab_collated     = []
        h_vocab_collated      = []
        o_vocab_collated      = []

        for b_idx, props in enumerate(region_props):
            boxes  = props['boxes']
            scores = props['scores']
            labels = props['labels']
            feats  = props['feat']

            is_human = labels == self.human_idx
            n_h      = torch.sum(is_human)
            n        = len(boxes)

            if not torch.all(labels[:n_h] == self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm  = torch.cat([h_idx, o_idx])
                boxes  = boxes[perm]
                scores = scores[perm]
                labels = labels[perm]

            vocab_size = self.lm_head_embeddings.shape[0]

            if n_h == 0 or n <= 1:
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(
                    torch.zeros(2, 0, self.num_classes, device=device)
                )
                ho_vocab_collated.append(torch.zeros(0, vocab_size, device=device))
                h_vocab_collated.append( torch.zeros(0, vocab_size, device=device))
                o_vocab_collated.append( torch.zeros(0, vocab_size, device=device))
                continue

            x, y   = torch.meshgrid(
                torch.arange(n, device=device), torch.arange(n, device=device)
            )
            x_keep, y_keep = torch.nonzero(
                torch.logical_and(x != y, x < n_h)
            ).unbind(1)

            if len(x_keep) == 0:
                raise ValueError("There are no valid human-object pairs")

            h_unique_indices, h_inverse_indices = torch.unique(x_keep, return_inverse=True)
            o_unique_indices, o_inverse_indices = torch.unique(y_keep, return_inverse=True)

            _stride = QWEN_PATCH_SIZE * QWEN_MERGE_SIZE  # 28
            img_H   = QWEN_IMAGE_SIZE  # 448
            img_W   = QWEN_IMAGE_SIZE  # 448

            snap_H = (img_H // _stride) * _stride  # 448
            snap_W = (img_W // _stride) * _stride  # 448
            grid_h = snap_H // _stride              # 16
            grid_w = snap_W // _stride              # 16

            pairwise_spatial = compute_spatial_encodings(
                [boxes[x.flatten()], ], [boxes[y.flatten()], ], [(img_H, img_W), ]
            ).to(torch.bfloat16)
            pairwise_spatial          = self.spatial_head(pairwise_spatial)
            pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)[x_keep, y_keep]

            gt_bx_h = self.recover_boxes(targets[0]['boxes_h'], targets[0]['size'])
            gt_bx_o = self.recover_boxes(targets[0]['boxes_o'], targets[0]['size'])

            bbox_2_tokens = bbox_to_token(
                (img_H, img_W), boxes, grid_h, grid_w
            )
            bool_h     = bbox_2_tokens[x_keep]
            bool_o     = bbox_2_tokens[y_keep]
            bool_union = bool_h | bool_o

            # ------------------------------------------------------------------
            # [JOINT] Phase 1 (frozen): get full-sequence inputs_embeds via hook
            # ------------------------------------------------------------------
            with torch.no_grad():
                _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
                _CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
                pil_img = tensor_to_pil(
                    image.decompose()[0][b_idx:b_idx + 1].cpu(),
                    mean=_CLIP_MEAN, std=_CLIP_STD,
                )
                inputs_embeds, img_start, img_end, _ = get_qwen_inputs_embeds(
                    self.clip_head, pil_img, (snap_H, snap_W)
                )
            # inputs_embeds : [1, seq_len, 2048]  (on Qwen's device, detached)

            x_boxes     = boxes[x_keep]
            y_boxes     = boxes[y_keep]
            x1 = torch.min(x_boxes[:, 0], y_boxes[:, 0])
            y1 = torch.min(x_boxes[:, 1], y_boxes[:, 1])
            x2 = torch.max(x_boxes[:, 2], y_boxes[:, 2])
            y2 = torch.max(x_boxes[:, 3], y_boxes[:, 3])
            union_boxes = torch.stack([x1, y1, x2, y2], dim=1)

            def _roi_boxes(idx_tensor):
                bx   = boxes[idx_tensor]
                bidx = torch.zeros(bx.size(0), dtype=torch.long, device=bx.device)
                return torch.cat([bidx[:, None].float(), bx], dim=1)

            roi_boxes  = _roi_boxes(h_unique_indices)
            roi_boxes1 = _roi_boxes(o_unique_indices)
            roi_boxes2 = torch.cat([
                torch.zeros(union_boxes.size(0), 1, dtype=union_boxes.dtype, device=union_boxes.device),
                union_boxes
            ], dim=1)

            # ------------------------------------------------------------------
            # [JOINT] ROI align on image-token slice of inputs_embeds
            # ------------------------------------------------------------------
            img_embeds = inputs_embeds[:, img_start:img_end, :]  # [1, N_img, 2048]
            llava_feat_for_roi = (
                img_embeds.float()
                .view(1, grid_h, grid_w, QWEN_HIDDEN_SIZE)
                .permute(0, 3, 1, 2)
            )  # [1, 2048, grid_h, grid_w]

            _scale = 1.0 / _stride  # 1/28

            h_feats0  = torchvision.ops.roi_align(
                llava_feat_for_roi, roi_boxes,
                output_size=(7, 7), spatial_scale=_scale, aligned=True
            )
            o_feats0  = torchvision.ops.roi_align(
                llava_feat_for_roi, roi_boxes1,
                output_size=(7, 7), spatial_scale=_scale, aligned=True
            )
            ho_feats0 = torchvision.ops.roi_align(
                llava_feat_for_roi, roi_boxes2,
                output_size=(7, 7), spatial_scale=_scale, aligned=True
            )

            ho_feats0 = ho_feats0.flatten(2).mean(-1).unsqueeze(0).to(img_embeds.dtype)
            h_feats0  = h_feats0.flatten(2).mean(-1).unsqueeze(0).to(img_embeds.dtype)
            o_feats0  = o_feats0.flatten(2).mean(-1).unsqueeze(0).to(img_embeds.dtype)

            # ------------------------------------------------------------------
            # [JOINT] Build region token embeddings and concatenate into sequence
            # ------------------------------------------------------------------
            region_embeds = (
                ho_feats0                                                         # [1, N_region, 2048]
                + pairwise_spatial_reshaped.unsqueeze(0)
                + self.object_embedding[labels[o_inverse_indices]].to(torch.bfloat16).unsqueeze(0)
                + self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16).unsqueeze(0)
            )  # [1, N_region, 2048]

            N_region = region_embeds.shape[1]

            # Move inputs_embeds to training device if needed
            combined = torch.cat(
                [inputs_embeds.to(region_embeds.device), region_embeds], dim=1
            )  # [1, seq_len + N_region, 2048]

            # ------------------------------------------------------------------
            # [JOINT] Phase 2 (trainable): single self-attend pass with LoRA
            # ------------------------------------------------------------------
            joint_out = self.joint_lora(combined)           # [1, seq+N, 2048]
            normed_ho = joint_out[0, -N_region:, :]         # [N_region, 2048]

            # ------------------------------------------------------------------
            # lm_head projection — identical to ours_qwen_new_old.py
            # ------------------------------------------------------------------
            lm_head_t = (
                self.lm_head_embeddings.T
                .to(normed_ho.device)
                .detach()
            )  # [QWEN_HIDDEN_SIZE, vocab_size]

            ho_vocab = normed_ho @ lm_head_t                # [N_region, vocab_size]

            all_probs_ho = F.softmax(ho_vocab, dim=-1)
            ho_vocab_weighted_mean = (all_probs_ho * ho_vocab).sum(-1)

            if self.num_classes == 117:
                form_counts = (
                    self.verb_token_mask.sum(dim=-1).unsqueeze(0).clamp(min=1)
                )  # [1, 117, 6]

                def _verb_avg(vocab):
                    scores_ = vocab[:, self.verb_token_ids]
                    scores_ = scores_.masked_fill(
                        ~self.verb_token_mask.unsqueeze(0), 0.0
                    )
                    return (scores_.sum(dim=-1) / form_counts).max(dim=-1).values

                ho_verb_avg = _verb_avg(ho_vocab)
                probs_ho    = torch.sigmoid(ho_verb_avg - ho_vocab_weighted_mean.unsqueeze(-1))
                logits      = probs_ho

            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(
                self.compute_prior_scores(x_keep, y_keep, scores, labels)
            )
            all_logits_collated.append(logits)

        return (
            all_logits_collated,
            prior_collated,
            boxes_h_collated,
            boxes_o_collated,
            object_class_collated,
        )

    # -----------------------------------------------------------------------
    def recover_boxes(self, boxes, size):
        boxes     = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w      = size
        scale_fct = torch.stack([w, h, w, h])
        return boxes * scale_fct

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets):
        n      = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])

        x, y = torch.nonzero(
            torch.min(box_iou(boxes_h, gt_bx_h), box_iou(boxes_o, gt_bx_o))
            >= self.fg_iou_thresh
        ).unbind(1)

        if self.num_classes in (117, 24, 407):
            labels[x, targets['labels'][y]] = 1
        else:
            labels[x, targets['hoi'][y]] = 1
        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets):
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])
        prior  = torch.cat(prior, dim=1).prod(0)
        x, y   = torch.nonzero(prior).unbind(1)
        logits = torch.cat(logits)
        logits = logits[x, y]
        prior  = prior[x, y]
        labels = labels[x, y]

        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss(
            prior * logits, labels,
            reduction='sum', alpha=self.alpha, gamma=self.gamma,
        )
        return loss / n_p

    def prepare_region_proposals(self, results):
        region_props = []
        for res in results:
            sc, lb, bx, feat = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc   = sc[keep].view(-1)
            lb   = lb[keep].view(-1)
            bx   = bx[keep].view(-1, 4)
            feat = feat[keep].view(-1, 256)

            keep     = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)
            is_human = lb == self.human_idx
            hum      = torch.nonzero(is_human).squeeze(1)
            obj      = torch.nonzero(is_human == 0).squeeze(1)
            n_human  = is_human[keep].sum()
            n_object = len(keep) - n_human

            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = keep[torch.nonzero(is_human[keep]).squeeze(1)]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = keep[torch.nonzero(is_human[keep] == 0).squeeze(1)]

            keep = torch.cat([keep_h, keep_o])
            region_props.append(dict(
                boxes=bx[keep], scores=sc[keep], labels=lb[keep], feat=feat[keep]
            ))
        return region_props

    # -----------------------------------------------------------------------
    def forward(
        self,
        images:  List[Tensor],
        targets: Optional[List[dict]] = None,
    ) -> List[dict]:

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images_orig  = [im[0].float() for im in images]
        images_clip  = [im[1] for im in images]

        device       = images_clip[0].device
        image_sizes  = torch.as_tensor(
            [im.size()[-2:] for im in images_clip], device=device
        )
        image_sizes_orig = torch.as_tensor(
            [im.size()[-2:] for im in images_orig], device=device
        )
        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)

        self.detector.eval()
        features, pos = self.detector.backbone(images_orig.to(device))
        src, mask     = features[-1].decompose()
        hs, detr_memory = self.detector.transformer(
            self.detector.input_proj(src), mask,
            self.detector.query_embed.weight, pos[-1],
        )
        outputs_class = self.detector.class_embed(hs)
        outputs_coord = self.detector.bbox_embed(hs).sigmoid()

        if self.dataset == 'vcoco' and outputs_class.shape[-1] == 92:
            outputs_class = outputs_class[:, :, :, self.reserve_indices]
            assert outputs_class.shape[-1] == 81

        results      = {
            'pred_logits': outputs_class[-1],
            'pred_boxes':  outputs_coord[-1],
            'feats':       hs[-1],
        }
        results      = self.postprocessor(results, image_sizes)
        region_props = self.prepare_region_proposals(results)
        images_clip  = nested_tensor_from_tensor_list(images_clip)

        logits, prior, bh, bo, objects = self.compute_sim_scores(
            region_props, images_clip, targets, None, None,
        )
        boxes = [r['boxes'] for r in region_props]

        if self.training:
            interaction_loss = self.compute_interaction_loss(
                boxes, bh, bo, logits, prior, targets
            )
            return dict(interaction_loss=interaction_loss)

        if len(logits) == 0:
            print(targets)
            return None

        return self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes):
        n      = [len(b) for b in bh]
        logits = torch.cat(logits).split(n)

        detections = []
        for bx, h, o, lg, pr, obj, size in zip(
            boxes, bh, bo, logits, prior, objects, image_sizes
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = lg[x, y]
            detections.append(dict(
                boxes=bx,
                pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y],
                labels=y,
                objects=obj[x],
                size=size,
            ))
        return detections


# ---------------------------------------------------------------------------
# Object embedding helper
# ---------------------------------------------------------------------------

def compute_object_embeddings_qwen(model_state, obj_class_names, device="cpu"):
    """
    Compute per-class object embeddings from Qwen's LM head.
    Returns [num_classes, QWEN_HIDDEN_SIZE].
    """
    tokenizer = model_state["tokenizer"]
    lm_head   = model_state["model"].lm_head.weight.detach().float()

    embs = []
    for name in obj_class_names:
        ids = tokenizer.encode(name, add_special_tokens=False)
        if ids:
            embs.append(lm_head[ids].mean(0))
        else:
            embs.append(torch.zeros(QWEN_HIDDEN_SIZE, dtype=torch.float))

    return torch.stack(embs).to(torch.bfloat16).to(device)


# ---------------------------------------------------------------------------
# build_detector
# ---------------------------------------------------------------------------

def build_detector(args, class_corr, object_n_verb_to_interaction,
                   clip_model_path, rank):
    num_classes = 80
    if args.dataset == 'vcoco' and 'e632da11' in args.pretrained:
        num_classes = 91

    backbone    = build_backbone(args)
    transformer = build_transformer(args)
    detr        = DETR(
        backbone, transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    postprocessors = {'bbox': PostProcess()}

    if os.path.exists(args.pretrained):
        print(f"Load weights for the object detector from {args.pretrained}")
        if 'e632da11' in args.pretrained:
            detr.load_state_dict(
                torch.load(args.pretrained, map_location='cpu')['model']
            )
        else:
            detr.load_state_dict(
                torch.load(args.pretrained, map_location='cpu')['model_state_dict']
            )

    model = load_qwen_state(rank)

    if args.num_classes == 117:
        classnames = hico_verbs_sentence
    elif args.num_classes == 24:
        classnames = vcoco_verbs_sentence
    elif args.num_classes == 600:
        classnames = list(hico_text_label.hico_text_label.values())
    else:
        raise NotImplementedError

    obj_class_names  = [obj[1] for obj in hico_text_label.hico_obj_text_label]
    object_embedding = compute_object_embeddings_qwen(
        model, obj_class_names, device=f"cuda:{rank}"
    )
    object_embedding = object_embedding.clone().detach()

    detector = HOIQWEN_Joint(
        args,
        detr, postprocessors['bbox'], model, object_embedding,
        human_idx=args.human_idx,
        num_classes=args.num_classes,
        alpha=args.alpha,
        gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        object_class_to_target_class=class_corr,
        object_n_verb_to_interaction=object_n_verb_to_interaction,
    )
    return detector
