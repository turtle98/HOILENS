"""
Unary-pairwise transformer for human-object interaction detection
using Qwen2.5-VL-3B as the vision-language backbone.

Analogous to ours_llava_new_old.py — all Qwen-specific changes are
marked with  # [QWEN]  comments so diffs are easy to follow.

Key changes vs. ours_llava_new_old.py:
  - visual_output_dim  : 4096 → QWEN_HIDDEN_SIZE (2048)
  - image grid         : 24×24 → QWEN_GRID_H × QWEN_GRID_W (12×12)
  - spatial_scale      : 24/336 → QWEN_GRID_H/QWEN_IMAGE_SIZE
  - start_layer        : 30 → QWEN_START_LAYER (26)
  - CrossAttendWithLoRA: MHA → GQA (16 Q / 8 KV heads)
  - image input        : pre-processed tensor → PIL Image via tensor_to_pil()
  - LM head / obj embs : loaded from Qwen model at init time
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

sys.path.insert(0, 'detr')
from detr.models.backbone import build_backbone
from detr.models.transformer import build_transformer
from detr.models.detr import DETR
from detr.util import box_ops
from detr.util.misc import nested_tensor_from_tensor_list
sys.path.pop(0)

from transformers import CLIPVisionModel, CLIPImageProcessor, AutoConfig

# [QWEN] — import Qwen utils instead of llava_utils
from methods.qwen_utils import (
    load_qwen_state,
    run_qwen_model,
    generate_qwen_model,
    compute_conditional_likelihood_qwen,
    retrieve_logit_lens_qwen,
    get_img_idx_qwen,
    tensor_to_pil,
    get_lm_head_embeddings_qwen,
    _get_qwen_grid,
    QWEN_HIDDEN_SIZE,      # 2048
    QWEN_NUM_LAYERS,       # 28
    QWEN_GRID_H,           # 12
    QWEN_GRID_W,           # 12
    QWEN_NUM_IMAGE_TOKENS, # 144
    QWEN_IMAGE_SIZE,       # 336
    QWEN_PATCH_SIZE,       # 14
    QWEN_MERGE_SIZE,       # 2
    QWEN_START_LAYER,      # 26
    QWEN_NUM_HEADS,        # 16
    QWEN_NUM_KV_HEADS,     # 8
    QWEN_HEAD_DIM,         # 128
    QWEN_KV_HIDDEN_SIZE,   # 1024
)
from methods.attention import llama_modify


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
# CrossAttendWithLoRA  — updated for Qwen GQA
# ---------------------------------------------------------------------------

class CrossAttendWithLoRA(nn.Module):
    """Cross-attention using cloned Qwen2 layers with LoRA adapters.

    [QWEN] Qwen2 uses GQA: num_key_value_heads (8) < num_heads (16).
           k_proj / v_proj output dim = QWEN_KV_HIDDEN_SIZE (1024) not 2048.
           K and V are expanded with repeat_interleave to match Q head count.
    """

    def __init__(self, cloned_layers, lora_rank=16, lora_alpha=16):
        super().__init__()
        self.cloned_layers = cloned_layers

        for layer in self.cloned_layers:
            for param in layer.parameters():
                param.requires_grad = False

        self.lora_q = nn.ModuleList()
        self.lora_k = nn.ModuleList()
        self.lora_v = nn.ModuleList()
        self.lora_o = nn.ModuleList()

        for layer in cloned_layers:
            attn   = layer.self_attn
            q_dim  = attn.q_proj.weight.shape[0]   # 2048  [QWEN]
            kv_dim = attn.k_proj.weight.shape[0]   # 1024  [QWEN] GQA

            # [QWEN] k / v LoRA take (hidden → kv_dim), not (hidden → hidden)
            self.lora_q.append(LoRALinear(q_dim, q_dim,  lora_rank, lora_alpha))
            self.lora_k.append(LoRALinear(q_dim, kv_dim, lora_rank, lora_alpha))
            self.lora_v.append(LoRALinear(q_dim, kv_dim, lora_rank, lora_alpha))
            self.lora_o.append(LoRALinear(q_dim, q_dim,  lora_rank, lora_alpha))

    def forward(self, queries, context, roi_mask=None, return_attn_weights=False):
        hidden_states    = queries
        attn_weights_all = []

        for idx, layer in enumerate(self.cloned_layers):
            attn              = layer.self_attn
            bsz, q_len, _    = hidden_states.shape
            ctx               = context[idx] if isinstance(context, list) else context
            kv_len            = ctx.shape[1]

            residual  = hidden_states
            normed    = layer.input_layernorm(hidden_states)
            normed_kv = layer.input_layernorm(ctx)

            q = self.lora_q[idx](normed,    attn.q_proj)   # [bsz, q_len,  q_dim]
            k = self.lora_k[idx](normed_kv, attn.k_proj)   # [bsz, kv_len, kv_dim]
            v = self.lora_v[idx](normed_kv, attn.v_proj)   # [bsz, kv_len, kv_dim]

            head_dim     = attn.head_dim                    # 128  [QWEN]
            num_heads    = attn.num_heads                   # 16   [QWEN]
            num_kv_heads = attn.num_key_value_heads         # 8    [QWEN]
            groups       = num_heads // num_kv_heads        # 2    [QWEN]

            q = q.view(bsz, q_len,  num_heads,    head_dim).transpose(1, 2)  # [bsz, 16, q_len, 128]
            k = k.view(bsz, kv_len, num_kv_heads, head_dim).transpose(1, 2)  # [bsz,  8, kv_len, 128]
            v = v.view(bsz, kv_len, num_kv_heads, head_dim).transpose(1, 2)  # [bsz,  8, kv_len, 128]

            # [QWEN] expand KV to match Q head count (standard GQA)
            k = k.repeat_interleave(groups, dim=1)   # [bsz, 16, kv_len, 128]
            v = v.repeat_interleave(groups, dim=1)   # [bsz, 16, kv_len, 128]

            attn_mask = None
            if roi_mask is not None:
                attn_mask                = torch.zeros_like(roi_mask, dtype=q.dtype)
                attn_mask[~roi_mask]     = float('-inf')
                attn_mask                = attn_mask.unsqueeze(0).unsqueeze(0)

            if return_attn_weights:
                scale   = head_dim ** -0.5
                scores  = torch.matmul(q, k.transpose(-2, -1)) * scale
                if attn_mask is not None:
                    scores = scores + attn_mask
                attn_w  = F.softmax(scores, dim=-1)
                attn_weights_all.append(attn_w)
                attn_out = torch.matmul(attn_w, v)
            else:
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask)

            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_out = self.lora_o[idx](attn_out, attn.o_proj)

            hidden_states = residual + attn_out
            residual      = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        if return_attn_weights:
            return hidden_states, attn_weights_all
        return hidden_states


# ---------------------------------------------------------------------------
# HOIQWEN  (was HOILLAVA)
# ---------------------------------------------------------------------------

class HOIQWEN(nn.Module):
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

        # [QWEN] hidden dim is 2048, not 4096
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

        # [QWEN] pull LM-head weights directly from the model
        qwen_model = self.clip_head["model"]
        self.lm_head_embeddings = qwen_model.lm_head.weight.detach().cpu()
        # shape: [vocab_size, QWEN_HIDDEN_SIZE]

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
                ('brush with',     'brushing with',  'brushed with'),
                ('buy',            'buying',         'bought'),
                ('carry',          'carrying',       'carried'),
                ('catch',          'catching',       'caught'),
                ('chase',          'chasing',        'chased'),
                ('check',          'checking',       'checked'),
                ('clean',          'cleaning',       'cleaned'),
                ('control',        'controlling',    'controlled'),
                ('cook',           'cooking',        'cooked'),
                ('cut',            'cutting',        'cut'),
                ('cut with',       'cutting with',   'cut with'),
                ('direct',         'directing',      'directed'),
                ('drag',           'dragging',       'dragged'),
                ('dribble',        'dribbling',      'dribbled'),
                ('drink with',     'drinking with',  'drank with'),
                ('drive',          'driving',        'drove'),
                ('dry',            'drying',         'dried'),
                ('eat',            'eating',         'ate'),
                ('eat at',         'eating at',      'ate at'),
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
                ('hop on',         'hopping on',     'hopped on'),
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
                ('lie on',         'lying on',       'lay on'),
                ('lift',           'lifting',        'lifted'),
                ('light',          'lighting',       'lit'),
                ('load',           'loading',        'loaded'),
                ('lose',           'losing',         'lost'),
                ('make',           'making',         'made'),
                ('milk',           'milking',        'milked'),
                ('move',           'moving',         'moved'),
                ('no interaction', 'no interaction', 'no interaction'),
                ('open',           'opening',        'opened'),
                ('operate',        'operating',      'operated'),
                ('pack',           'packing',        'packed'),
                ('paint',          'painting',       'painted'),
                ('park',           'parking',        'parked'),
                ('pay',            'paying',         'paid'),
                ('peel',           'peeling',        'peeled'),
                ('pet',            'petting',        'petted'),
                ('pick',           'picking',        'picked'),
                ('pick up',        'picking up',     'picked up'),
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
                ('sit at',         'sitting at',     'sat at'),
                ('sit on',         'sitting on',     'sat on'),
                ('slide',          'sliding',        'slid'),
                ('smell',          'smelling',       'smelled'),
                ('spin',           'spinning',       'spun'),
                ('squeeze',        'squeezing',      'squeezed'),
                ('stab',           'stabbing',       'stabbed'),
                ('stand on',       'standing on',    'stood on'),
                ('stand under',    'standing under', 'stood under'),
                ('stick',          'sticking',       'stuck'),
                ('stir',           'stirring',       'stirred'),
                ('stop at',        'stopping at',    'stopped at'),
                ('straddle',       'straddling',     'straddled'),
                ('swing',          'swinging',       'swung'),
                ('tag',            'tagging',        'tagged'),
                ('talk on',        'talking on',     'talked on'),
                ('teach',          'teaching',       'taught'),
                ('text on',        'texting on',     'texted on'),
                ('throw',          'throwing',       'threw'),
                ('tie',            'tying',          'tied'),
                ('toast',          'toasting',       'toasted'),
                ('train',          'training',       'trained'),
                ('turn',           'turning',        'turned'),
                ('type on',        'typing on',      'typed on'),
                ('walk',           'walking',        'walked'),
                ('wash',           'washing',        'washed'),
                ('watch',          'watching',       'watched'),
                ('wave',           'waving',         'waved'),
                ('wear',           'wearing',        'wore'),
                ('wield',          'wielding',       'wielded'),
                ('zip',            'zipping',        'zipped'),
            ]

            # Add space-prefixed forms to cover mid-sentence BPE tokenization
            # verb_forms_3 = [
            #     (base, pres, past, ' ' + base, ' ' + pres, ' ' + past)
            #     for base, pres, past in verb_forms_3
            # ]

            all_form_ids = []
            for forms in verb_forms_3:
                form_ids = []
                for form in forms:
                    # [QWEN] no BOS token in Qwen tokenizer — no [1:] skip
                    ids = tokenizer.encode(form, add_special_tokens=False)
                    ids = list(dict.fromkeys(ids))   # deduplicate
                    form_ids.append(ids)
                all_form_ids.append(form_ids)

            num_verbs  = len(verb_forms_3)
            num_forms  = len(verb_forms_3[0])
            max_tokens = max(
                len(ids) for form_ids in all_form_ids for ids in form_ids
            )
            padded = torch.zeros(num_verbs, num_forms, max_tokens, dtype=torch.long)
            mask   = torch.zeros(num_verbs, num_forms, max_tokens, dtype=torch.bool)
            for i, form_ids in enumerate(all_form_ids):
                for j, ids in enumerate(form_ids):
                    padded[i, j, :len(ids)] = torch.tensor(ids)
                    mask[i, j, :len(ids)]   = True
            self.register_buffer('verb_token_ids',  padded)
            self.register_buffer('verb_token_mask', mask)

        # ------------------------------------------------------------------
        # HOI token ids (600-class path)
        # ------------------------------------------------------------------
        if self.num_classes == 600:
            self.hoi_labels   = [
                v.replace("a photo of ", "")
                for k, v in hico_text_label.hico_text_label.items()
            ]
            # [QWEN] no [1:] skip
            hoi_token_ids = [
                tokenizer.encode(v, add_special_tokens=False)
                for v in self.hoi_labels
            ]
            max_hoi_tokens = max(len(ids) for ids in hoi_token_ids)
            hoi_padded = torch.zeros(
                len(self.hoi_labels), max_hoi_tokens, dtype=torch.long
            )
            hoi_mask = torch.zeros(
                len(self.hoi_labels), max_hoi_tokens, dtype=torch.bool
            )
            for i, ids in enumerate(hoi_token_ids):
                hoi_padded[i, :len(ids)] = torch.tensor(ids)
                hoi_mask[i, :len(ids)]   = True
            self.register_buffer('hoi_token_ids',  hoi_padded)
            self.register_buffer('hoi_token_mask', hoi_mask)

        # Zero-shot filtering
        if args.zs:
            self.zs_type        = args.zs_type
            self.filtered_hoi_idx = hico_text_label.hico_unseen_index[self.zs_type]
        else:
            self.filtered_hoi_idx = []
            self.zs_type          = None

        self.seen_verb_idxs = list(set([
            HOI_IDX_TO_ACT_IDX[idx]
            for idx in range(600)
            if idx not in self.filtered_hoi_idx
        ]))
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
        # [QWEN] Cross-attention: clone layers from Qwen (start_layer = 26)
        # ------------------------------------------------------------------
        self.start_layer      = 0 #QWEN_START_LAYER   # 26
        self.num_cross_layers = 36

        # cloned_layers_h = nn.ModuleList([
        #     copy.deepcopy(self.clip_head["model"].model.language_model.layers[i])
        #     for i in range(self.start_layer, self.start_layer + self.num_cross_layers)
        # ])
        # self.cross_attend_lora_h = CrossAttendWithLoRA(
        #     cloned_layers_h, lora_rank=8, lora_alpha=8
        # )

        # cloned_layers_o = nn.ModuleList([
        #     copy.deepcopy(self.clip_head["model"].model.language_model.layers[i])
        #     for i in range(self.start_layer, self.start_layer + self.num_cross_layers)
        # ])
        # self.cross_attend_lora_o = CrossAttendWithLoRA(
        #     cloned_layers_o, lora_rank=8, lora_alpha=8
        # )

        cloned_layers_ho = nn.ModuleList([
            copy.deepcopy(self.clip_head["model"].model.language_model.layers[i])
            for i in range(self.start_layer, self.start_layer + self.num_cross_layers)
        ])
        self.cross_attend_lora_ho = CrossAttendWithLoRA(
            cloned_layers_ho, lora_rank=8, lora_alpha=8
        )

        # Clone the final norm
        self.final_norm = copy.deepcopy(self.clip_head["model"].model.language_model.norm)
        for param in self.final_norm.parameters():
            param.requires_grad = False

        # ------------------------------------------------------------------
        # [QWEN] Query projection MLPs: output dim = QWEN_HIDDEN_SIZE (2048)
        # ------------------------------------------------------------------
        #self.ho_query_proj = MLP(512, 128, QWEN_HIDDEN_SIZE, 2).to(torch.bfloat16)
        #self.h_query_proj  = MLP(256, 128, QWEN_HIDDEN_SIZE, 2).to(torch.bfloat16)
        #self.o_query_proj  = MLP(256, 128, QWEN_HIDDEN_SIZE, 2).to(torch.bfloat16)

        # [QWEN] Spatial head: output dim = QWEN_HIDDEN_SIZE (2048)
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),  nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, QWEN_HIDDEN_SIZE), nn.ReLU(),
        ).to(torch.bfloat16)

        # # Projects concatenated [h, o] ROI features → QWEN_HIDDEN_SIZE, then added to ho_feats0
        # self.ho_fusion_mlp = nn.Sequential(
        #     nn.Linear(2 * QWEN_HIDDEN_SIZE, 256), nn.ReLU(),
        #     nn.Linear(256, QWEN_HIDDEN_SIZE)
        # ).to(torch.bfloat16)
        # self.fusion_scale = nn.Parameter(torch.tensor(0.01, dtype=torch.bfloat16))

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
        pil_images: list = None,   # [QWEN] list of PIL images for each batch item
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

            # ho_detr_feats = self.ho_query_proj(
            #     torch.cat([feats[x_keep], feats[y_keep]], dim=-1).to(torch.bfloat16)
            # )
            # h_detr_feats  = self.h_query_proj(feats[h_unique_indices].to(torch.bfloat16))
            # o_detr_feats  = self.o_query_proj(feats[o_unique_indices].to(torch.bfloat16))


            _stride = QWEN_PATCH_SIZE * QWEN_MERGE_SIZE  # 28
            img_H   = QWEN_IMAGE_SIZE  # 448
            img_W   = QWEN_IMAGE_SIZE  # 448

            # Snap to 28-pixel boundary (Qwen requirement)
            snap_H = (img_H // _stride) * _stride  # 448
            snap_W = (img_W // _stride) * _stride  # 448
            grid_h = snap_H // _stride  # 16
            grid_w = snap_W // _stride  # 16

            pairwise_spatial = compute_spatial_encodings(
                [boxes[x.flatten()], ], [boxes[y.flatten()], ], [(img_H, img_W), ]
            ).to(torch.bfloat16)
            pairwise_spatial         = self.spatial_head(pairwise_spatial)
            pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)[x_keep, y_keep]

            gt_bx_h = self.recover_boxes(targets[0]['boxes_h'], targets[0]['size'])
            gt_bx_o = self.recover_boxes(targets[0]['boxes_o'], targets[0]['size'])

            # token mask — grid_h × grid_w (may be non-square at native resolution)
            bbox_2_tokens = bbox_to_token(
                (img_H, img_W), boxes, grid_h, grid_w
            )
            bool_h     = bbox_2_tokens[x_keep]
            bool_o     = bbox_2_tokens[y_keep]
            bool_union = bool_h | bool_o

            # Build PIL from the DETR-normalised tensor (ImageNet stats).
            # Crop out the unpadded region before passing to Qwen.
            with torch.no_grad():
                _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
                _CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
                pil_img = tensor_to_pil(
                    image.decompose()[0][b_idx:b_idx + 1].cpu(),
                    mean=_CLIP_MEAN, std=_CLIP_STD,
                )
                hidden_states, output, _, _ = run_qwen_model(
                    self.clip_head,
                    self.clip_head['model_name'],
                    pil_img,
                    (snap_H, snap_W),         
                    self.clip_head['tokenizer'],
                    hidden_states=True,
                    text_prompt=".",
                )
            #import pdb; pdb.set_trace()
            x_boxes   = boxes[x_keep]
            y_boxes   = boxes[y_keep]
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

            # Extract hidden states and reshape to native grid_h × grid_w
            # spatial_scale = 1/28 is always correct: grid_h/snap_H = 1/_stride
            llava_features = hidden_states[self.start_layer]  # [1, grid_h*grid_w, 2048]
            #import pdb; pdb.set_trace()
            llava_feat_for_roi  = (
                llava_features.float()
                .view(1, grid_h, grid_w, QWEN_HIDDEN_SIZE)
                .permute(0, 3, 1, 2)
            )   # [1, 2048, grid_h, grid_w]

            _scale = 1.0 / _stride  # = 1/28, exact at any native resolution
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

            ho_feats0 = ho_feats0.flatten(2).mean(-1).unsqueeze(0).to(llava_features.dtype)
            h_feats0  = h_feats0.flatten(2).mean(-1).unsqueeze(0).to(llava_features.dtype)
            o_feats0  = o_feats0.flatten(2).mean(-1).unsqueeze(0).to(llava_features.dtype)

            # Fuse h and o features and add to ho
            #import pdb; pdb.set_trace()
           # ho_feats0 = ho_feats0 + self.fusion_scale * self.ho_fusion_mlp(torch.cat([h_feats0[0][h_inverse_indices], o_feats0[0][o_inverse_indices]], dim=-1))

            llava_context_list = [
                hidden_states[self.start_layer + i]
                for i in range(self.num_cross_layers)
            ]

            ho_out = self.cross_attend_lora_ho(
                ho_feats0 
                + pairwise_spatial_reshaped
                + self.object_embedding[labels[o_inverse_indices]].to(torch.bfloat16)
                + self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16),
                llava_context_list,
            )

            #normed_h  = self.final_norm(h_out.squeeze(0))
            #normed_o  = self.final_norm(o_out.squeeze(0))
            normed_ho = self.final_norm(ho_out.squeeze(0))

            lm_head_t = (
                self.lm_head_embeddings.T
                .to(hidden_states.device)
                .detach()
            )   # [QWEN_HIDDEN_SIZE, vocab_size]

            #h_vocab  = normed_h  @ lm_head_t
            #o_vocab  = normed_o  @ lm_head_t
            ho_vocab = normed_ho @ lm_head_t

            all_probs_ho = F.softmax(ho_vocab, dim=-1)
            #all_probs_h  = F.softmax(h_vocab,  dim=-1)
            #all_probs_o  = F.softmax(o_vocab,  dim=-1)

            ho_vocab_weighted_mean = (all_probs_ho * ho_vocab).sum(-1)
            #h_vocab_weighted_mean  = (all_probs_h  * h_vocab).sum(-1)
            #o_vocab_weighted_mean  = (all_probs_o  * o_vocab).sum(-1)

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

                #h_verb_avg  = _verb_avg(h_vocab)
                #o_verb_avg  = _verb_avg(o_vocab)
                ho_verb_avg = _verb_avg(ho_vocab)

                probs_ho = torch.sigmoid(ho_verb_avg - ho_vocab_weighted_mean.unsqueeze(-1))
                #probs_h  = torch.sigmoid(h_verb_avg  - h_vocab_weighted_mean.unsqueeze(-1)).clamp(1e-6, 1 - 1e-6)
                #probs_o  = torch.sigmoid(o_verb_avg  - o_vocab_weighted_mean.unsqueeze(-1)).clamp(1e-6, 1 - 1e-6)
                logits   = (probs_ho) #/ 3

            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(
                self.compute_prior_scores(x_keep, y_keep, scores, labels)
            )
            all_logits_collated.append(logits)
            #import pdb; pdb.set_trace()
            #print(self.clip_head['tokenizer'].decode(torch.topk(ho_vocab[0], 100, dim=-1)[1]))
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

        images_orig  = [im[0].float() for im in images]   # DETR input
        images_clip  = [im[1] for im in images]            # normalised clip tensor

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

    Uses the mean LM-head row across each class name's tokens,
    giving a [num_classes, QWEN_HIDDEN_SIZE] tensor analogous to
    obj_classifier_7b.pt (which was computed from LLaVA 7B).
    """
    tokenizer = model_state["tokenizer"]
    lm_head   = model_state["model"].lm_head.weight.detach().float()  # [vocab, 2048]

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
    # Build DETR backbone
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

    # [QWEN] load Qwen2.5-VL-3B instead of LLaVA
    model = load_qwen_state(rank)

    if args.num_classes == 117:
        classnames = hico_verbs_sentence
    elif args.num_classes == 24:
        classnames = vcoco_verbs_sentence
    elif args.num_classes == 600:
        classnames = list(hico_text_label.hico_text_label.values())
    else:
        raise NotImplementedError

    # [QWEN] compute object embeddings from Qwen LM head
    obj_class_names  = [obj[1] for obj in hico_text_label.hico_obj_text_label]
    object_embedding = compute_object_embeddings_qwen(
        model, obj_class_names, device=f"cuda:{rank}"
    )

    #import pdb; pdb.set_trace()
    object_embedding = object_embedding.clone().detach()



    detector = HOIQWEN(
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