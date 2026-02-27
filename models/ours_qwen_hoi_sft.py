"""
ours_qwen_hoi_sft.py — HOI detection via dual BCE + SFT generation loss on
Qwen2.5-VL-3B with PEFT LoRA.

Architecture summary
--------------------
Prompt fed to the model (question part, with add_generation_prompt=True):
  <|im_start|>system\n...<|im_end|>\n
  <|im_start|>user\n
  <|vision_start|><|image_pad|>×N<|vision_end|>
  The interaction features are <hoi_feat>×M.
  Provide all human-object interactions in this image.
  <|im_end|>\n<|im_start|>assistant\n

Answer part (teacher-forced during training):
  person VERB1 OBJ1, person VERB2 OBJ2, ...<|im_end|>

Key design decisions
--------------------
1. PEFT LoRA on q_proj/k_proj/v_proj/o_proj of the language model (r=8).
2. <hoi_feat> positions in inputs_embeds are replaced with ROI-aligned H-O
   features computed by a trainable spatial_head (same as ours_qwen_new_old).
3. 4D causal attention mask with an additional block:
     mask[q_len:, img_start:img_end] = -inf
   so that generation tokens CANNOT attend to raw image tokens (they must go
   through the <hoi_feat> bottleneck).
4. BCE focal loss on hidden[hoi_feat_positions] → lm_head → verb_avg scores.
5. Cross-entropy generation loss on hidden[q_len-1:q_len+a_len-1].
6. Total loss = bce_loss + sft_loss_weight * gen_loss.
"""

import os
import sys
import copy
import math
from collections import defaultdict
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torchvision.ops.boxes import batched_nms, box_iou
import torchvision

from peft import LoraConfig, get_peft_model

from utils.hico_list import hico_verbs_sentence, hico_verb_object_list, hico_verbs, hico_objects
from utils.vcoco_list import vcoco_verbs_sentence
from utils.hico_utils import reserve_indices, HOI_IDX_TO_ACT_IDX
from utils.postprocessor import PostProcess
from utils.ops import (
    binary_focal_loss_with_logits, binary_focal_loss,
    vectorized_bboxes_and_indices, bbox_to_token, compute_spatial_encodings,
)
from utils import hico_text_label

sys.path.insert(0, 'detr')
from detr.models.backbone import build_backbone
from detr.models.transformer import build_transformer
from detr.models.detr import DETR
from detr.util import box_ops
from detr.util.misc import nested_tensor_from_tensor_list
sys.path.pop(0)

from methods.qwen_utils import (
    load_qwen_state,
    tensor_to_pil,
    _build_qwen_inputs,
    _get_image_token_range,
    QWEN_HIDDEN_SIZE,
    QWEN_GRID_H, QWEN_GRID_W,
    QWEN_IMAGE_SIZE, QWEN_PATCH_SIZE, QWEN_MERGE_SIZE,
    QWEN_NUM_IMAGE_TOKENS,
    QWEN_START_LAYER, QWEN_NUM_HEADS, QWEN_NUM_KV_HEADS,
    QWEN_HEAD_DIM, QWEN_KV_HIDDEN_SIZE,
)
from methods.hoi_prompt_utils import (
    HOI_FEAT_TOKEN,
    build_hoi_text_prompt,
    build_target_text,
    get_hoi_token_id,
)


# ---------------------------------------------------------------------------
# Object embedding helper (copied from ours_qwen_new_old)
# ---------------------------------------------------------------------------

def compute_object_embeddings_qwen(model_state, obj_class_names, device="cpu"):
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
# Main model
# ---------------------------------------------------------------------------

class HOIQWEN_SFT(nn.Module):
    """
    HOI detector using Qwen2.5-VL-3B + PEFT LoRA with dual BCE + SFT loss.

    Parameters mirror HOIQWEN in ours_qwen_new_old.py.  New parameters:
      sft_loss_weight  : weight for the generation loss (default 1.0)
      lora_rank        : LoRA rank for q/k/v/o projections (default 8)
    """

    def __init__(
        self,
        args,
        detector:       nn.Module,
        postprocessor:  nn.Module,
        model_state:    dict,          # from load_qwen_state()
        object_embedding: torch.Tensor,
        human_idx:      int,
        num_classes:    int,
        alpha:          float = 0.5,
        gamma:          float = 2.0,
        box_score_thresh:  float = 0.2,
        fg_iou_thresh:     float = 0.5,
        min_instances:     int   = 3,
        max_instances:     int   = 15,
        object_class_to_target_class: List[list] = None,
        object_n_verb_to_interaction: List[list] = None,
        sft_loss_weight: float = 1.0,
        lora_rank:       int   = 8,
    ) -> None:
        super().__init__()

        self.detector      = detector
        self.postprocessor = postprocessor

        self.register_buffer("object_embedding", object_embedding)

        self.visual_output_dim           = QWEN_HIDDEN_SIZE
        self.object_n_verb_to_interaction = np.asarray(
            object_n_verb_to_interaction, dtype=float
        )
        self.args          = args
        self.human_idx     = human_idx
        self.num_classes   = num_classes
        self.hyper_lambda  = args.hyper_lambda
        self.alpha         = alpha
        self.gamma         = gamma
        self.box_score_thresh            = box_score_thresh
        self.fg_iou_thresh               = fg_iou_thresh
        self.min_instances               = min_instances
        self.max_instances               = max_instances
        self.object_class_to_target_class = object_class_to_target_class
        self.dataset                     = args.dataset
        self.reserve_indices             = reserve_indices
        self.sft_loss_weight             = sft_loss_weight

        # ------------------------------------------------------------------
        # Qwen model + processor + tokenizer
        # ------------------------------------------------------------------
        qwen_model = model_state["model"]     # Qwen2_5_VLForConditionalGeneration
        self.processor  = model_state["processor"]
        self.tokenizer  = model_state["tokenizer"]

        # Grab the VLM reference BEFORE PEFT modifies the linear layers
        self.vlm = qwen_model.model           # Qwen2_5_VLModel

        # Save frozen lm_head weight for BCE scoring and gen loss projection
        # (lm_head is NOT in LoRA targets so it stays frozen throughout)
        self.register_buffer(
            "lm_head_weight",
            qwen_model.lm_head.weight.detach().cpu().float(),
        )  # [vocab_size, QWEN_HIDDEN_SIZE]

        # ------------------------------------------------------------------
        # Register <hoi_feat> special token
        # ------------------------------------------------------------------
        if HOI_FEAT_TOKEN not in self.tokenizer.all_special_tokens:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [HOI_FEAT_TOKEN]}
            )
            qwen_model.resize_token_embeddings(len(self.tokenizer))

        self.hoi_feat_token_id = get_hoi_token_id(self.tokenizer)

        # ------------------------------------------------------------------
        # Apply PEFT LoRA to the language model layers
        # (visual encoder and lm_head are excluded automatically because
        #  "q_proj", "k_proj", "v_proj", "o_proj" do not appear in their paths)
        # ------------------------------------------------------------------
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            lora_dropout=0.0,
        )
        # get_peft_model modifies qwen_model in-place (replaces linears)
        # and wraps it in a PeftModel container.
        self.peft_qwen = get_peft_model(qwen_model, peft_config)
        # self.vlm is still valid — same object, now with LoRA-wrapped linears

        # Freeze everything except LoRA adapters
        for name, p in self.peft_qwen.named_parameters():
            p.requires_grad = ("lora_" in name)

        # ------------------------------------------------------------------
        # Spatial encoding MLP (trainable) — same as ours_qwen_new_old
        # ------------------------------------------------------------------
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),  nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, QWEN_HIDDEN_SIZE), nn.ReLU(),
        ).to(torch.bfloat16)

        self.ho_fusion_mlp = nn.Sequential(
            nn.Linear(2 * QWEN_HIDDEN_SIZE, 256), nn.ReLU(),
            nn.Linear(256, QWEN_HIDDEN_SIZE)
        ).to(torch.bfloat16)
        self.fusion_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.bfloat16))
        # ------------------------------------------------------------------
        # Verb / HOI token buffers for BCE scoring — copied from HOIQWEN
        # ------------------------------------------------------------------
        tokenizer = self.tokenizer

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

            # verb_forms_3 = [
            #     (base, pres, past, ' ' + base, ' ' + pres, ' ' + past)
            #     for base, pres, past in verb_forms_3
            # ]

            all_form_ids = []
            for forms in verb_forms_3:
                form_ids = []
                for form in forms:
                    ids = tokenizer.encode(form, add_special_tokens=False)
                    ids = list(dict.fromkeys(ids))
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

            verb_forms_3 = [
                (base, pres, past, ' ' + base, ' ' + pres, ' ' + past)
                for base, pres, past in verb_forms_3
            ]

            all_form_ids = []
            for forms in verb_forms_3:
                form_ids = []
                for form in forms:
                    ids = tokenizer.encode(form, add_special_tokens=False)
                    ids = list(dict.fromkeys(ids))
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
            self.register_buffer('verb_token_ids_test',  padded)
            self.register_buffer('verb_token_mask_test', mask)

            
        if self.num_classes == 600:
            self.hoi_labels = [
                v.replace("a photo of ", "")
                for k, v in hico_text_label.hico_text_label.items()
            ]
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
            self.zs_type              = args.zs_type
            self.filtered_hoi_idx     = hico_text_label.hico_unseen_index[self.zs_type]
        else:
            self.filtered_hoi_idx     = []
            self.zs_type              = None

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

    # -----------------------------------------------------------------------
    # Qwen inputs builder (SFT version — returns input_ids for position finding)
    # -----------------------------------------------------------------------

    def _build_sft_question_embeds(self, pil_img, snap_H, snap_W, n_pairs):
        """
        Build question inputs_embeds and related tensors for the SFT forward.

        Returns
        -------
        inputs_embeds  : Tensor [1, q_len, 2048]  (detached, no grad)
        position_ids   : Tensor [3, 1, q_len]     (detached)
        attn_mask_1d   : Tensor [1, q_len]         (1=valid, 0=pad)
        img_start      : int  — first image_pad index in q input_ids
        img_end        : int  — one past last image_pad index
        hoi_positions  : Tensor [n_pairs]  — indices of <hoi_feat> tokens
        q_len          : int  — total question sequence length
        """
        device = next(self.peft_qwen.parameters()).device
        text_prompt = build_hoi_text_prompt(n_pairs)

        with torch.no_grad():
            inputs         = _build_qwen_inputs(
                self.processor, pil_img, text_prompt, device, snap_H, snap_W
            )
            input_ids      = inputs["input_ids"]
            pixel_values   = inputs.get("pixel_values")
            image_grid_thw = inputs.get("image_grid_thw")
            attn_mask_1d   = inputs.get(
                "attention_mask", torch.ones_like(input_ids)
            )

            # Step 1: text embeddings
            inputs_embeds = self.vlm.get_input_embeddings()(input_ids)

            # Step 2: merge ViT image features
            if pixel_values is not None:
                image_embeds = self.vlm.get_image_features(
                    pixel_values, image_grid_thw
                )
                image_embeds = torch.cat(image_embeds, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                image_mask, _ = self.vlm.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # Step 3: 3D mrope position ids
            position_ids, _ = self.vlm.get_rope_index(
                input_ids, image_grid_thw, None, attention_mask=attn_mask_1d
            )  # [3, 1, q_len]

        # Find image token span
        img_start, img_end = _get_image_token_range(input_ids, self.processor)

        # Find <hoi_feat> token positions
        #import pdb; pdb.set_trace()

        #

        hoi_positions = (input_ids[0] == self.hoi_feat_token_id).nonzero(
            as_tuple=True
        )[0]  # [n_pairs]

        q_len = input_ids.shape[1]
        return (
            inputs_embeds.detach(),
            position_ids.detach(),
            attn_mask_1d,
            img_start,
            img_end,
            hoi_positions,
            q_len,
        )

    # -----------------------------------------------------------------------
    # Prior scores (identical to HOIQWEN)
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
        pair_idx    = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        flat_target = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target] = s_h[pair_idx]
        prior_o[pair_idx, flat_target] = s_o[pair_idx]
        return torch.stack([prior_h, prior_o])

    # -----------------------------------------------------------------------
    # compute_sim_scores — main per-image logic
    # -----------------------------------------------------------------------

    def compute_sim_scores(
        self,
        region_props: List[dict],
        image,
        targets,
        text_labels: Optional[List[str]] = None,  # GT text labels for SFT loss
    ):
        """
        Returns
        -------
        all_logits_collated, prior_collated, boxes_h_collated,
        boxes_o_collated, object_class_collated, gen_losses
        """
        device = image.tensors.device

        boxes_h_collated      = []
        boxes_o_collated      = []
        prior_collated        = []
        object_class_collated = []
        all_logits_collated   = []
        gen_losses            = []

        for b_idx, props in enumerate(region_props):
            boxes  = props['boxes']
            scores = props['scores']
            labels = props['labels']

            # is_human = labels == self.human_idx
            # n_h      = torch.sum(is_human)
            # n        = len(boxes)

            # if not torch.all(labels[:n_h] == self.human_idx):
            #     h_idx = torch.nonzero(is_human).squeeze(1)
            #     o_idx = torch.nonzero(is_human == 0).squeeze(1)
            #     perm  = torch.cat([h_idx, o_idx])
            #     boxes  = boxes[perm]
            #     scores = scores[perm]
            #     labels = labels[perm]

            # vocab_size = self.lm_head_weight.shape[0]

            # if n_h == 0 or n <= 1:
            #     boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            #     boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            #     object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            #     prior_collated.append(
            #         torch.zeros(2, 0, self.num_classes, device=device)
            #     )
            #     all_logits_collated.append(torch.zeros(0, self.num_classes, device=device))
            #     continue

            # x, y = torch.meshgrid(
            #     torch.arange(n, device=device), torch.arange(n, device=device)
            # )
            # x_keep, y_keep = torch.nonzero(
            #     torch.logical_and(x != y, x < n_h)
            # ).unbind(1)

            # if len(x_keep) == 0:
            #     raise ValueError("There are no valid human-object pairs")

            # if self.training:
            #     gt_bx_h = self.recover_boxes(targets[b_idx]['boxes_h'], targets[b_idx]['size']).to(targets[b_idx]['object'].device)
            #     gt_bx_o = self.recover_boxes(targets[b_idx]['boxes_o'], targets[b_idx]['size']).to(targets[b_idx]['object'].device)
            #     boxes = torch.cat([gt_bx_h, gt_bx_o]).to(targets[b_idx]['object'].device)
            #     scores = torch.ones([boxes.shape[0]]).to(targets[b_idx]['object'].device)
            #     labels = torch.cat([torch.zeros([len(gt_bx_h)], dtype=torch.long, device=targets[b_idx]['object'].device), targets[b_idx]['object']])
            #     N = labels.numel()
            #     mid = N // 2
            #     idx = torch.arange(N, device=labels.device)
            #     x_keep = idx[:mid]
            #     y_keep = idx[mid:]
            #     n_h = mid; n = N
            # else:
            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if not torch.all(labels[:n_h] == self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
        
            if len(x_keep) == 0:
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                all_logits_collated.append(torch.zeros(0, self.num_classes, device=device))
                continue

            h_unique_indices, h_inverse_indices = torch.unique(x_keep, return_inverse=True)
            o_unique_indices, o_inverse_indices = torch.unique(y_keep, return_inverse=True)

            n_pairs = len(x_keep)

            # ------------------------------------------------------------------
            # Spatial encodings
            # ------------------------------------------------------------------
            _stride = QWEN_PATCH_SIZE * QWEN_MERGE_SIZE   # 28
            img_H   = QWEN_IMAGE_SIZE                      # 448
            img_W   = QWEN_IMAGE_SIZE
            snap_H  = (img_H // _stride) * _stride
            snap_W  = (img_W // _stride) * _stride
            grid_h  = snap_H // _stride                    # 16
            grid_w  = snap_W // _stride                    # 16

            x_boxes    = boxes[x_keep]
            y_boxes    = boxes[y_keep]
            x1 = torch.min(x_boxes[:, 0], y_boxes[:, 0])
            y1 = torch.min(x_boxes[:, 1], y_boxes[:, 1])
            x2 = torch.max(x_boxes[:, 2], y_boxes[:, 2])
            y2 = torch.max(x_boxes[:, 3], y_boxes[:, 3])
            union_boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # if self.training:
            #     pairwise_spatial = compute_spatial_encodings(
            #         [boxes[x_keep], ], [boxes[y_keep], ], [(img_H, img_W), ]
            #     ).to(torch.bfloat16)
            #     pairwise_spatial          = self.spatial_head(pairwise_spatial)
            #     pairwise_spatial_reshaped = pairwise_spatial          # [n_pairs, hidden]
            # else:
            pairwise_spatial = compute_spatial_encodings(
                [boxes[x.flatten()], ], [boxes[y.flatten()], ], [(img_H, img_W), ]
            ).to(torch.bfloat16)
            pairwise_spatial          = self.spatial_head(pairwise_spatial)
            pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)[x_keep, y_keep]

            # ------------------------------------------------------------------
            # Build PIL image and question embeddings (no_grad frozen pass)
            # ------------------------------------------------------------------
            _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
            _CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
            pil_img = tensor_to_pil(
                image.decompose()[0][b_idx:b_idx + 1].cpu(),
                mean=_CLIP_MEAN, std=_CLIP_STD,
            )

            (
                q_embeds,
                q_pos_ids,
                q_attn_1d,
                img_start,
                img_end,
                hoi_positions,
                q_len,
            ) = self._build_sft_question_embeds(pil_img, snap_H, snap_W, n_pairs)

            # ------------------------------------------------------------------
            # ROI align on ViT image features (same as ours_qwen_new_old)
            # ------------------------------------------------------------------
            img_feat = q_embeds[:, img_start:img_end, :]  # [1, num_img_tok, 2048]
            llava_feat_for_roi = (
                img_feat.float()
                .view(1, grid_h, grid_w, QWEN_HIDDEN_SIZE)
                .permute(0, 3, 1, 2)
            )  # [1, 2048, grid_h, grid_w]

            _scale = 1.0 / _stride   # 1/28

            def _roi_boxes(idx_tensor):
                bx   = boxes[idx_tensor]
                bidx = torch.zeros(bx.size(0), dtype=torch.long, device=bx.device)
                return torch.cat([bidx[:, None].float(), bx], dim=1)

            roi_ho = torch.cat([
                torch.zeros(union_boxes.size(0), 1, dtype=union_boxes.dtype,
                            device=union_boxes.device),
                union_boxes,
            ], dim=1)
            roi_boxes  = _roi_boxes(h_unique_indices)
            roi_boxes1 = _roi_boxes(o_unique_indices)

            h_feats0  = torchvision.ops.roi_align(
                llava_feat_for_roi, roi_boxes,
                output_size=(7, 7), spatial_scale=_scale, aligned=True
            )
            o_feats0  = torchvision.ops.roi_align(
                llava_feat_for_roi, roi_boxes1,
                output_size=(7, 7), spatial_scale=_scale, aligned=True
            )

            ho_feats0 = torchvision.ops.roi_align(
                llava_feat_for_roi, roi_ho,
                output_size=(7, 7), spatial_scale=_scale, aligned=True
            )
            ho_feats0 = ho_feats0.flatten(2).mean(-1).unsqueeze(0).to(img_feat.dtype)
            h_feats0  = h_feats0.flatten(2).mean(-1).unsqueeze(0).to(img_feat.dtype)
            o_feats0  = o_feats0.flatten(2).mean(-1).unsqueeze(0).to(img_feat.dtype)
            # [1, n_pairs, 2048]
            ho_feats0 = ho_feats0 + self.fusion_scale * self.ho_fusion_mlp(torch.cat([h_feats0[0][h_inverse_indices], o_feats0[0][o_inverse_indices]], dim=-1))

            # ------------------------------------------------------------------
            # Build ho_queries: ROI + spatial + object embeddings
            # ------------------------------------------------------------------
            ho_queries = (
                ho_feats0
                + pairwise_spatial_reshaped.unsqueeze(0)
                + self.object_embedding[labels[x_keep]].to(torch.bfloat16).unsqueeze(0)
                + self.object_embedding[labels[y_keep]].to(torch.bfloat16).unsqueeze(0)
            )  # [1, n_pairs, 2048]

            # ------------------------------------------------------------------
            # Splice ho_queries into question embeddings at hoi_feat positions
            # ------------------------------------------------------------------
            if len(hoi_positions) != n_pairs:
                # Mismatch — skip this sample gracefully
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                all_logits_collated.append(torch.zeros(0, self.num_classes, device=device))
                continue

            hoi_start = hoi_positions[0].item()
            hoi_end   = hoi_positions[-1].item() + 1   # exclusive

            # Detach prefix and suffix (they come from frozen no_grad forward)
            q_prefix = q_embeds[:, :hoi_start, :].detach()
            q_suffix = q_embeds[:, hoi_end:, :].detach()
            # ho_queries retains grad through spatial_head
            combined_q = torch.cat([q_prefix, ho_queries, q_suffix], dim=1)
            # [1, q_len, 2048] — hoi_feat positions now hold ho_queries

            # ------------------------------------------------------------------
            # Tokenise answer and build combined sequence
            # ------------------------------------------------------------------
            if self.training and text_labels is not None and text_labels[b_idx] is not None:
                ans_text = text_labels[b_idx]
            else:
                ans_text = "none"   # dummy during inference / no-label

            eos_tok  = self.tokenizer.eos_token or "<|im_end|>"
            full_ans = ans_text + eos_tok
            ans_ids  = self.tokenizer.encode(
                full_ans, add_special_tokens=False, return_tensors="pt"
            ).to(device)           # [1, a_len]
            a_len = ans_ids.shape[1]

            ans_embeds = self.vlm.get_input_embeddings()(ans_ids).to(
                combined_q.dtype
            )  # [1, a_len, 2048]  (no grad needed for BCE; grad flows via LM)

            combined = torch.cat([combined_q, ans_embeds], dim=1)
            # [1, q_len + a_len, 2048]
            total_len = q_len + a_len

            # ------------------------------------------------------------------
            # 4D attention mask:
            #   - causal (upper-triangular = -inf)
            #   - block: answer tokens cannot attend to image tokens
            # ------------------------------------------------------------------
            mask4d = torch.zeros(
                1, 1, total_len, total_len, dtype=combined.dtype, device=device
            )
            # Standard causal mask (upper triangular = -inf)
            mask4d += torch.triu(
                torch.full_like(mask4d, float("-inf")), diagonal=1
            )
            # Block answer → image
            mask4d[:, :, q_len:, img_start:img_end] = float("-inf")

            # ------------------------------------------------------------------
            # Extend position_ids for answer tokens
            # ------------------------------------------------------------------
            # q_pos_ids: [3, 1, q_len]
            # For answer tokens, all 3 mrope dims just continue from last q pos
            last_pos   = q_pos_ids[:, 0, -1]                       # [3]
            ans_offsets = torch.arange(1, a_len + 1, device=device) # [a_len]
            ans_pos = (last_pos[:, None] + ans_offsets[None, :]).unsqueeze(1)
            # [3, 1, a_len]
            pos_ids_full = torch.cat(
                [q_pos_ids.to(device), ans_pos.to(device)], dim=-1
            )  # [3, 1, total_len]

            # ------------------------------------------------------------------
            # Forward through language model (LoRA active)
            # ------------------------------------------------------------------
            output = self.vlm.language_model(
                input_ids=None,
                position_ids=pos_ids_full,
                attention_mask=mask4d,
                inputs_embeds=combined,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = output.last_hidden_state   # [1, total_len, 2048]
            #if not self.training:
            # with torch.no_grad():
            #     generated_ids = self.peft_qwen.generate(
            #         inputs_embeds=combined_q,        # [1, q_len, 2048]  — question only
            #         attention_mask=torch.ones(
            #             1, q_len, device=device, dtype=torch.long
            #         ),
            #         max_new_tokens=128,
            #         do_sample=True,      # Enable multinomial sampling
            #         temperature=0.7,     # High (>1.0) is more random, Low (<1.0) is more confident
            #         top_p=0.9,           # Nucleus sampling: keep top 90% cumulative prob tokens
            #         top_k=50,            # Keep top 50 highest probability tokens
                    
            #         eos_token_id=self.tokenizer.eos_token_id,
            #         pad_token_id=self.tokenizer.eos_token_id,
            #     )

            # # generated_ids[0] contains only the new tokens if using return_dict_in_generate,
            # # otherwise the full sequence; decode only the generated part:
            # gen_text = self.tokenizer.decode(
            #     generated_ids[0], skip_special_tokens=True
            # )

            # ------------------------------------------------------------------
            # BCE scoring  (same path as HOIQWEN.compute_sim_scores)
            # ------------------------------------------------------------------
            lm_head_t = (
                self.lm_head_weight.T.to(device).to(hidden.dtype)
            )  # [2048, vocab_size]

            hidden_hoi = hidden[:, hoi_start:hoi_end, :]   # [1, n_pairs, 2048]
            # normed by the LM's final norm (already applied inside language_model)
            ho_vocab   = hidden_hoi.squeeze(0) @ lm_head_t  # [n_pairs, vocab_size]

            all_probs_ho           = F.softmax(ho_vocab, dim=-1)
            ho_vocab_weighted_mean = (all_probs_ho * ho_vocab).sum(-1)#.detach()

           # import pdb;pdb.set_trace()

            if self.num_classes == 117:
                form_counts = (
                    self.verb_token_mask_test.sum(dim=-1).unsqueeze(0).clamp(min=1)
                )  # [1, 117, num_forms]

                scores_m = ho_vocab[:, self.verb_token_ids_test]   # [n_pairs, 117, nf, max_tok]
                scores_m = scores_m.masked_fill(
                    ~self.verb_token_mask_test.unsqueeze(0), 0.0
                )
                ho_verb_avg = (scores_m.sum(dim=-1) / form_counts).max(dim=-1).values
                # [n_pairs, 117]
                logits = ho_verb_avg - ho_vocab_weighted_mean.unsqueeze(-1)
                

            elif self.num_classes == 600:
                hoi_counts = self.hoi_token_mask.sum(dim=-1).unsqueeze(0).clamp(min=1)
                scores_hoi = ho_vocab[:, self.hoi_token_ids]
                scores_hoi = scores_hoi.masked_fill(
                    ~self.hoi_token_mask.unsqueeze(0), 0.0
                )
                ho_hoi_avg = scores_hoi.sum(dim=-1) / hoi_counts
                logits = ho_hoi_avg - ho_vocab_weighted_mean.unsqueeze(-1)
                

            # ------------------------------------------------------------------
            # Generation loss (SFT cross-entropy)
            # ------------------------------------------------------------------
            # if self.training and a_len > 0 and ans_text != "none":
            #     gen_hidden = hidden[:, q_len - 1:q_len + a_len - 1, :]
            #     # [1, a_len, 2048]
            #     gen_logits = gen_hidden.float() @ lm_head_t.float()
            #     # [1, a_len, vocab_size]
            #     gen_loss = F.cross_entropy(
            #         gen_logits[0],   # [a_len, vocab_size]
            #         ans_ids[0],      # [a_len]
            #         ignore_index=-100,
            #     )
            #     gen_losses.append(gen_loss)
            # print(self.tokenizer.decode(torch.topk(ho_vocab[3], 100, dim=-1)[1]))
            # import pdb; pdb.set_trace()

            # self.tokenizer.decode(gen_logits[0].argmax(-1))
            # self.tokenizer.decode(ans_ids[0])
            # ------------------------------------------------------------------
            # Collect outputs
            # ------------------------------------------------------------------
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(
                self.compute_prior_scores(x_keep, y_keep, scores, labels)
            )
            all_logits_collated.append(logits)
            #import pdb; pdb.set_trace()
        return (
            all_logits_collated,
            prior_collated,
            boxes_h_collated,
            boxes_o_collated,
            object_class_collated,
            gen_losses,
        )

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

        loss = binary_focal_loss_with_logits(
        torch.log(
            prior / (1 + torch.exp(-logits) - prior) + 1e-8
        ), labels, reduction='sum',
                alpha=self.alpha, gamma=self.gamma
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
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        images:      List[Tensor],
        targets:     Optional[List[dict]] = None,
    ) -> List[dict]:

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images_orig = [im[0].float() for im in images]
        images_clip = [im[1] for im in images]

        device      = images_clip[0].device
        image_sizes = torch.as_tensor(
            [im.size()[-2:] for im in images_clip], device=device
        )

        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)

        self.detector.eval()
        features, pos = self.detector.backbone(images_orig.to(device))
        src, mask     = features[-1].decompose()
        hs, _ = self.detector.transformer(
            self.detector.input_proj(src), mask,
            self.detector.query_embed.weight, pos[-1],
        )
        outputs_class = self.detector.class_embed(hs)
        outputs_coord = self.detector.bbox_embed(hs).sigmoid()

        if self.dataset == 'vcoco' and outputs_class.shape[-1] == 92:
            outputs_class = outputs_class[:, :, :, self.reserve_indices]

        results = {
            'pred_logits': outputs_class[-1],
            'pred_boxes':  outputs_coord[-1],
            'feats':       hs[-1],
        }
        results      = self.postprocessor(results, image_sizes)
        region_props = self.prepare_region_proposals(results)
        images_clip  = nested_tensor_from_tensor_list(images_clip)

        text_labels = None
        if self.training and targets is not None:
            text_labels = [t.get('text_label', None) for t in targets]

        logits, prior, bh, bo, objects, gen_losses = self.compute_sim_scores(
            region_props, images_clip, targets, text_labels=text_labels,
        )
        # if self.training:
        #     boxes = []
        #     for t in targets:
        #         gt_bx_h = self.recover_boxes(t['boxes_h'], t['size']).to(device)
        #         gt_bx_o = self.recover_boxes(t['boxes_o'], t['size']).to(device)
        #         boxes.append(torch.cat([gt_bx_h, gt_bx_o]))
        # else:
        boxes = [r['boxes'] for r in region_props]

        if self.training:
            bce_loss = self.compute_interaction_loss(
                boxes, bh, bo, logits, prior, targets,
            )
            if gen_losses:
                gen_loss_sum = sum(gen_losses) 
                gen_loss_count = torch.tensor(len(gen_losses), dtype=torch.float, device=device)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(gen_loss_count, op=dist.ReduceOp.SUM)
                gen_loss = gen_loss_sum / gen_loss_count.clamp(min=1)
                total_loss = bce_loss + self.sft_loss_weight * gen_loss
                #import pdb; pdb.set_trace()

            else:
                total_loss = bce_loss
                #import pdb; pdb.set_trace()
            return dict(interaction_loss=total_loss)

        if len(logits) == 0:
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
            scores = torch.sigmoid(lg[x, y])
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
# build_hoi_sft_detector
# ---------------------------------------------------------------------------

def build_hoi_sft_detector(
    args,
    class_corr,
    object_n_verb_to_interaction,
    clip_model_path,
    rank,
    sft_loss_weight: float = 1.0,
    lora_rank:       int   = 8,
):
    """
    Build HOIQWEN_SFT detector.  Mirrors build_detector() in ours_qwen_new_old.py.
    """
    # DETR backbone
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

    # Qwen2.5-VL-3B
    model_state = load_qwen_state(rank)

    # Object embeddings from Qwen LM head
    obj_class_names  = [
        obj[1].replace("a photo of a ", "").replace("a photo of an ", "")
        for obj in hico_text_label.hico_obj_text_label
    ]
    object_embedding = compute_object_embeddings_qwen(
        model_state, obj_class_names, device=f"cuda:{rank}"
    ).clone().detach()

    detector = HOIQWEN_SFT(
        args,
        detr, postprocessors['bbox'], model_state, object_embedding,
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
        sft_loss_weight=sft_loss_weight,
        lora_rank=lora_rank,
    )
    return detector
