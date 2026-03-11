"""
ours_qwen_hoi_sft_listwise.py — Listwise HOI scoring via Qwen2.5-VL-3B + LoRA.

For each H-O pair (Bh, Bo, Co), the model builds a prompt:

  "The interaction features are <hoi_feat>. Select the correct interaction
   from the list: person verb1 obj<hoi>, person verb2 obj<hoi>, ..., person verbM obj<hoi>."

<hoi_feat> is replaced with the pair's ho_query feature. Pairs sharing the
same object class Co are batched together (one LLM forward per unique object
class per image). The confidence score for candidate k is:

    S[k] = cosine(hidden_at_<hoi>_k, hidden_at_<hoi_feat>)

Loss: binary focal loss on prior * S.clamp(0, 1) vs GT labels.
      Same compute_interaction_loss as the original model.
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
    HOI_SEP_TOKEN,
    HOI_NEG_TOKEN,
    get_hoi_token_id,
)


# ---------------------------------------------------------------------------
# Object embedding helper
# ---------------------------------------------------------------------------

def compute_object_embeddings_qwen(model_state, obj_class_names, device="cpu"):
    tokenizer = model_state["tokenizer"]
    lm_head   = model_state["model"].lm_head.weight.detach()
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
    Listwise HOI detector.

    For each H-O pair, builds a prompt listing all candidate interactions for
    that object class, each followed by <hoi>. Scores = cosine similarity
    between hidden state at <hoi_feat> (= ho_query) and at each <hoi> token.
    """

    def __init__(
        self,
        args,
        detector:       nn.Module,
        postprocessor:  nn.Module,
        model_state:    dict,
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
        lora_rank:        int   = 8,
        use_gt_boxes:     bool  = False,
        use_prior:        bool  = False,
        mask_ans_to_img:  bool  = True,
    ) -> None:
        super().__init__()

        self.detector      = detector
        self.postprocessor = postprocessor

        self.register_buffer("object_embedding", object_embedding)

        self.visual_output_dim            = QWEN_HIDDEN_SIZE
        self.object_n_verb_to_interaction = np.asarray(
            object_n_verb_to_interaction, dtype=float
        )
        self.args          = args
        self.human_idx     = human_idx
        self.num_classes   = num_classes
        self.hyper_lambda  = args.hyper_lambda
        self.alpha         = alpha
        self.gamma         = gamma
        self.box_score_thresh             = box_score_thresh
        self.fg_iou_thresh                = fg_iou_thresh
        self.min_instances                = min_instances
        self.max_instances                = max_instances
        self.object_class_to_target_class = object_class_to_target_class
        self.dataset                      = args.dataset
        self.reserve_indices              = reserve_indices
        self.use_gt_boxes                 = use_gt_boxes
        self.use_prior                    = use_prior
        self.mask_ans_to_img              = mask_ans_to_img
        self.training_stage               = 2  # no stage-1 interactiveness head here

        # ------------------------------------------------------------------
        # Qwen model + processor + tokenizer
        # ------------------------------------------------------------------
        qwen_model      = model_state["model"]
        self.processor  = model_state["processor"]
        self.tokenizer  = model_state["tokenizer"]
        self.vlm        = qwen_model.model   # Qwen2_5_VLModel

        self.register_buffer(
            "lm_head_weight",
            qwen_model.lm_head.weight.detach().cpu().to(torch.bfloat16).contiguous(),
        )

        # ------------------------------------------------------------------
        # Register special tokens: <hoi_feat>, <hoi>, <hoi_neg>
        # ------------------------------------------------------------------
        missing = [t for t in [HOI_FEAT_TOKEN, HOI_SEP_TOKEN, HOI_NEG_TOKEN]
                   if t not in self.tokenizer.all_special_tokens]
        if missing:
            self.tokenizer.add_special_tokens({"additional_special_tokens": missing})
            qwen_model.resize_token_embeddings(len(self.tokenizer))

        self.hoi_feat_token_id = get_hoi_token_id(self.tokenizer)
        self.hoi_sep_token_id  = self.tokenizer.convert_tokens_to_ids(HOI_SEP_TOKEN)
        self.hoi_neg_token_id  = self.tokenizer.convert_tokens_to_ids(HOI_NEG_TOKEN)

        # ------------------------------------------------------------------
        # Apply PEFT LoRA
        # ------------------------------------------------------------------
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            lora_dropout=0.1,
        )
        self.peft_qwen = get_peft_model(qwen_model, peft_config)
        self.peft_qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.peft_qwen.enable_input_require_grads()

        for name, p in self.peft_qwen.named_parameters():
            p.requires_grad = ("lora_" in name)

        # ------------------------------------------------------------------
        # Spatial / fusion heads (same as original model)
        # ------------------------------------------------------------------
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),  nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, QWEN_HIDDEN_SIZE), nn.ReLU(),
        )
        self.ho_fusion_mlp = nn.Sequential(
            nn.Linear(2 * QWEN_HIDDEN_SIZE, 256), nn.ReLU(),
            nn.Linear(256, QWEN_HIDDEN_SIZE),
        )
        self.ho_spatial_fusion_mlp = nn.Sequential(
            nn.Linear(2 * QWEN_HIDDEN_SIZE, 256), nn.ReLU(),
            nn.Linear(256, QWEN_HIDDEN_SIZE),
        )

        # ------------------------------------------------------------------
        # Zero-shot filtering
        # ------------------------------------------------------------------
        if args.zs:
            self.zs_type          = args.zs_type
            self.filtered_hoi_idx = hico_text_label.hico_unseen_index[self.zs_type]
        else:
            self.filtered_hoi_idx = []
            self.zs_type          = None

        self.seen_verb_idxs = list(set([
            HOI_IDX_TO_ACT_IDX[idx]
            for idx in range(600)
            if idx not in self.filtered_hoi_idx
        ]))

    # -----------------------------------------------------------------------
    # Candidate text builder
    # -----------------------------------------------------------------------

    def _build_candidate_text(self, obj_class: int):
        """
        Build the candidate list text and return corresponding class indices.

        Returns
        -------
        candidate_text : str
            "person verb1 obj<hoi> person verb2 obj<hoi> ..."
        cand_indices : list[int]
            Verb (num_classes=117) or HOI (num_classes=600) indices, one per candidate.
        """
        candidates = self.object_class_to_target_class[obj_class]
        if not candidates:
            return "", []

        texts = []
        if self.num_classes == 117:
            obj_name = hico_objects[obj_class]
            for verb_idx in candidates:
                texts.append(
                    f"person {hico_verbs[verb_idx]} {obj_name}{HOI_SEP_TOKEN}"
                )
        else:
            for hoi_idx in candidates:
                verb, obj = hico_verb_object_list[hoi_idx]
                texts.append(f"person {verb} {obj}{HOI_SEP_TOKEN}")

        return " ".join(texts), list(candidates)

    # -----------------------------------------------------------------------
    # Prompt embedding builder (listwise format)
    # -----------------------------------------------------------------------

    def _build_listwise_embeds(self, pil_img, snap_H, snap_W, candidate_text: str):
        """
        Build question embeddings for one object-class candidate group.

        Prompt:
          "The interaction features are <hoi_feat>. Select the correct
           interaction from the list: {candidate_text}"

        Returns
        -------
        q_embeds       : Tensor [1, q_len, H]  (detached)
        q_pos_ids      : Tensor [3, 1, q_len]  (detached)
        img_start      : int
        img_end        : int
        hoi_feat_pos   : int — index of the single <hoi_feat> token
        hoi_sep_positions : Tensor [M] — indices of the M <hoi> tokens
        q_len          : int
        """
        device = next(self.peft_qwen.parameters()).device
        text_prompt = (
            f"The interaction features are {HOI_FEAT_TOKEN}. "
            f"Select the correct interaction from the list: {candidate_text}"
        )
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

            inputs_embeds = self.vlm.get_input_embeddings()(input_ids)

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

            position_ids, _ = self.vlm.get_rope_index(
                input_ids, image_grid_thw, None, attention_mask=attn_mask_1d
            )

        img_start, img_end = _get_image_token_range(input_ids, self.processor)

        feat_positions = (input_ids[0] == self.hoi_feat_token_id).nonzero(
            as_tuple=True
        )[0]
        assert len(feat_positions) == 1, (
            f"Expected exactly 1 <hoi_feat>, got {len(feat_positions)}"
        )
        hoi_feat_pos = feat_positions[0].item()

        hoi_sep_positions = (input_ids[0] == self.hoi_sep_token_id).nonzero(
            as_tuple=True
        )[0]  # [M]

        q_len = input_ids.shape[1]
        return (
            inputs_embeds.detach(),
            position_ids.detach(),
            img_start,
            img_end,
            hoi_feat_pos,
            hoi_sep_positions,
            q_len,
        )

    # -----------------------------------------------------------------------
    # Prior scores (identical to original model)
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
    # compute_sim_scores — listwise forward
    # -----------------------------------------------------------------------

    def compute_sim_scores(
        self,
        region_props: List[dict],
        image,
        targets,
    ):
        """
        For each image, groups H-O pairs by object class and runs one LLM
        forward per group (batched over pairs in the group). Cosine similarity
        between hidden_at_<hoi_feat> and hidden_at_each_<hoi> gives scores.

        Returns standard collated outputs compatible with compute_interaction_loss.
        """
        device = image.tensors.device

        boxes_collated        = []
        boxes_h_collated      = []
        boxes_o_collated      = []
        prior_collated        = []
        object_class_collated = []
        all_logits_collated   = []

        _stride = QWEN_PATCH_SIZE * QWEN_MERGE_SIZE   # 28
        img_H   = QWEN_IMAGE_SIZE                      # 448
        img_W   = QWEN_IMAGE_SIZE
        snap_H  = (img_H // _stride) * _stride
        snap_W  = (img_W // _stride) * _stride
        grid_h  = snap_H // _stride                    # 16
        grid_w  = snap_W // _stride

        _CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
        _CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

        for b_idx, props in enumerate(region_props):
            # ------------------------------------------------------------------
            # Box / pair extraction (same as original)
            # ------------------------------------------------------------------
            if self.training and self.use_gt_boxes and targets is not None:
                gt_bx_h_all = self.recover_boxes(
                    targets[b_idx]['boxes_h'], targets[b_idx]['size']
                ).to(device)
                gt_bx_o_all = self.recover_boxes(
                    targets[b_idx]['boxes_o'], targets[b_idx]['size']
                ).to(device)
                gt_obj_all  = targets[b_idx]['object'].to(device)

                def _iou_dedup(boxes, iou_thresh=0.9):
                    if len(boxes) == 0:
                        return boxes, torch.zeros(0, dtype=torch.long, device=boxes.device)
                    scores_dummy = torch.ones(len(boxes), device=boxes.device)
                    keep   = torchvision.ops.nms(boxes.float(), scores_dummy, iou_thresh)
                    unique = boxes[keep]
                    inv    = box_iou(boxes.float(), unique.float()).argmax(dim=1)
                    return unique, inv

                gt_bx_h, h_inv = _iou_dedup(gt_bx_h_all)
                gt_bx_o, o_inv = _iou_dedup(gt_bx_o_all)

                o_labels = torch.zeros(len(gt_bx_o), dtype=torch.long, device=device)
                o_labels[o_inv] = gt_obj_all

                n_h = len(gt_bx_h)
                n_o = len(gt_bx_o)
                boxes  = torch.cat([gt_bx_h, gt_bx_o])
                scores = torch.full((n_h + n_o,), 1.0, device=device)
                labels = torch.cat([
                    torch.zeros(n_h, dtype=torch.long, device=device),
                    o_labels,
                ])
                xg, yg = torch.meshgrid(
                    torch.arange(n_h, device=device),
                    torch.arange(n_h, n_h + n_o, device=device),
                    indexing='ij',
                )
                x_keep = xg.flatten()
                y_keep = yg.flatten()
            else:
                boxes  = props['boxes']
                scores = props['scores']
                labels = props['labels']
                is_human = labels == self.human_idx
                n_h = torch.sum(is_human); n = len(boxes)
                if not torch.all(labels[:n_h] == self.human_idx):
                    h_idx = torch.nonzero(is_human).squeeze(1)
                    o_idx = torch.nonzero(is_human == 0).squeeze(1)
                    perm  = torch.cat([h_idx, o_idx])
                    boxes = boxes[perm]; scores = scores[perm]; labels = labels[perm]
                x, y = torch.meshgrid(
                    torch.arange(n, device=device),
                    torch.arange(n, device=device),
                )
                x_keep, y_keep = torch.nonzero(
                    torch.logical_and(x != y, x < n_h)
                ).unbind(1)

            if len(x_keep) == 0:
                boxes_collated.append(boxes)
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
            # Spatial encodings → ho_queries (same as original)
            # ------------------------------------------------------------------
            x_boxes = boxes[x_keep]
            y_boxes = boxes[y_keep]

            pairwise_spatial = compute_spatial_encodings(
                [boxes[x_keep]], [boxes[y_keep]], [(img_H, img_W)]
            ).to(torch.bfloat16)
            pairwise_spatial_reshaped = self.spatial_head(
                pairwise_spatial.float()
            ).to(torch.bfloat16)  # [n_pairs, H]

            # Build PIL image
            pil_img = tensor_to_pil(
                image.decompose()[0][b_idx:b_idx + 1].cpu(),
                mean=_CLIP_MEAN, std=_CLIP_STD,
            )

            # ROI-align: need a quick dummy forward to get image features
            # We'll extract them via a single no-grad pass for the base image,
            # then reuse across groups.
            with torch.no_grad():
                _base_inputs = _build_qwen_inputs(
                    self.processor, pil_img,
                    f"The interaction features are {HOI_FEAT_TOKEN}.",
                    device, snap_H, snap_W,
                )
                _base_ids    = _base_inputs["input_ids"]
                _pv          = _base_inputs.get("pixel_values")
                _thw         = _base_inputs.get("image_grid_thw")
                _base_embeds = self.vlm.get_input_embeddings()(_base_ids)
                if _pv is not None:
                    _img_embs = self.vlm.get_image_features(_pv, _thw)
                    _img_embs = torch.cat(_img_embs, dim=0).to(
                        _base_embeds.device, _base_embeds.dtype
                    )
                    _img_mask, _ = self.vlm.get_placeholder_mask(
                        _base_ids, inputs_embeds=_base_embeds, image_features=_img_embs
                    )
                    _base_embeds = _base_embeds.masked_scatter(_img_mask, _img_embs)
                _img_s, _img_e = _get_image_token_range(_base_ids, self.processor)

            img_feat = _base_embeds[:, _img_s:_img_e, :]   # [1, num_img_tok, H]
            llava_feat_for_roi = (
                img_feat
                .view(1, grid_h, grid_w, QWEN_HIDDEN_SIZE)
                .permute(0, 3, 1, 2)
            )  # [1, H, grid_h, grid_w]

            _scale = 1.0 / _stride

            def _roi_boxes(idx_tensor):
                bx   = boxes[idx_tensor]
                bidx = torch.zeros(bx.size(0), dtype=torch.long, device=bx.device)
                return torch.cat([bidx[:, None], bx], dim=1)

            _roi_input = llava_feat_for_roi.float()
            h_feats0 = torchvision.ops.roi_align(
                _roi_input, _roi_boxes(h_unique_indices),
                output_size=(7, 7), spatial_scale=_scale, aligned=True,
            )
            o_feats0 = torchvision.ops.roi_align(
                _roi_input, _roi_boxes(o_unique_indices),
                output_size=(7, 7), spatial_scale=_scale, aligned=True,
            )
            del _roi_input

            h_feats0 = h_feats0.flatten(2).mean(-1).unsqueeze(0).to(img_feat.dtype)
            o_feats0 = o_feats0.flatten(2).mean(-1).unsqueeze(0).to(img_feat.dtype)

            ho_fuse  = self.ho_fusion_mlp(
                torch.cat([
                    h_feats0[0][h_inverse_indices],
                    o_feats0[0][o_inverse_indices],
                ], dim=-1).unsqueeze(0).float()
            )
            ho_feats0 = (
                h_feats0[0][h_inverse_indices] + o_feats0[0][o_inverse_indices]
            ).unsqueeze(0)

            ho_fuse1  = self.ho_spatial_fusion_mlp(
                torch.cat([ho_fuse, pairwise_spatial_reshaped.unsqueeze(0)], dim=-1).float()
            ).to(torch.bfloat16)
            ho_queries = (
                ho_fuse1
                + ho_feats0
                + self.object_embedding[labels[x_keep]].to(torch.bfloat16).unsqueeze(0)
                + self.object_embedding[labels[y_keep]].to(torch.bfloat16).unsqueeze(0)
            )  # [1, n_pairs, H]

            if self.training:
                pair_perm  = torch.randperm(n_pairs, device=device)
                ho_queries = ho_queries[:, pair_perm, :]
                x_keep     = x_keep[pair_perm]
                y_keep     = y_keep[pair_perm]

            # ------------------------------------------------------------------
            # Listwise scoring: one forward pass per object class (batched)
            # ------------------------------------------------------------------
            pair_obj_classes = labels[y_keep]            # [n_pairs]
            logits = torch.zeros(n_pairs, self.num_classes, device=device)

            # Build prompt cache and group pair indices by object class
            _prompt_cache: dict = {}
            cls_to_pairs: dict = defaultdict(list)
            for pair_idx in range(n_pairs):
                obj_cls_int = pair_obj_classes[pair_idx].item()
                if obj_cls_int not in _prompt_cache:
                    candidate_text, cand_indices = self._build_candidate_text(obj_cls_int)
                    if not cand_indices:
                        _prompt_cache[obj_cls_int] = None
                    else:
                        result = self._build_listwise_embeds(
                            pil_img, snap_H, snap_W, candidate_text
                        )
                        M = len(result[5])  # hoi_sep_positions
                        if M != len(cand_indices):
                            _prompt_cache[obj_cls_int] = None
                        else:
                            _prompt_cache[obj_cls_int] = (result, cand_indices)
                if _prompt_cache[obj_cls_int] is not None:
                    cls_to_pairs[obj_cls_int].append(pair_idx)

            # One LLM forward per chunk of pairs sharing the same object class.
            # MAX_FWD_BATCH caps peak memory: each chunk is [chunk, q_len, H].
            MAX_FWD_BATCH = 4

            for obj_cls_int, pair_indices in cls_to_pairs.items():
                (
                    q_embeds,
                    q_pos_ids,
                    img_start,
                    img_end,
                    hoi_feat_pos,
                    hoi_sep_positions,
                    q_len,
                ), cand_indices = _prompt_cache[obj_cls_int]

                cand_tensor = torch.tensor(cand_indices, device=device, dtype=torch.long)

                # Shared causal mask template [1, 1, q_len, q_len] — reused per chunk
                mask_base = torch.triu(
                    torch.full(
                        (1, 1, q_len, q_len), float("-inf"),
                        dtype=q_embeds.dtype, device=device,
                    ),
                    diagonal=1,
                )
                if self.mask_ans_to_img:
                    mask_base[:, :, hoi_feat_pos + 1:, img_start:img_end] = float("-inf")

                # Process in sub-batches to bound peak memory
                for chunk_start in range(0, len(pair_indices), MAX_FWD_BATCH):
                    chunk = pair_indices[chunk_start: chunk_start + MAX_FWD_BATCH]
                    B = len(chunk)

                    # [B, q_len, H] — inject per-pair ho_query at <hoi_feat>
                    batch_embeds = q_embeds.expand(B, -1, -1).clone()
                    for i, pidx in enumerate(chunk):
                        batch_embeds[i, hoi_feat_pos, :] = ho_queries[0, pidx, :].to(batch_embeds.dtype)

                    output = self.vlm.language_model(
                        input_ids=None,
                        position_ids=q_pos_ids.expand(-1, B, -1),
                        attention_mask=mask_base.expand(B, -1, -1, -1),
                        inputs_embeds=batch_embeds,
                        use_cache=False,
                        output_hidden_states=False,
                        output_attentions=False,
                        return_dict=True,
                    )
                    hidden = output.last_hidden_state  # [B, q_len, H]

                    f_inter = F.normalize(hidden[:, hoi_feat_pos, :].float(), dim=-1)       # [B, H]
                    f_hoi   = F.normalize(hidden[:, hoi_sep_positions, :].float(), dim=-1)  # [B, M, H]
                    pair_scores = (f_hoi * f_inter.unsqueeze(1)).sum(dim=-1)                # [B, M]

                    for i, pidx in enumerate(chunk):
                        logits[pidx, cand_tensor] = pair_scores[i].to(logits.dtype)


            # ------------------------------------------------------------------
            # Collect outputs
            # ------------------------------------------------------------------
            boxes_collated.append(boxes)
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
            boxes_collated,
        )

    # -----------------------------------------------------------------------
    # Ground-truth association and interaction loss (unchanged from original)
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
            prior * logits.clamp(0, 1),
            labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma,
        )
        return loss / max(n_p, 1)

    # -----------------------------------------------------------------------
    # Region proposals (unchanged)
    # -----------------------------------------------------------------------

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
        images:  List[Tensor],
        targets: Optional[List[dict]] = None,
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

        results      = {
            'pred_logits': outputs_class[-1],
            'pred_boxes':  outputs_coord[-1],
            'feats':       hs[-1],
        }
        results      = self.postprocessor(results, image_sizes)
        region_props = self.prepare_region_proposals(results)
        images_clip  = nested_tensor_from_tensor_list(images_clip)

        logits, prior, bh, bo, objects, boxes = self.compute_sim_scores(
            region_props, images_clip, targets,
        )

        if self.training:
            bce_loss   = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets)
            return dict(interaction_loss=bce_loss)

        if len(logits) == 0:
            return None

        return self.postprocessing(
            boxes, bh, bo, logits, prior, objects, image_sizes,
            use_prior=self.use_prior,
        )

    def postprocessing(
        self, boxes, bh, bo, logits, prior, objects, image_sizes, use_prior=True
    ):
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
# build_hoi_listwise_detector
# ---------------------------------------------------------------------------

def build_hoi_sft_detector(
    args,
    class_corr,
    object_n_verb_to_interaction,
    clip_model_path,
    rank,
    sft_loss_weight:    float = 1.0,   # accepted but unused (no gen loss here)
    use_gen_loss:       bool  = True,  # accepted but unused
    lora_rank:          int   = 8,
    use_img_cross_attn: bool  = False, # accepted but unused
    use_gt_boxes:       bool  = False,
    use_prior:          bool  = False,
    mask_ans_to_img:    bool  = True,
):
    """Build HOIQWEN_LISTWISE detector."""
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

    model_state = load_qwen_state(rank)

    obj_class_names = [
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
        lora_rank=lora_rank,
        use_gt_boxes=use_gt_boxes,
        use_prior=use_prior,
        mask_ans_to_img=mask_ans_to_img,
    )
    return detector
