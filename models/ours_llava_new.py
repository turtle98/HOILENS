"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>
The Australian National University
Australian Centre for Robotic Vision
"""

from entmax import sparsemax
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

import numpy as np
import torchvision 
#import wandb

from utils.hico_list import hico_verbs_sentence, hico_verb_object_list, hico_verbs, hico_objects
from utils.vcoco_list import vcoco_verbs_sentence
from utils.hico_utils import reserve_indices, HOI_IDX_TO_ACT_IDX
from utils.postprocessor import PostProcess
from utils.ops import binary_focal_loss_with_logits, binary_focal_loss, vectorized_bboxes_and_indices, bbox_to_token, compute_spatial_encodings
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

#from llava.model.multimodal_projector.builder import build_vision_projector
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import AutoConfig

sys.path.pop(0)
import math
import random

import copy
import nltk
from nltk import word_tokenize, pos_tag
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
# Clone a LLaMA layer directly

# Usage:
# llama_layer = model["model"].model.layers[15]  # pick a layer
# decoder = create_decoder_from_llama_layer(llama_layer, num_heads=32)


from methods.llava_utils import retrieve_logit_lens_llava, load_llava_state, run_llava_model, generate_llava_model, compute_conditional_likelihood_llava, get_img_idx
from methods.attention import llama_modify

def rms_normalize(x, dim=-1, eps=1e-6):
    return x / x.pow(2).mean(dim, keepdim=True).add(eps).sqrt()

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def expand_boxes_for_layers(boxes, num_layers):
    # boxes: (num_boxes, 5) with batch_idx in first column
    num_boxes = boxes.shape[0]
    expanded = boxes[:, 1:].repeat(num_layers, 1)  # (num_layers * num_boxes, 4)
    batch_idx = torch.arange(num_layers, device=boxes.device).repeat_interleave(num_boxes).unsqueeze(1)
    return torch.cat([batch_idx.float(), expanded], dim=1)

class LoRALinear(nn.Module):
    """LoRA adapter for a frozen linear layer."""
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x, frozen_weight):
        """
        Args:
            x: input tensor
            frozen_weight: the frozen nn.Linear layer
        """
        # Frozen path: cast input to match frozen weight dtype (e.g., fp16)
        frozen_dtype = frozen_weight.weight.dtype
        frozen_out = frozen_weight(x.to(frozen_dtype))
        # LoRA path: compute in fp32 for training stability, then cast back to frozen weight dtype
        lora_out = self.lora_B(self.lora_A(x.float())).to(frozen_dtype) * self.scaling
        return frozen_out + lora_out


class CrossAttendWithLoRA(nn.Module):
    """Cross-attention using cloned LLaMA layers with LoRA adapters."""
    
    def __init__(self, cloned_layers, lora_rank=16, lora_alpha=16):
        super().__init__()
        self.cloned_layers = cloned_layers
        
        # Freeze cloned layers
        for layer in self.cloned_layers:
            for param in layer.parameters():
                param.requires_grad = False
            # for param in layer.mlp.parameters():
            #     param.requires_grad = True
        
        # Create LoRA adapters for each layer's attention
        self.lora_q = nn.ModuleList()
        self.lora_k = nn.ModuleList()
        self.lora_v = nn.ModuleList()
        self.lora_o = nn.ModuleList()
        
        for layer in cloned_layers:
            hidden_size = layer.self_attn.q_proj.weight.shape[0]
            self.lora_q.append(LoRALinear(hidden_size, hidden_size, lora_rank, lora_alpha))
            self.lora_k.append(LoRALinear(hidden_size, hidden_size, lora_rank, lora_alpha))
            self.lora_v.append(LoRALinear(hidden_size, hidden_size, lora_rank, lora_alpha))
            self.lora_o.append(LoRALinear(hidden_size, hidden_size, lora_rank, lora_alpha))

    def forward(self, queries, context, roi_mask=None, return_attn_weights=False):
        hidden_states = queries
        attn_weights_all = []

        for idx, layer in enumerate(self.cloned_layers):
            attn = layer.self_attn
            bsz, q_len, _ = hidden_states.shape
            kv_len = context.shape[1]

            # Pre-norm and residual
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            normed_kv = layer.input_layernorm(context)  # <-- ADD THIS: normalize context too

            # Project with LoRA: frozen + low-rank adaptation
            q = self.lora_q[idx](normed, attn.q_proj)
            k = self.lora_k[idx](normed_kv, attn.k_proj)
            v = self.lora_v[idx](normed_kv, attn.v_proj)

            head_dim = attn.head_dim
            num_heads = attn.num_heads

            q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, kv_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(bsz, kv_len, num_heads, head_dim).transpose(1, 2)

            # Attention mask (must match q's dtype)
            attn_mask = None
            if roi_mask is not None:
               attn_mask = torch.zeros_like(roi_mask, dtype=q.dtype)
               attn_mask[~roi_mask] = float('-inf')
               attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

            if return_attn_weights:
                # Manual attention to get weights
                scale = head_dim ** -0.5
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                if attn_mask is not None:
                    scores = scores + attn_mask
                attn_w = F.softmax(scores, dim=-1)
                attn_weights_all.append(attn_w)
                attn_out = torch.matmul(attn_w, v)
            else:
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask)
            
            #import pdb; pdb.set_trace()

            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)

            # Output projection with LoRA
            attn_out = self.lora_o[idx](attn_out, attn.o_proj)

            hidden_states = residual + attn_out

            # FFN (frozen, no LoRA)
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        if return_attn_weights:
            return hidden_states, attn_out
        return hidden_states


class HOILLAVA(nn.Module):
    def __init__(self,
        args,
        detector: nn.Module,
        postprocessor: nn.Module,
        model: nn.Module,
        object_embedding: torch.tensor,
        human_idx: int, num_classes: int,
        alpha: float = 0.5, gamma: float = 2.0,
        box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
        min_instances: int = 3, max_instances: int = 15,
        object_class_to_target_class: List[list] = None,
        object_n_verb_to_interaction: List[list] = None,

    ) -> None:
        super().__init__()
        self.detector = detector
        #self.detector.eval()
        self.postprocessor = postprocessor
        self.clip_head = model

        self.register_buffer("object_embedding",object_embedding)
        self.visual_output_dim = 4096
        self.object_n_verb_to_interaction = np.asarray(
                                object_n_verb_to_interaction, dtype=float
                            )

        self.args = args

        self.human_idx = human_idx
        self.num_classes = num_classes
        self.hyper_lambda = args.hyper_lambda
        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.object_class_to_target_class = object_class_to_target_class

        self.num_classes = num_classes

        self.dataset = args.dataset
        self.reserve_indices = reserve_indices

        self.lm_head_embeddings = torch.load("/home/taehoon/HOICLIP/training_linearshortcut/lm_head_embedding_7b.pt", "cpu")
        self.verb_classifier_ho = torch.load("/home/taehoon/HOICLIP/training_linearshortcut/verb_classifier_weights_ho_7b.pt", "cpu").to(torch.bfloat16)
       # self.verb_classifier_ho = F.normalize(self.verb_classifier_ho, p=2, dim=1)
        self.verb_projection_ho = nn.Linear(4096, 117, bias=False).to(torch.bfloat16)
        self.verb_projection_ho.weight.data = self.verb_classifier_ho
        for param in self.verb_projection_ho.parameters():
            param.requires_grad = False

        if self.num_classes == 117:
            # 117 verbs × 3 forms each: (base, -ing, past)
            verb_forms = [
                'adjusting','assembling','blocking',
                'blowing',
                'boarding',
                'breaking',
                'brushing with',
                'buying',
                'carrying',
                'catching',
                'chasing',
                'checking',
                'cleaning',
                'controlling',
                'cooking',
                'cutting',
                'cutting with',
                'directing',
                'dragging',
                'dribbling',
                'drinking with',
                'driving',
                'drying',
                'eating',
                'eating at',
                'exiting',
                'feeding',
                'filling',
                'flipping',
                'flushing',
                'flying',
                'greeting',
                'grinding',
                'grooming',
                'herding',
                'hitting',
                'holding',
                'hopping on',
                'hosing',
                'hugging',
                'hunting',
                'inspecting',
                'installing',
                'jumping',
                'kicking',
                'kissing',
                'lassoing',
                'launching',
                'licking',
                'lying on',
                'lifting',
                'lighting',
                'loading',
                'losing',
                'making',
                'milking',
                'moving',
                'no interaction',
                'opening',
                'operating',
                'packing',
                'painting',
                'parking',
                'paying',
                'peeling',
                'petting',
                'picking',
                'picking up',
                'pointing',
                'pouring',
                'pulling',
                'pushing',
                'racing',
                'reading',
                'releasing',
                'repairing',
                'riding',
                'rowing',
                'running',
                'sailing',
                'scratching',
                'serving',
                'setting',
                'shearing',
                'signing',
                'sipping',
                'sitting at',
                'sitting on',
                'sliding',
                'smelling',
                'spinning',
                'squeezing',
                'stabbing',
                'standing on',
                'standing under',
                'sticking',
                'stirring',
                'stopping at',
                'straddling',
                'swinging',
                'tagging',
                'talking on',
                'teaching',
                'texting on',
                'throwing',
                'tying',
                'toasting',
                'training',
                'turning',
                'typing on',
                'walking',
                'washing',
                'watching',
                'waving',
                'wearing',
                'wielding',
                'zipping',
            ]
            tokenizer = self.clip_head['tokenizer']
            verb_token_ids = []
            for form in verb_forms:
                ids = tokenizer.encode(form)[1:]  # skip BOS
                ids = list(dict.fromkeys(ids))  # deduplicate preserving order
                verb_token_ids.append(ids)

            num_verbs = len(verb_forms)
            max_tokens = max(len(ids) for ids in verb_token_ids)
            # Pad with 0 and create mask
            padded = torch.zeros(num_verbs, max_tokens, dtype=torch.long)
            mask = torch.zeros(num_verbs, max_tokens, dtype=torch.bool)
            for i, ids in enumerate(verb_token_ids):
                padded[i, :len(ids)] = torch.tensor(ids)
                mask[i, :len(ids)] = True
            self.register_buffer('verb_token_ids', padded)      # [num_verbs, max_tokens]
            self.register_buffer('verb_token_mask', mask)

        if self.num_classes == 600:
            tokenizer = self.clip_head['tokenizer']
            hoi_labels = [v.replace("a photo of ", "") for k, v in hico_text_label.hico_text_label.items()]
            hoi_token_ids = [tokenizer.encode(v)[1:] for v in hoi_labels]  # skip BOS
            max_hoi_tokens = max(len(ids) for ids in hoi_token_ids)
            hoi_padded = torch.zeros(len(hoi_labels), max_hoi_tokens, dtype=torch.long)
            hoi_mask = torch.zeros(len(hoi_labels), max_hoi_tokens, dtype=torch.bool)
            for i, ids in enumerate(hoi_token_ids):
                hoi_padded[i, :len(ids)] = torch.tensor(ids)
                hoi_mask[i, :len(ids)] = True
            self.register_buffer('hoi_token_ids', hoi_padded)    # [600, max_hoi_tokens]
            self.register_buffer('hoi_token_mask', hoi_mask) 
        
        # self.text_2_queries = MLP(4096, 128, args.adapt_dim, 2)
        # self.ho_llava_2_queries = nn.Linear(4096, args.adapt_dim, bias = False)
        # self.h_llava_2_queries = nn.Linear(4096, args.adapt_dim, bias = False)
        # self.o_llava_2_queries = nn.Linear(4096, args.adapt_dim, bias = False)
        #self.ho_query_proj = MLP(512, 128, 4096, 2).to(torch.bfloat16)
        self.h_query_proj = MLP(256, 128, 4096, 2).to(torch.bfloat16)
        self.o_query_proj = MLP(256, 128, 4096, 2).to(torch.bfloat16)
        # self.ho_text_query_proj = MLP(args.adapt_dim*2, 128, args.adapt_dim, 2)


        if args.zs:
            self.zs_type = args.zs_type
            self.filtered_hoi_idx = hico_text_label.hico_unseen_index[self.zs_type]
        else:
            self.filtered_hoi_idx = []
            self.zs_type = None

        self.seen_verb_idxs = list(set([HOI_IDX_TO_ACT_IDX[idx] for idx in range(600) if idx not in self.filtered_hoi_idx]))
        
        if self.num_classes == 117:
            self.unseen_verb_idxs = [i for i in range(self.num_classes) if i not in self.seen_verb_idxs]
        elif self.num_classes == 600:
            self.seen_hoi_idxs = [i for i in range(self.num_classes) if i not in self.filtered_hoi_idx]

        start_layer = 31
        num_layers = 1

        #Clone layers for H (human)
        cloned_layers_h = nn.ModuleList([
            copy.deepcopy(self.clip_head["model"].model.layers[i])
            for i in range(start_layer, start_layer + num_layers)
        ])
        self.cross_attend_lora_h = CrossAttendWithLoRA(cloned_layers_h, lora_rank=16, lora_alpha=16)

        # Clone layers for O (object)
        cloned_layers_o = nn.ModuleList([
            copy.deepcopy(self.clip_head["model"].model.layers[i])
            for i in range(start_layer, start_layer + num_layers)
        ])
        self.cross_attend_lora_o = CrossAttendWithLoRA(cloned_layers_o, lora_rank=16, lora_alpha=16)

        # #Clone layers for HO (human + object union)
        # cloned_layers_ho = nn.ModuleList([
        #     copy.deepcopy(self.clip_head["model"].model.layers[i])
        #     for i in range(start_layer, start_layer + num_layers)
        # ])
        # self.cross_attend_lora_ho = CrossAttendWithLoRA(cloned_layers_ho, lora_rank=16, lora_alpha=16)


        final_norm = self.clip_head["model"].model.norm
        self.final_norm = copy.deepcopy(final_norm)
        for param in self.final_norm.parameters():
            param.requires_grad = False

        # self.spatial_head = nn.Sequential(
        #     nn.Linear(36, 128), nn.ReLU(),
        #     nn.Linear(128, 256), nn.ReLU(),
        #     nn.Linear(256, 4096), nn.ReLU(),
        # ).to(torch.bfloat16)

    def _reset_parameters(self):  ## xxx
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
    
    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
    ) -> Tensor:  ### √

        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else self.hyper_lambda
  
        s_h = scores[x].pow(p)# * torch.sigmoid(ing_logits).pow(p)
        s_o = scores[y].pow(p)#* torch.sigmoid(ed_logits).pow(p)


            
        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        #import pdb; pdb.set_trace()
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]
        #import pdb; pdb.set_trace()
        return torch.stack([prior_h, prior_o])
    

    def compute_sim_scores(self, region_props: List[dict], image, targets, priors=None):
        device = image.tensors.device
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        # pairwise_tokens_collated = []
        all_logits_collated = []
        distill_collated = []
        all_boxes_collated = []
        all_feats_collated = []
        # get updated HO tokens.
        for b_idx, props in enumerate(region_props):
            # local_features = features[b_idx]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            feats = props['feat']

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)

            # Permute human instances to the top
            if not torch.all(labels[:n_h]==self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]
            # Skip image when there are no valid human-object pairs
            if n_h == 0 or n <= 1:
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
            #import pdb; pdb.set_trace()
                continue
            # Get the pairwise indices
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)

            
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            #import pdb; pdb.set_trace()
            h_unique_indices, h_inverse_indices = torch.unique(x_keep, return_inverse=True)
            o_unique_indices, o_inverse_indices = torch.unique(y_keep, return_inverse=True)


            #human = labels[x_keep] 
            #ho_detr_feats = self.ho_query_proj(torch.cat([feats[x_keep],feats[y_keep]],dim=-1).to(torch.bfloat16))
            h_detr_feats = self.h_query_proj(feats[h_unique_indices].to(torch.bfloat16))
            o_detr_feats = self.o_query_proj(feats[o_unique_indices].to(torch.bfloat16))
            #objects = targets[0]["object"]#.unique()

            # pairwise_spatial = compute_spatial_encodings(
            #         [boxes[x.flatten()], ], [boxes[y.flatten()], ], [(336,336), ]
            # ).to(torch.bfloat16)
            # pairwise_spatial = self.spatial_head(pairwise_spatial)
            # pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)
            # pairwise_spatial_reshaped = pairwise_spatial_reshaped[x_keep, y_keep]

            objects = labels[y_keep].unique()

            gt_bx_h = self.recover_boxes(targets[0]['boxes_h'], targets[0]['size'])
            gt_bx_o = self.recover_boxes(targets[0]['boxes_o'], targets[0]['size'])
            
            bbox_2_tokens = bbox_to_token((336,336),boxes, 24)
            bool_h = bbox_2_tokens[x_keep]
            bool_o = bbox_2_tokens[y_keep]
            bool_union = bool_h | bool_o

            #generated_text = self.clip_head['tokenizer'].batch_decode(output.sequences, skip_special_token==True)[0].strip()
            with torch.no_grad():
                hidden_states, output, generated_ids, gen_scores = generate_llava_model(
                    self.clip_head,
                    self.clip_head['model_name'],
                    image.decompose()[0][b_idx:b_idx + 1].to(torch.bfloat16),
                    (336,336),
                    self.clip_head['tokenizer'],
                    hidden_states=True,
                    text_prompt="Describe all human-object interactions in this image."
                )

            gen_text = self.clip_head['tokenizer'].decode(generated_ids.squeeze(0), skip_special_tokens=True)
            tokens = word_tokenize(gen_text)
            tagged = pos_tag(tokens)

            #gen_scores = torch.stack(gen_scores, dim=0).squeeze(1)  # [T-1, V]
            gen_list = generated_ids.squeeze(0).tolist()  # [T]

            gen_verb_dict = {}
            for word, tag in tagged:
                if not tag.startswith("VB"):
                    continue
                word_ids = self.clip_head['tokenizer'].encode(word, add_special_tokens=False)
                n = len(word_ids)
                if n == 0:
                    continue
                for t in range(len(gen_list) - n + 1):
                    if gen_list[t:t+n] == word_ids:
                        for j in range(n):
                            score_idx = t + j - 1
                            tid = word_ids[j]
                            if score_idx < 0 or score_idx >= gen_scores.shape[0]:
                                continue
                            prob = torch.softmax(gen_scores[score_idx], dim=-1)[tid]
                            if tid not in gen_verb_dict or prob > gen_verb_dict[tid]:
                                gen_verb_dict[tid] = prob
                        break



            x_boxes = boxes[x_keep]  # shape: (N, 4)
            y_boxes = boxes[y_keep]  # shape: (N, 4)


            x1 = torch.min(x_boxes[:, 0], y_boxes[:, 0])
            y1 = torch.min(x_boxes[:, 1], y_boxes[:, 1])
            x2 = torch.max(x_boxes[:, 2], y_boxes[:, 2])
            y2 = torch.max(x_boxes[:, 3], y_boxes[:, 3])

            union_boxes = torch.stack([x1, y1, x2, y2], dim=1)  # shape: (N, 4)
            union_tokens  = bbox_to_token((336,336),union_boxes, 24)

            
            boxes_xyxy = boxes[h_unique_indices]  # shape [K, 4]
            batch_indices = torch.zeros((boxes_xyxy.size(0),), dtype=torch.long, device=boxes_xyxy.device)  # shape [K]
            roi_boxes = torch.cat([batch_indices[:, None].float(), boxes_xyxy], dim=1)  # shape [K, 5]

            boxes_xyxy1 = boxes[o_unique_indices]  # shape [K, 4]
            batch_indices1 = torch.zeros((boxes_xyxy1.size(0),), dtype=torch.long, device=boxes_xyxy1.device)  # shape [K]
            roi_boxes1 = torch.cat([batch_indices1[:, None].float(), boxes_xyxy1], dim=1)  # shape [K, 5]


            boxes_xyxy2 = union_boxes  # shape [K, 4]
            batch_indices2 = torch.zeros((boxes_xyxy2.size(0),), dtype=torch.long, device=boxes_xyxy2.device)  # shape [K]
            roi_boxes2 = torch.cat([batch_indices2[:, None].float(), boxes_xyxy2], dim=1)  # shape [K, 5]


            llava_features = hidden_states[self.args.layer]

            llava_feat_for_roi = llava_features.float().view(1, 24, 24, 4096).permute(0, 3, 1, 2)
            h_feats0 = torchvision.ops.roi_align(llava_feat_for_roi, roi_boxes,output_size=(7, 7),spatial_scale=24 / 336, aligned=True)
            o_feats0 = torchvision.ops.roi_align(llava_feat_for_roi, roi_boxes1,
                                                    output_size=(7, 7),
                                                    spatial_scale=24 / 336, aligned=True)
            ho_feats0 = torchvision.ops.roi_align(llava_feat_for_roi, roi_boxes2,
                                                    output_size=(7, 7),
                                                    spatial_scale=24 / 336, aligned=True)

            # Convert back to original dtype
            ho_feats0 = ho_feats0.flatten(2).mean(-1).unsqueeze(0).to(llava_features.dtype)
            h_feats0 = h_feats0.flatten(2).mean(-1).unsqueeze(0).to(llava_features.dtype)
            o_feats0 = o_feats0.flatten(2).mean(-1).unsqueeze(0).to(llava_features.dtype)



            bool_o_inverse = bbox_2_tokens[o_inverse_indices]  # [num_unique_o, 576]
            bool_h_inverse = bbox_2_tokens[h_inverse_indices]  # [num_unique_o, 576]
           # import pdb; pdb.set_trace()
            #import pdb;pdb.set_trace()
            h_out  = self.cross_attend_lora_h(h_feats0[0][h_inverse_indices].unsqueeze(0) + h_detr_feats[h_inverse_indices], llava_features, bool_o_inverse) 
            o_out = self.cross_attend_lora_o(o_feats0[0][o_inverse_indices].unsqueeze(0) + o_detr_feats[o_inverse_indices], llava_features, bool_h_inverse) 
            #ho_out = self.cross_attend_lora_o(ho_feats0 + ho_detr_feats, llava_features) 

            # h_out  = self.cross_attend_lora_h(h_feats0 + h_detr_feats, llava_features, ) 
            # o_out = self.cross_attend_lora_o(o_feats0 + o_detr_feats, llava_features,) 
            # ho_out = self.cross_attend_lora_o(ho_feats0 + ho_detr_feats, llava_features,) 
            #logits_reg = ho_feats_final @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
            normed_h = self.final_norm(h_out.squeeze(0))# - lm_head_t.mean(1, keepdim=True).T #+ self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16) # [N, 4096]+ self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16)
            normed_o = self.final_norm(o_out.squeeze(0)) #- lm_head_t.mean(1, keepdim=True).T
            #normed_ho = self.final_norm(ho_out.squeeze(0))# - ln(1, keepdim=True).T  #+ self.object_embedding[labels[o_inverse_indices]].to(torch.bfloat16) # [N, 4096]

            lm_head_t = self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16).detach() 
            centered_lm_head = lm_head_t

            
            h_vocab = normed_h @ centered_lm_head
            o_vocab = normed_o @ centered_lm_head
            #ho_vocab = normed_ho @ centered_lm_head

            #topk_ho_vocab = torch.topk(ho_vocab, k=256, dim=-1).values
            topk_h_vocab = torch.topk(h_vocab, k=256, dim=-1).values
            topk_o_vocab = torch.topk(o_vocab, k=256, dim=-1).values

            if self.num_classes == 117:
                #import pdb; pdb.set_trace()
                h_verb_scores = h_vocab[:, self.verb_token_ids] #- global_verb_scores # [N, 117, max_tokens] - global_verb_logits.unsqueeze(0)
                h_verb_scores = h_verb_scores.masked_fill(~self.verb_token_mask.unsqueeze(0), 0.0)

                o_verb_scores = o_vocab[:, self.verb_token_ids] #- global_verb_scores # [N, 117, max_tokens]
                o_verb_scores = o_verb_scores.masked_fill(~self.verb_token_mask.unsqueeze(0), 0.0)

                # ho_verb_scores = ho_vocab[:, self.verb_token_ids] #- global_verb_scores # [N, 117, max_tokens]
                # ho_verb_scores = ho_verb_scores.masked_fill(~self.verb_token_mask.unsqueeze(0), 0.0)

                #decaying weighted sum
                # max_tokens = self.verb_token_mask.shape[-1]
                # positions = torch.arange(max_tokens, device=self.verb_token_mask.device, dtype=h_verb_scores.dtype)
                # decay = torch.exp(-0.5 * positions)  # [max_tokens], e.g. [1.0, 0.61, 0.37, 0.22, ...]
                # decay = decay * self.verb_token_mask  # [117, max_tokens]
                # decay_sum = decay.sum(dim=-1).unsqueeze(0)  # [1, 117]


                logits_o = o_verb_scores.sum(dim=-1) / self.verb_token_mask.sum(dim=-1).unsqueeze(0)
                logits_h = h_verb_scores.sum(dim=-1) / self.verb_token_mask.sum(dim=-1).unsqueeze(0)
            #     #logits_h = (h_verb_scores * decay.unsqueeze(0)).sum(dim=-1) / decay_sum
            #     #logits_o = (o_verb_scores * decay.unsqueeze(0)).sum(dim=-1) / decay_sum
            #     #logits_ho = (ho_verb_scores * decay.unsqueeze(0)).sum(dim=-1) / decay_sum
            # if self.num_classes == 600:
            #     h_hoi_scores = h_vocab[:, self.hoi_token_ids]  # [N, 117, max_tokens]
            #     h_hoi_scores = h_hoi_scores.masked_fill(~self.hoi_token_mask.unsqueeze(0), 0.0)
            #     logits_h = h_hoi_scores.sum(dim=-1) / self.hoi_token_mask.sum(dim=-1).unsqueeze(0) #+ lens_per_pair[-1] # [N, 117]

            #     o_hoi_scores = o_vocab[:, self.hoi_token_ids]  # [N, 117, max_tokens]
            #     o_hoi_scores = o_hoi_scores.masked_fill(~self.hoi_token_mask.unsqueeze(0), 0.0)
            #     logits_o = o_hoi_scores.sum(dim=-1) / self.hoi_token_mask.sum(dim=-1).unsqueeze(0)
                
                #logits_h = h_verb_scores.sum(dim=-1) / self.verb_token_mask.sum(dim=-1).unsqueeze(0) #+ lens_per_pair[-1] # [N, 117]


                #logits_h = h_verb_scores.min(dim=-1).values 
                ##o_verb_scores = o_vocab[:, self.verb_token_ids] #- global_verb_scores # [N, 117, max_tokens]
                #o_verb_scores = o_verb_scores.masked_fill(~self.verb_token_mask.unsqueeze(0), 0.0)
                #logits_o = o_verb_scores.min(dim=-1).values 
                #logits_o = o_verb_scores.sum(dim=-1) / self.verb_token_mask.sum(dim=-1).unsqueeze(0) #+ lens_per_pair[-1] # [N, 117] + lens_per_pair
                
                # ho_verb_scores = ho_vocab[:, self.verb_token_ids] #- global_verb_scores # [N, 117, max_tokens]
                # ho_verb_scores = ho_verb_scores.masked_fill(~self.verb_token_mask.unsqueeze(0), 0.0)
                # #logits_o = o_verb_scores.min(dim=-1).values 
                #logits_ho = ho_verb_scores.sum(dim=-1) / self.verb_token_mask.sum(dim=-1).unsqueeze(0) #+ lens_

            # probs_ho = torch.sigmoid(self.verb_projection_ho(normed_ho).unsqueeze(-1) - topk_ho_vocab.unsqueeze(1)).mean(-1) #-  #- normed_ho @ self.mu.T #- global_logit
            # probs_h = torch.sigmoid(self.verb_projection_ho(normed_h).unsqueeze(-1) - topk_h_vocab.unsqueeze(1)).mean(-1) #torch.log(torch.sigmoid(self.verb_projection_ho(normed_h).unsqueeze(-1) - topk_h_vocab.unsqueeze(1)).sum(-1)) / torch.log(torch.tensor(100.0)) #-  #- normed_ho @ self.mu.T #- global_logit #- normed_h @ self.mu.T  #- global_logit
            # probs_o = torch.sigmoid(self.verb_projection_ho(normed_o).unsqueeze(-1) - topk_o_vocab.unsqueeze(1)).mean(-1) #torch.log(torch.sigmoid(self.verb_projection_ho(normed_o).unsqueeze(-1) - topk_o_vocab.unsqueeze(1)).sum(-1)) / torch.log(torch.tensor(100.0)) #-  #- normed_ho @ self.mu.T #- global_logit #- normed_o @ self.mu.T #- global_logit


            #probs_ho = torch.sigmoid(logits_ho.unsqueeze(-1) - topk_ho_vocab.unsqueeze(1)).mean(-1) #-  #- normed_ho @ self.mu.T #- global_logit
            probs_h = torch.sigmoid(logits_h.unsqueeze(-1) - topk_h_vocab.unsqueeze(1)).mean(-1) #torch.log(torch.sigmoid(self.verb_projection_ho(normed_h).unsqueeze(-1) - topk_h_vocab.unsqueeze(1)).sum(-1)) / torch.log(torch.tensor(100.0)) #-  #- normed_ho @ self.mu.T #- global_logit #- normed_h @ self.mu.T  #- global_logit
            probs_o = torch.sigmoid(logits_o.unsqueeze(-1) - topk_o_vocab.unsqueeze(1)).mean(-1) #torch.log(torch.sigmoid(self.verb_projection_ho(normed_o).unsqueeze(-1) - topk_o_vocab.unsqueeze(1)).sum(-1)) / torch.log(torch.tensor(100.0)) #-  #- normed_ho @ self.mu.T #- global_logit #- normed_o @ self.mu.T #- global_logit
            
            logits = (probs_h+ probs_o) / 2
            #logits = (probs_h[h_inverse_indices] + probs_o[o_inverse_indices]+ probs_ho) / 3
            #logits1= (probs_h[h_inverse_indices] + probs_o[o_inverse_indices])/2
            #import pdb; pdb.set_trace()
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            #gt_labels = torch.ones([N])torch.sigmoid(logits)torch.sigmoid(logits)
            all_logits_collated.append(logits)
            #all_llava_logits_collated.append(llava_target)
            areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) / (336*336)
            # print(self.clip_head['tokenizer'].decode(torch.topk(o_vocab[7], 100, dim=-1)[1]))
            #
            if len(gen_verb_dict) > 0:
                gv_ids = torch.tensor(list(gen_verb_dict.keys()), device=device)
                gv_probs = torch.stack(list(gen_verb_dict.values())).detach()
                distill_collated.append((h_vocab, o_vocab, topk_h_vocab, topk_o_vocab, gv_ids, gv_probs))
            else:
                distill_collated.append(None)
            # bbox_2_tokens[o_inverse_indices][1].nonzero()
            #import pdb; pdb.set_trace()

        return all_logits_collated, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated, distill_collated #all_llava_logits_collated

    def recover_boxes(self, boxes, size):
        #import pdb; pdb.set_trace()
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        #import pdb; pdb.set_trace()
        return boxes

    
    def associate_with_ground_truth(self, boxes_h, boxes_o, targets): ## for training
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

       # import pdb; pdb.set_trace()
        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])

        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)
        # print("pair gt,",len(x),len(y))
        # IndexError: tensors used as indices must be long, byte or bool tensors
        #import pdb; pdb.set_trace()
        if self.num_classes == 117 or self.num_classes == 24 or self.num_classes == 407:
            labels[x, targets['labels'][y]] = 1  ## target['labels']: verb/action
        else:
            labels[x, targets['hoi'][y]] = 1
        # print("#(labels==1) = ", torch.sum(labels))
        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets, distill_data): ### loss
        ## bx, bo: indices of boxes

        #import pdb; pdb.set_trace()
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])

        #import pdb; pdb.set_trace()
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        #import pdb; pdb.set_trace()
        #llava_logits = torch.cat(llava_logits)
        logits = torch.cat(logits)
        #model_probs = torch.sigmoid(logits)

        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]

        n_p = len(torch.nonzero(labels))

        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss(
            prior * logits,
            labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )
        distill_loss = torch.tensor(0.0, device=logits.device)
        n_distill = 0
        for item in distill_data:
            if item is None:
                continue
            h_v, o_v, topk_h, topk_o, gv_ids, gv_probs = item
            target = gv_probs.unsqueeze(0).float()  # [1, K]

            pred_h = torch.sigmoid(h_v.float()[:, gv_ids].unsqueeze(-1) - topk_h.float().unsqueeze(1)).mean(-1)  # [N, K]
            pred_o = torch.sigmoid(o_v.float()[:, gv_ids].unsqueeze(-1) - topk_o.float().unsqueeze(1)).mean(-1)  # [N, K]

            distill_loss = distill_loss + binary_focal_loss(
                pred_h, target.expand_as(pred_h), reduction='sum',
                alpha=self.alpha, gamma=self.gamma
            ) + binary_focal_loss(
                pred_o, target.expand_as(pred_o), reduction='sum',
                alpha=self.alpha, gamma=self.gamma
            )
            n_distill += pred_h.numel() + pred_o.numel()

        #import pdb; pdb.set_trace()
        # loss = binary_focal_loss_with_logits(
        # torch.log(
        #     prior / (1 + torch.exp(-logits) - prior) + 1e-8
        # ), labels, reduction='sum',
                # alpha=self.alpha, gamma=self.gamma
                # )

        #Distillation loss


        if dist.is_initialized():
            n_distill = torch.as_tensor([float(n_distill)], device=logits.device)
            dist.barrier()
            dist.all_reduce(n_distill)
            dist.all_reduce(distill_loss)
            n_distill = n_distill.item()

        if n_distill > 0:
            distill_loss = distill_loss / n_distill

        alpha_distill = 0.1
        #import pdb; pdb.set_trace()
        return (loss / n_p) + alpha_distill * distill_loss


    def prepare_region_proposals(self, results): ## √ detr extracts the human-object pairs
        region_props = []
        for res in results:
            sc, lb, bx, feat = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            feat = feat[keep].view(-1,256)

            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum(); n_object = len(keep) - n_human
            # Keep the number of human and object instances in a specified interval
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                feat=feat[keep]
            ))

        return region_props

    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        batch_size = len(images)
        images_orig = [im[0].float() for im in images]
        images_clip = [im[1] for im in images]
        device = images_clip[0].device
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images_clip
        ], device=device)
        image_sizes_orig = torch.as_tensor([
            im.size()[-2:] for im in images_orig
            ], device=device)
        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)

        self.detector.eval()
        features, pos = self.detector.backbone(images_orig.to(device))
        src, mask = features[-1].decompose()
        hs, detr_memory = self.detector.transformer(self.detector.input_proj(src), mask, self.detector.query_embed.weight, pos[-1])
        outputs_class = self.detector.class_embed(hs) # 6x8x100x81 or 6x8x100x92
        outputs_coord = self.detector.bbox_embed(hs).sigmoid() # 6x8x100x4
        if self.dataset == 'vcoco' and outputs_class.shape[-1] == 92:
            outputs_class = outputs_class[:, :, :, self.reserve_indices]
            assert outputs_class.shape[-1] == 81, 'reserved shape NOT match 81'
        
        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'feats': hs[-1]}
        results = self.postprocessor(results, image_sizes)
        region_props = self.prepare_region_proposals(results)
        images_clip = nested_tensor_from_tensor_list(images_clip)
        #priors = self.get_prior(region_props,image_sizes)
        #import pdb; pdb.set_trace()
        logits, prior, bh, bo, objects, distill_data = self.compute_sim_scores(region_props,images_clip,targets, None )
        boxes = [r['boxes'] for r in region_props]
        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets, distill_data)

            loss_dict = dict(
                interaction_loss=interaction_loss
            )


            return loss_dict

        if len(logits) == 0:
            print(targets)
            return None

        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
        #import pdb; pdb.set_trace()
        return detections

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes): ### √
        n = [len(b) for b in bh]
        logits = torch.cat(logits)
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, size,  in zip(
                boxes, bh, bo, logits, prior, objects, image_sizes,
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            #this is for logits
            #scores = torch.sigmoid(lg[x, y])
            scores = lg[x, y]
            #import pdb; pdb.set_trace()
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], size=size
            ))

        return detections


###added rank
def build_detector(args, class_corr, object_n_verb_to_interaction, clip_model_path, rank):
    # build DETR
    num_classes = 80
    if args.dataset == 'vcoco' and 'e632da11' in args.pretrained:
        num_classes = 91
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    detr = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )


    postprocessors = {'bbox': PostProcess()}


    if os.path.exists(args.pretrained):
        #if dist.get_rank() == 0:
        print(f"Load weights for the object detector from {args.pretrained}")
        if 'e632da11' in args.pretrained:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model']) 
        else:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])



    model_name = "llava7b" 

    if model_name.startswith("llava"):
        model = load_llava_state(rank)

    if args.num_classes == 117:
        classnames = hico_verbs_sentence
    elif args.num_classes == 24:
        classnames = vcoco_verbs_sentence
    elif args.num_classes == 600:
        classnames = list(hico_text_label.hico_text_label.values())
    else:
        raise NotImplementedError

    #model = CustomCLIP(args, classnames=classnames, clip_model=clip_model)

    obj_class_names = [obj[1] for obj in hico_text_label.hico_obj_text_label]
    object_embedding = torch.load("/home/taehoon/HOICLIP/training_linearshortcut/obj_classifier_7b.pt", "cpu")
    object_embedding = object_embedding.clone().detach()


    #import pdb; pdb.set_trace()
    detector = HOILLAVA(args,
        detr, postprocessors['bbox'], model, object_embedding,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        object_class_to_target_class=class_corr,
        object_n_verb_to_interaction=object_n_verb_to_interaction,
    )

    return detector

