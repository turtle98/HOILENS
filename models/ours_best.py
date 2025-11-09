"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

'''
boxes_xyxy = boxes[h_unique_indices]  # shape [K, 4]
batch_indices = torch.zeros((boxes_xyxy.size(0),), dtype=torch.long, device=boxes_xyxy.device)  # shape [K]
roi_boxes = torch.cat([batch_indices[:, None].float(), boxes_xyxy], dim=1)  # shape [K, 5]

boxes_xyxy1 = boxes[o_unique_indices]  # shape [K, 4]
batch_indices1 = torch.zeros((boxes_xyxy1.size(0),), dtype=torch.long, device=boxes_xyxy1.device)  # shape [K]
roi_boxes1 = torch.cat([batch_indices1[:, None].float(), boxes_xyxy1], dim=1)  # shape [K, 5]


boxes_xyxy2 = union_boxes  # shape [K, 4]
batch_indices2 = torch.zeros((boxes_xyxy2.size(0),), dtype=torch.long, device=boxes_xyxy2.device)  # shape [K]
roi_boxes2 = torch.cat([batch_indices2[:, None].float(), boxes_xyxy2], dim=1)  # shape [K, 5]

h_feats = torchvision.ops.roi_align(llava_features.permute(1, 2, 0).view(-1, 5120, 24, 24),  roi_boxes,
                                        output_size=(7, 7),
                                    spatial_scale=24 / 336, aligned=True)
o_feats = torchvision.ops.roi_align(llava_features.permute(1, 2, 0).view(-1, 5120, 24, 24),  roi_boxes1,
                                        output_size=(7, 7),
                                        spatial_scale=24 / 336, aligned=True)
ho_feats = torchvision.ops.roi_align(llava_features.permute(1, 2, 0).view(-1, 5120, 24, 24),  roi_boxes2,
                            output_size=(7, 7),
                            spatial_scale=24 / 336, aligned=True)
decode_logits = o_feats0 @ self.lm_head_embeddings.transpose(1,0).to(o_feats0.device)
(Pdb) decode_logits.shape
torch.Size([6, 32000])
(Pdb) topk_values, topk_indices = torch.topk(decode_logits, k =5)
'''
The Australian National University
Australian Centre for Robotic Vision
"""
import os
import sys

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

from utils.hico_list import hico_verbs_sentence
from utils.vcoco_list import vcoco_verbs_sentence
from utils.hico_utils import reserve_indices
from utils.postprocessor import PostProcess
from utils.ops import binary_focal_loss_with_logits, vectorized_bboxes_and_indices, bbox_to_token
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
import math

#from llava.model.multimodal_projector.builder import build_vision_projector
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import AutoConfig
sys.path.pop(0)

"""
body_parts = ["mouth", "eyes","arms","hand","feet","leg"]
"""
def standardize_and_topk(logits, k=117):
    mean = logits.mean(dim=1, keepdim=True)
    std = logits.std(dim=1, keepdim=True) + 1e-6
    logits_std = (logits - mean) / std

    # Keep only top-k per row
    topk_vals, topk_idx = torch.topk(logits_std, k, dim=1)
    mask = torch.zeros_like(logits_std)
    mask.scatter_(1, topk_idx, 1.0)
    logits_topk = logits_std * mask
    return logits_topk

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

class Llavaproj(nn.Module):
    def __init__(self, in_dim=5120, out_dim=256):
        super().__init__()
        self.encoder = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        z = self.encoder(x)  # projection
        z_normed = self.norm(z)
        return z_normed

class verbsteer(nn.Module):
    def __init__(self, in_dim=5120, out_dim=5120, rank=128):
        super().__init__()
        self.lora_up = Llavaproj(rank, 5120)
        self.norm = nn.LayerNorm(rank)
        self.decoder = TransformerDecoder(input_dim = 5120, embed_dim=rank, num_heads=8, num_layers=1)
        self.llava_proj =  Llavaproj(5120, rank)

    def forward(self, x, detr_feats, boxes, size, llava_feats, obj_embeds):
        x_down = self.norm(x)
        x_down = x + detr_feats + obj_embeds
        x_down = self.decoder(x_down.unsqueeze(0), self.llava_proj(llava_feats))
        out = self.lora_up(x_down)
        return out.squeeze(0), x


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(input_dim,embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, queries, keys_values):
        for layer in self.layers:
            queries = layer(queries, keys_values)
        return self.final_norm(queries)

class DecoderBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)


        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, queries, keys_values):
        q = self.norm1(queries)
        self_attn_output, _ = self.self_attn(q, q, q)  # Self-attention
        queries = queries + self.dropout1(self_attn_output)  # Residual
        q = self.norm2(queries)
        #cross_attn_output, _ = self.cross_attn(self.cross_attn_q_proj(q), self.cross_attn_k_proj(keys_values), self.cross_attn_v_proj(keys_values))
        cross_attn_output, _ = self.cross_attn(q, keys_values, keys_values)
        queries = queries + self.dropout2(cross_attn_output)  # Residual connection
        # Pre-Norm Feedforward
        ffn_input = self.norm3(queries)
        ffn_output = self.ffn(ffn_input)
        output = queries + self.dropout3(ffn_output)  # Residual
        return output

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
        self.postprocessor = postprocessor
        self.clip_head = model

        self.register_buffer("object_embedding",object_embedding)
        self.visual_output_dim = 5120
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

        self.h_steer = verbsteer(in_dim=5120, out_dim=5120, rank=args.adapt_dim)
        self.o_steer = verbsteer(in_dim=5120, out_dim=5120, rank=args.adapt_dim)
        self.ho_steer = verbsteer(in_dim=5120, out_dim=5120, rank=args.adapt_dim)
        self.lm_head_embeddings = torch.load("/hub_data1/taehoonsong/HOICLIP/training_linearshortcut/lm_head_embedding.pt", "cpu")

        self.verb_classifier_ho = torch.load("/hub_data1/taehoonsong/LAIN/internal_verbs_5_alllayers.pt", "cpu").float()
        #self.verb_classifier_ho = self.clip_head.linear_shortcut(self.verb_classifier_ho)
        self.verb_classifier_ho = F.normalize(self.verb_classifier_ho[:, self.args.layer], p=2, dim=1)
        self.verb_projection_ho = nn.Linear(5120, 117, bias=False)
        self.verb_projection_ho.weight.data = self.verb_classifier_ho
        for param in self.verb_projection_ho.parameters():
            param.requires_grad = False

        self.text_2_queries = MLP(5120, 128, args.adapt_dim, 2)#nn.Linear(5120, args.adapt_dim, bias = False)
        self.ho_llava_2_queries = nn.Linear(5120, args.adapt_dim, bias = False)
        self.h_llava_2_queries = nn.Linear(5120, args.adapt_dim, bias = False) #nn.Linear(5120, args.adapt_dim, bias = False)
        self.o_llava_2_queries =nn.Linear(5120, args.adapt_dim, bias = False) #nn.Linear(5120, args.adapt_dim, bias = False)
        self.ho_query_proj = MLP(512, 128, args.adapt_dim, 2)
        self.h_query_proj = MLP(256, 128, args.adapt_dim, 2)
        self.o_query_proj = MLP(256, 128, args.adapt_dim, 2)
        self.ho_text_query_proj = MLP(args.adapt_dim*2, 128, args.adapt_dim, 2)

        self.h_alpha_logit = nn.Parameter(torch.tensor(0.0)) 
        self.ho_alpha_logit = nn.Parameter(torch.tensor(0.0)) 
        self.o_alpha_logit = nn.Parameter(torch.tensor(0.0)) 

    def _reset_parameters(self):  ## xxx
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    
    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor,
    ) -> Tensor:  ### √

        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else self.hyper_lambda
  
        s_h = scores[x].pow(p) #* torch.sigmoid(ing_logits).pow(p)
        s_o = scores[y].pow(p) #* torch.sigmoid(ed_logits).pow(p)


        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
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
        # get updated HO tokens.
        self.clip_head.eval()
        for b_idx, props in enumerate(region_props):
            # local_features = features[b_idx]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            feats = props['feat']

    
            llava_features1 = self.clip_head(image.decompose()[0][b_idx:b_idx + 1])

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if self.args.cls_token:
                cls_token = llava_features1[:,0]
                llava_features = llava_features1[:,1:]
            else:
                llava_features = llava_features1
                # torch.topk(cls_token @ self.lm_head_embeddings.T.to(cls_token.device), 20 ,-1)
            #import pdb; pdb.set_trace()
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
            #i mport pdb; pdb.set_trace()
                continue
            # Get the pairwise indices
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            # Valid human-object pairs
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)

            
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            #import pdb; pdb.set_trace()
            h_unique_indices, h_inverse_indices = torch.unique(x_keep, return_inverse=True)
            o_unique_indices, o_inverse_indices = torch.unique(y_keep, return_inverse=True)

            #import pdb; pdb.set_trace()
            ho_detr_feats = self.ho_query_proj(torch.cat([feats[x_keep],feats[y_keep]],dim=-1))
            h_detr_feats = self.h_query_proj(feats[h_unique_indices])
            o_detr_feats = self.o_query_proj(feats[o_unique_indices])

            text_2_query = self.text_2_queries(F.normalize(self.object_embedding[:,self.args.layer].float(), 2 -1))
            h_text = text_2_query[labels[h_unique_indices]]
            o_text = text_2_query[labels[o_unique_indices]]
            ho_text = self.ho_text_query_proj(torch.cat([h_text[h_inverse_indices],o_text[o_inverse_indices]],dim=-1)) #+ ing_dir.unsqueeze(0)
           
            bbox_2_tokens = bbox_to_token((336,336),boxes, 24)
            x_boxes = boxes[x_keep]  # shape: (N, 4)
            y_boxes = boxes[y_keep]  # shape: (N, 4)

            # Union box: min of top-left corner, max of bottom-right corner
            x1 = torch.min(x_boxes[:, 0], y_boxes[:, 0])
            y1 = torch.min(x_boxes[:, 1], y_boxes[:, 1])
            x2 = torch.max(x_boxes[:, 2], y_boxes[:, 2])
            y2 = torch.max(x_boxes[:, 3], y_boxes[:, 3])

            union_boxes = torch.stack([x1, y1, x2, y2], dim=1)  

            boxes_xyxy = boxes[h_unique_indices]  # shape [K, 4]
            batch_indices = torch.zeros((boxes_xyxy.size(0),), dtype=torch.long, device=boxes_xyxy.device)  # shape [K]
            roi_boxes = torch.cat([batch_indices[:, None].float(), boxes_xyxy], dim=1)  # shape [K, 5]

            boxes_xyxy1 = boxes[o_unique_indices]  # shape [K, 4]
            batch_indices1 = torch.zeros((boxes_xyxy1.size(0),), dtype=torch.long, device=boxes_xyxy1.device)  # shape [K]
            roi_boxes1 = torch.cat([batch_indices1[:, None].float(), boxes_xyxy1], dim=1)  # shape [K, 5]


            boxes_xyxy2 = union_boxes  # shape [K, 4]
            batch_indices2 = torch.zeros((boxes_xyxy2.size(0),), dtype=torch.long, device=boxes_xyxy2.device)  # shape [K]
            roi_boxes2 = torch.cat([batch_indices2[:, None].float(), boxes_xyxy2], dim=1)  # shape [K, 5]

            h_feats0 = torchvision.ops.roi_align(llava_features.view(1, 24, 24, 5120).permute(0, 3, 1, 2),  roi_boxes,
                                                    output_size=(7, 7),
                                                spatial_scale=24 / 336, aligned=True)
            o_feats0 = torchvision.ops.roi_align(llava_features.view(1, 24, 24, 5120).permute(0, 3, 1, 2),  roi_boxes1,
                                                    output_size=(7, 7),
                                                    spatial_scale=24 / 336, aligned=True)
            ho_feats0 = torchvision.ops.roi_align(llava_features.view(1, 24, 24, 5120).permute(0, 3, 1, 2),  roi_boxes2,
                                        output_size=(7, 7),
                                        spatial_scale=24 / 336, aligned=True) # (B, D)


            ho_feats0 = ho_feats0.flatten(2).mean(-1)
            h_feats0 = h_feats0.flatten(2).mean(-1)
            o_feats0 = o_feats0.flatten(2).mean(-1)
            if self.args.cls_token:
                h_feats1 = self.h_llava_2_queries(h_feats0 +cls_token)
                o_feats1 = self.o_llava_2_queries(o_feats0+cls_token)
                ho_feats1 = self.ho_llava_2_queries(ho_feats0+cls_token)
            else:
                h_feats1 = self.h_llava_2_queries(h_feats0)
                o_feats1 = self.o_llava_2_queries(o_feats0)
                ho_feats1 = self.ho_llava_2_queries(ho_feats0)


            h_tokens , _ = self.h_steer(h_feats1, h_detr_feats, boxes[h_unique_indices], targets[b_idx]['size'], llava_features1, h_text)
            o_tokens , _ = self.o_steer(o_feats1, o_detr_feats, boxes[o_unique_indices], targets[b_idx]['size'], llava_features1, o_text) 
            ho_tokens, _ = self.ho_steer(ho_feats1, ho_detr_feats, union_boxes, targets[b_idx]['size'], llava_features1, ho_text)


            ho_logits = self.verb_projection_ho(ho_tokens + self.ho_alpha_logit*ho_feats0) #/ math.sqrt(5120)
            h_logits = self.verb_projection_ho(h_tokens + self.h_alpha_logit*h_feats0) #/ math.sqrt(5120)
            o_logits = self.verb_projection_ho(o_tokens + self.o_alpha_logit*o_feats0) #/ math.sqrt(5120)
            logits = (ho_logits + h_logits[h_inverse_indices] + o_logits[o_inverse_indices])/3
            if self.args.zse:
                ho_logits1 = self.verb_projection_ho(ho_feats0) #/ math.sqrt(5120)
                h_logits1 = self.verb_projection_ho(h_feats0) #/ math.sqrt(5120)
                o_logits1 = self.verb_projection_ho(o_feats0) #/ math.sqrt(5120)

                # Standardize each row
                mean = ho_logits1.mean(dim=1, keepdim=True)
                std = ho_logits1.std(dim=1, keepdim=True) + 1e-6  # prevent division by zero
                ho_logits_std1 = (ho_logits1 - mean) / std

                mean = h_logits1.mean(dim=1, keepdim=True)
                std = h_logits1.std(dim=1, keepdim=True) + 1e-6  # prevent division by zero
                h_logits_std1 = (h_logits1 - mean) / std

                mean = o_logits1.mean(dim=1, keepdim=True)
                std = o_logits1.std(dim=1, keepdim=True) + 1e-6  # prevent division by zero
                o_logits_std1 = (o_logits1 - mean) / std
                logits = (ho_logits + ho_logits_std1+ h_logits[h_inverse_indices] + h_logits_std1[h_inverse_indices] + o_logits[o_inverse_indices] + o_logits_std1[o_inverse_indices])/3
            
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            all_logits_collated.append(logits)
            #import pdb; pdb.set_trace()

        return all_logits_collated, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated

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

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets): ### loss
        ## bx, bo: indices of boxes

        #import pdb; pdb.set_trace()
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])


        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        logits = torch.cat(logits)
        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]

        # compute average L2 norm squared per sample globally
        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        

        #import pdb; pdb.set_trace()
        loss = binary_focal_loss_with_logits(
        torch.log(
            prior / (1 + torch.exp(-logits) - prior) + 1e-8
        ), labels, reduction='sum',
        alpha=self.alpha, gamma=self.gamma
        )
        #import pdb; pdb.set_trace()
        return (loss / n_p)# + reg_loss / n_e
    
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

        #import pdb; pdb.set_trace()

        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)
            #import pdb; pdb.set_trace()
        self.detector.eval()
        features, pos = self.detector.backbone(images_orig)
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
        logits, prior, bh, bo, objects = self.compute_sim_scores(region_props,images_clip,targets, None)
        boxes = [r['boxes'] for r in region_props]
        #import pdb; pdb.set_trace()
        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets)

            loss_dict = dict(
                interaction_loss=interaction_loss
            )
            return loss_dict

        if len(logits) == 0:
            print(targets)
            return None

        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
       # import pdb; pdb.set_trace()
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
            scores = torch.sigmoid(lg[x, y])

            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], size=size
            ))
            #import pdb; pdb.set_trace()

        return detections

@torch.no_grad()
def get_obj_text_emb(args, clip_model, obj_class_names):
    obj_text_inputs = torch.cat([tokenize(obj_text) for obj_text in obj_class_names])
    with torch.no_grad():
        obj_text_embedding = clip_model.encode_text(obj_text_inputs)
        object_embedding = obj_text_embedding
        # obj_text_embedding = obj_text_embedding[hoi_obj_list,:]
    return object_embedding

class MLPProbe(nn.Module):
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

class linearshortcut(nn.Module):
    def __init__(self, llava_dim, linear):
        super().__init__()
        self.d = llava_dim
        self.linear = linear
        self.initialize_probe()
        self.norm = nn.LayerNorm(self.d)

    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Linear(self.d, self.d)
        else:
            self.probe = MLPProbe(self.d, 1024, self.d, 3)


    def normalize(self, x):
        norm = x.norm(p=2, dim=1, keepdim=True)  # L2 norm across features
        normalized_x = x / (norm)          # avoid division by 0
        return normalized_x


    def forward(self, hidden_states):
        x = hidden_states
        #x = self.normalize(hidden_states)
        p = self.probe(x)
        p = self.norm(p)

        return p
    

class projector(nn.Module):
    def __init__(self):
        super(self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_features=1024, out_features=5120, bias=True),
            nn.GELU(approximate='none'),
            nn.Linear(in_features=5120, out_features=5120, bias=True)
        )

    def forward(self, x):
        return self.model(x)

class VisionProjector(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # Load frozen vision tower
        self.args = args
        self.vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower_name)
        #self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        # Build and load projector
        config_path = "/hub_data1/taehoonsong/HOICLIP/training_linearshortcut/config.json"
        config = AutoConfig.from_pretrained(config_path)
        self.projector = build_vision_projector(config)
        projector_weights = torch.load(
            "/hub_data1/taehoonsong/HOICLIP/training_linearshortcut/mm_projector_weights.pth",
            map_location="cpu"
        )
        projector_weights = {k: v.half() for k, v in projector_weights.items()}
        self.projector.load_state_dict(projector_weights, strict=True)
        
        # Load linear shortcut (⚡ NO device assignment yet)
        self.linear_shortcut = linearshortcut(5120, args.linear)

        shortcut_weights = torch.load(
            f"/hub_data1/taehoonsong/LAIN/checkpoints/CVPR/VKD_training_nores_withln/no_cls_token/{self.args.layer}/ckpt_94080_10.pt",
            map_location="cpu"
        )
        #import pdb; pdb.set_trace()
        state_dict = shortcut_weights['model_state_dict']
        new_state_dict = {k.replace("vkd_shortcut.", ""): v for k, v in state_dict.items()}

        # Load into your model
        if args.baseline: 
            for p in self.parameters():
                p.requires_grad = False
            for p in self.linear_shortcut.parameters():
                p.requires_grad = True
        else:
            self.linear_shortcut.load_state_dict(new_state_dict, strict=True)
            # Freeze everything
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        # Assume input x is already on device
        vision_out = self.vision_tower(x, output_hidden_states=True)
        #import pdb; pdb.set_trace()
        if self.args.cls_token: 
            hidden_states = vision_out.hidden_states[-2][:, ]
        else:
            hidden_states = vision_out.hidden_states[-2][:, 1:]  # take patch tokens
        #import pdb; pdb.set_trace()
        output = self.projector(hidden_states)
        #import pdb; pdb.set_trace()
        if self.args.linear_shortcut:
            output = self.linear_shortcut(output)
        #mport pdb; pdb.set_trace()
        return output


def build_detector(args, class_corr, object_n_verb_to_interaction, clip_model_path):
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
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        if 'e632da11' in args.pretrained:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model']) 
        else:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])

    if dist.get_rank() == 0:
        print(f"Load weights for the distilled Llava")
    model = VisionProjector(args)


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

    object_embedding = torch.load("/hub_data1/taehoonsong/LAIN/internal_objects_5_alllayers.pt", "cpu")
    object_embedding = object_embedding.clone().detach()

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

