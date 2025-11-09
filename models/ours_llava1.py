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

#from llava.model.multimodal_projector.builder import build_vision_projector
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import AutoConfig
sys.path.pop(0)

"""
body_parts = ["Mouth", "Eyes","Arms","Hand","Feet","Legs"]



"""

from methods.llava_utils import retrieve_logit_lens_llava, load_llava_state, run_llava_model

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
    def __init__(self, body_parts, in_dim=5120, out_dim=5120, rank=128):
        super().__init__()

        # LoRA adaptation: low-rank decomposition A (down) and B (up)
        #self.lora_down =   Llavaproj(5120, rank) #nn.Linear(in_dim, rank, bias = False) #
        self.lora_up = Llavaproj(rank, 5120)
        #nn.init.zeros_(self.lora_up.weight)
        #self.scale = nn.Parameter(torch.tensor(0.0))
        #self.body_parts = body_parts.mean(0) 
        # self.ffn = nn.Sequential(
        #     nn.Linear(rank*3, rank),
        #     nn.ReLU(),
        #     nn.Linear(rank, rank),
        #     nn.LayerNorm(rank)
        # )
        #self.pos_linear = MLP(256, 128, rank, 2)
        self.norm = nn.LayerNorm(rank)
        #self.text_norm = nn.LayerNorm(rank)
        # Freeze the original input (assumed identity in your case) or use external frozen encoder
        # Here, we assume x is identity-mapped (like residual)\
        self.decoder = TransformerDecoder(input_dim = 5120, embed_dim=rank, num_heads=8, num_layers=1)
        self.llava_proj =  Llavaproj(5120, rank)
        #self.norm2 = nn.LayerNorm(rank)
    def compute_box_pe(self, boxes, embeds, image_size):
        bx_norm = boxes / image_size[[1, 0, 1, 0]]
        bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2
        b_wh = bx_norm[:, 2:] - bx_norm[:, :2]

        c_pe = compute_sinusoidal_pe(bx_c[:, None], 20, embeds.shape[-1]//2).squeeze(1)
        wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20, embeds.shape[-1]//2).squeeze(1)

        box_pe = torch.cat([c_pe, wh_pe], dim=-1)

        return box_pe 

    def forward(self, x, detr_feats, boxes, size, llava_feats, obj_embeds):
        x_down = self.norm(x)
        #x_down = (x)
        #t_down = self.text_down(obj_embeds)
       # import pdb; pdb.set_trace()
        #x_down = self.ffn(torch.cat([x_down, obj_embeds, self.pos_linear((self.compute_box_pe(boxes, x_down, size)))], dim=-1))
        x_down += detr_feats + obj_embeds #+ self.pos_linear(self.compute_box_pe(boxes, x_down, size)) #+ self.detr_proj(detr_feats)
        # x_down1 = self.norm2(x_down)
        x_down = self.decoder(x_down.unsqueeze(0), self.llava_proj(llava_feats))
        out = self.lora_up(x_down)
        #out = lora_out.squeeze(0)
        return out.squeeze(0), x


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(input_dim,embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        #self.out_proj = Llavaproj(embed_dim, input_dim) 

    def forward(self, queries, keys_values):
        for layer in self.layers:
            queries = layer(queries, keys_values)
        return self.final_norm(queries)

class DecoderBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # self.q_proj = nn.Linear(embed_dim, embed_dim)
        # self.k_proj = nn.Linear(embed_dim, embed_dim)
        # self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        #self.norm3 = nn.LayerNorm(embed_dim)
        #self.norm4 = nn.LayerNorm(embed_dim)
        #self.cross_attn_q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        #self.cross_attn_k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        #self.cross_attn_v_proj = nn.Linear(embed_dim, embed_dim, bias=False)


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
        # self_attn_output, _ = self.self_attn(q, q, q
        queries = queries + self.dropout1(self_attn_output)  # Residual

        q = self.norm2(queries)
        #cross_attn_output, _ = self.cross_attn(self.cross_attn_q_proj(q), self.cross_attn_k_proj(keys_values), self.cross_attn_v_proj(keys_values))
        cross_attn_output, _ = self.cross_attn(q, keys_values, keys_values)
        queries = queries + self.dropout2(cross_attn_output)  # Residual connection
        # Pre-Norm Feedforward
        ffn_input = self.norm3(queries)
        ffn_output = self.ffn(ffn_input)
        output = queries + self.dropout3(ffn_output)  # Residual
        #import pdb; pdb.set_trace()
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
        # self.projector = projector
        # self.linear_shortcut = linear_shortcut
        # for p in self.linear_shortcut.parameters():
        #     p.requires_grad= False
        # for p in self.projector.parameters():
        #     p.requires_grad= False
        self.detector = detector
        self.postprocessor = postprocessor
        self.clip_head = model

        self.register_buffer("object_embedding",object_embedding)

       # self.visual_output_dim = model.image_encoder.output_dim
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

        #self.priors_initial_dim = self.visual_output_dim + 5
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))_
        self.lm_head_object_embedding = torch.load("/hub_data1/taehoonsong/HOICLIP/training_linearshortcut/obj_classifier_tensor.pt", "cpu")
        

        self.ed = torch.load("/hub_data1/taehoonsong/HOICLIP/training_linearshortcut/ing_direction.pt", "cpu")
        self.ing = torch.load("/hub_data1/taehoonsong/HOICLIP/training_linearshortcut/ed_direction.pt", "cpu")

        self.body_parts_classifier = torch.load("/hub_data1/taehoonsong/HOICLIP/training_linearshortcut/body_parts_weights.pt", "cpu")
       # #self.register_buffer("body_parts_classifier",self.body_parts_classifier)
        #self.priors_downproj = MLP(self.priors_initial_dim, 128, args.adapt_dim, 3) # old 512+5
        self.h_steer = verbsteer(self.body_parts_classifier, in_dim=5120, out_dim=5120, rank=args.adapt_dim)
        self.o_steer = verbsteer(self.body_parts_classifier, in_dim=5120, out_dim=5120, rank=args.adapt_dim)
        self.ho_steer = verbsteer(self.body_parts_classifier, in_dim=5120, out_dim=5120, rank=args.adapt_dim)
        self.lm_head_embeddings = torch.load("/hub_data1/taehoonsong/HOICLIP/training_linearshortcut/lm_head_embedding.pt", "cpu")
        self.verb_classifier_h = torch.load("/hub_data1/taehoonsong/HOICLIP/training_linearshortcut/verb_classifier_weights_ing.pt", "cpu")
        self.verb_projection_h = nn.Linear(5120, 117, bias=False)
        self.verb_projection_h.weight.data = self.verb_classifier_h #+ self.body_parts_classifier #/ self.verb_classifier_h.norm(dim=-1, keepdim= True)
        for param in self.verb_projection_h.parameters():
            param.requires_grad = False

        self.verb_classifier_ho = torch.load("/hub_data1/taehoonsong/HOICLIP/training_linearshortcut/verb_classifier_weights_ing.pt", "cpu")
        self.verb_projection_ho = nn.Linear(5120, 117, bias=False)
        self.verb_projection_ho.weight.data = self.verb_classifier_ho #+ self.body_parts_classifier #/ self.verb_classifier_ho.norm(dim=-1, keepdim= True)
        for param in self.verb_projection_ho.parameters():
            param.requires_grad = False

        self.verb_classifier_ed = torch.load("/hub_data1/taehoonsong/HOICLIP/training_linearshortcut/verb_classifier_weights_ed.pt", "cpu")
        self.verb_projection_ed = nn.Linear(5120, 117, bias=False)
        self.verb_projection_ed.weight.data = self.verb_classifier_ed #+ self.body_parts_classifier#/ self.verb_classifier_ed.norm(dim=-1, keepdim= True)
        for param in self.verb_projection_ed.parameters():
            param.requires_grad = False
        

        # self.logit_scale_h = nn.Parameter(torch.ones([]) * np.log(1 / 0.7)) 
        # self.logit_scale_ho = nn.Parameter(torch.ones([]) * np.log(1 / 0.7)) 
        # self.logit_scale_o = nn.Parameter(torch.ones([]) * np.log(1 / 0.7)) 

    def _reset_parameters(self):  ## xxx
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def compute_box_coord(self, boxes, image_size):
        bx_norm = boxes / image_size[[1, 0, 1, 0]]
        bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2
        b_wh = bx_norm[:, 2:] - bx_norm[:, :2]

        return torch.cat([bx_c,b_wh],dim=-1)
    
    
    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor, ing_logits: Tensor, ed_logits: Tensor,
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
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]
        #import pdb; pdb.set_trace()
        return torch.stack([prior_h, prior_o])
    
    def compute_masked_od_feature(self, mask, features):
        mask = mask.unsqueeze(-1).to(dtype=features.dtype)  # [B, T, 1]

        # Apply mask
        weighted_feat = features * mask         # [B, T, D]
        probs = F.softmax(features, dim = -1)
        max_probs, argmax_classes = probs.max(dim=-1)
        import pdb; pdb.set_trace()


        return weighted_feat#, weighted_feat
    

    def compute_masked_feature(self, mask, features):
        mask = mask.unsqueeze(-1).to(dtype=features.dtype)  # [B, T, 1]

        # Apply mask
        weighted_feat = features * mask         # [B, T, D]
        summed_feat = weighted_feat.sum(dim=1)  # [B, D]

        # Normalizepooled = torch.matmul(attn_weights, features_exp).squeeze(1)
        mask_sum = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        avg_feat = summed_feat / mask_sum           # [B, D]

        return avg_feat#, weighted_feat

    def compute_sim_scores(self, region_props: List[dict], image, targets, priors=None):
        device = image.tensors.device
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        # pairwise_tokens_collated = []
        all_logits_collated = []
        all_boxes_collated = []
        all_ing_logits_collated = []
        all_ed_logits_collated = []
        all_orig_feats_collated = []
        all_feats_collated = []
        # get updated HO tokens.
        for b_idx, props in enumerate(region_props):
            # local_features = features[b_idx]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            feats = props['feat']
            #import pdb; pdb.set_trace()
            
            input_ids, output = run_llava_model(
                self.clip_head['model'],
                self.clip_head["model_name"],
                image.decompose()[0][b_idx:b_idx + 1].half(),
                (336,336),
                self.clip_head["tokenizer"],
                hidden_states=True,
                text_prompt="."
            )

            # hidden_states = torch.stack(output.hidden_states[0])
            # image_token_index = input_ids.tolist()[0].index(-200)
            # hidden_states = hidden_states[:, :, image_token_index : image_token_index + (24 * 24),:].float()
            hidden_states = output.hidden_states[0][self.args.layer]
            image_token_index = input_ids.tolist()[0].index(-200)
            llava_features = hidden_states[:, image_token_index : image_token_index + (24 * 24),:].float()
            #import pdb; pdb.set_trace()
            #llava_features = self.clip_head(image.decompose()[0][b_idx:b_idx + 1])

            # body_logits = llava_features @ self.body_parts_classifier.to(llava_features.device).transpose(1,0)
            # #import pdb; pdb.set_trace()
            # #import pdb; pdb.set_trace()
            # bboxes_tensor_h, class_labels_h = vectorized_bboxes_and_indices(body_logits.squeeze(0), img_width=336, img_height=336, grid_size=24,threshold_prob=0.95, threshold_logit_min=10, threshold_logit_max=20)
            ########################
            # 7/15 더 많은 verb 형태 가져오기
            # human box를 mining with body logits?
            # human box안에 body text mining해서 explicit query
            #######################
            
            #import pdb; pdb.set_trace()
            #import pdb; pdb.set_trace()
      

            # #import pdb; pdb.set_trace()
            # if bboxes_tensor.numel() != 0:
            #     boxes = torch.cat([boxes, bboxes_tensor], dim=0)
            #     labels = torch.cat([labels, class_labels], dim=0)
            #     scores = torch.cat([scores, 0.2*torch.ones(bboxes_tensor.shape[0], device = device, dtype = torch.float32)])
            #     feats = torch.cat([feats, torch.zeros(bboxes_tensor.shape[0],256, device = device)])
            #import pdb; pdb.set_trace()

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
                all_ed_logits_collated.append(torch.zeros(0, device=device))
                all_ing_logits_collated.append(torch.zeros(0, device=device))
                all_orig_feats_collated.append(torch.zeros(0, device=device))
                all_feats_collated.append(torch.zeros(0, device=device))
                if bboxes_tensor.numel() != 0:
                    all_boxes_collated.append(bboxes_tensor)
                else:
                    all_boxes_collated.append(None)
                continue
            #import pdb; pdb.set_trace()
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

           
            bbox_2_tokens = bbox_to_token((336,336),boxes, 24)
            x_boxes = boxes[x_keep]  # shape: (N, 4)
            y_boxes = boxes[y_keep]  # shape: (N, 4)

            # Union box: min of top-left corner, max of bottom-right corner
            x1 = torch.min(x_boxes[:, 0], y_boxes[:, 0])
            y1 = torch.min(x_boxes[:, 1], y_boxes[:, 1])
            x2 = torch.max(x_boxes[:, 2], y_boxes[:, 2])
            y2 = torch.max(x_boxes[:, 3], y_boxes[:, 3])

            union_boxes = torch.stack([x1, y1, x2, y2], dim=1)  # shape: (N, 4)
            union_tokens  = bbox_to_token((336,336),union_boxes, 24)

            

            #import pdb; pdb.set_trace()
            # h_feats0  = self.compute_masked_feature(bbox_2_tokens[h_unique_indices], llava_features) #+ self.object_embedding[labels[h_unique_indices]]
            # o_feats0 = self.compute_masked_feature(bbox_2_tokens[o_unique_indices], llava_features) #+ self.object_embedding[labels[o_unique_indices]]
            #ho_feats0 = self.compute_masked_feature(union_tokens, llava_features) #+ (self.object
            boxes_xyxy2 = union_boxes  # shape [K, 4]
            batch_indices2 = torch.zeros((boxes_xyxy2.size(0),), dtype=torch.long, device=boxes_xyxy2.device)  # shape [K]
            roi_boxes2 = torch.cat([batch_indices2[:, None].float(), boxes_xyxy2], dim=1)  # shape [K, 5]

            ho_feats = torchvision.ops.roi_align(llava_features.view(1, 24, 24, 5120).permute(0, 3, 1, 2), roi_boxes2,
                            output_size=(7, 7),
                            spatial_scale=24 / 336, aligned=True)
            ho_feats0 = ho_feats.mean(dim=[2, 3])  # shape: [K, 5120]
            #import pdb; pdb.set_trace()
            # orig_path = targets[0]['filename']
            # filename = os.path.basename(orig_path) 
            # if "train" in filename: # e.g., HICO_train2015_00000048.jpg
            #     stem = os.path.splitext(filename)[0]   # removes .jpg
            #     save_path = os.path.join(self.args.save_dir,'train', f"{stem}.pt")
            #     torch.save(llava_features.cpu(), save_path)
            # else:
            #     stem = os.path.splitext(filename)[0]   # removes .jpg
            #     save_path = os.path.join(self.args.save_dir, 'test', f"{stem}.pt")
            #     torch.save(llava_features.cpu(), save_path)
            #import pdb; pdb.set_trace()
            

            bboxes_tensor = torch.empty((0, 4), dtype=torch.float, device=device)


            ho_logits = self.verb_projection_ho(ho_feats0)
            #h_logits = self.verb_projection_h(h_feats0)
            #o_logits = self.verb_projection_ed(o_feats0)
            ing_logits = torch.zeros(0, device=device)
            ed_logits = torch.zeros(0, device=device)
            logits = (ho_logits) #+ o_logits[o_inverse_indices] + h_logits[h_inverse_indices])

            # ing_logits = ((h_feats0) @ self.ing.to(h_feats0.device).T)[h_inverse_indices]
            # ed_logits = ((o_feats0) @ self.ed.to(o_feats0.device).T)[o_inverse_indices]
            
           # import pdb; pdb.set_trace()
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels, ing_logits, ed_logits)
            )
            all_logits_collated.append(logits)
            all_ed_logits_collated.append(ed_logits)
            all_ing_logits_collated.append(ing_logits)
            all_orig_feats_collated.append(feats)
            all_feats_collated.append(feats)
            if bboxes_tensor.numel() != 0:
                all_boxes_collated.append(bboxes_tensor)
            else:
                all_boxes_collated.append(None)

            gt_o_boxes = self.recover_boxes(targets[b_idx]['boxes_o'], targets[b_idx]['size'])
            #import pdb; pdb.set_trace()

#, 'boxes_o': tensor([[0.2051, 0.2116, 0.4101, 0.1223],
      #  [0.1921, 0.2012, 0.3842, 0.1536],
      #  [0.1815, 0.2103, 0.3631, 0.1406]]
        return all_logits_collated, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated, all_boxes_collated, all_ed_logits_collated, all_ing_logits_collated, all_orig_feats_collated, all_feats_collated

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

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets, ed_logits, ing_logits, orig_feats, feats): ### loss
        ## bx, bo: indices of boxes

        #import pdb; pdb.set_trace()
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])


        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        #import pdb; pdb.set_trace()
        logits = torch.cat(logits)
        ed_logits = torch.cat(ed_logits)
        ing_logits = torch.cat(ing_logits)
        collapsed_labels = (labels.sum(dim=1, keepdim=True) > 0).float()
       # import pdb; pdb.set_trace()
        loss2 = binary_focal_loss_with_logits(
            ed_logits.view(-1),
            collapsed_labels.view(-1),
            reduction='sum',
            alpha=self.alpha,
            gamma=self.gamma
        )

        loss3 = binary_focal_loss_with_logits(
            ing_logits.view(-1),
            collapsed_labels.view(-1),
            reduction='sum',
            alpha=self.alpha,
            gamma=self.gamma
        )


        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]
        n_i = len(torch.nonzero(collapsed_labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_i = torch.as_tensor([n_i], device='cuda')
            dist.barrier()
            dist.all_reduce(n_i)
            n_i = (n_i / world_size).item()


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


        # orig_feats = torch.cat(orig_feats, dim=0)
        # feats = torch.cat(feats, dim=0)
        # reg_loss = F.mse_loss(orig_feats, feats, reduction='sum')

        # # reg_loss = F.l1_loss(feats, orig_feats, reduction='sum')

        # # # Compute total number of elements across all GPUs
        # num_elements = torch.tensor([feats.numel()], dtype=torch.float32, device=feats.device)
        # if dist.is_initialized():
        #     dist.all_reduce(reg_loss)
        #     dist.all_reduce(num_elements)  # total number of elements across GPUs
        #     total_elements = num_elements.item()
        # else:
        #     total_elements = feats.numel()

        # # Normalize the loss globally
        # reg_loss = reg_loss / (total_elements + 1e-6)

        # text_reg_loss =  self.cross_identity_loss(self.text_2_queries.weight)
        # ho_img_reg_loss =  self.cross_identity_loss(self.ho_llava_2_queries.weight)
        # h_img_reg_loss =  self.cross_identity_loss(self.h_llava_2_queries.weight)
        # o_img_reg_loss =  self.cross_identity_loss(self.o_llava_2_queries.weight)
        #import pdb; pdb.set_trace()
        return (loss / n_p) # 0.1*((loss2 / n_i) + (loss3 / n_i))/2 #0.1*(text_reg_loss + ho_img_reg_loss + h_img_reg_loss + o_img_reg_loss)
    
    def cross_identity_loss(self, W1):
        W_product = W1 @ W1.T                # [d, d]
        I = torch.eye(W_product.size(0), device=W_product.device)
        loss = ((W_product - I) ** 2).sum()

        if dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()

        return loss

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
        #import pdb; pdb.set_trace()
        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)
        #import pdb; pdb.set_trace()
        features, pos = self.detector.backbone(images_orig.to(device))
        src, mask = features[-1].decompose()
        # assert mask is not None2
        #import pdb; pdb.set_trace()
        hs, detr_memory = self.detector.transformer(self.detector.input_proj(src), mask, self.detector.query_embed.weight, pos[-1])
        outputs_class = self.detector.class_embed(hs) # 6x8x100x81 or 6x8x100x92
        outputs_coord = self.detector.bbox_embed(hs).sigmoid() # 6x8x100x4
        #import pdb; pdb.set_trace()
        if self.dataset == 'vcoco' and outputs_class.shape[-1] == 92:
            outputs_class = outputs_class[:, :, :, self.reserve_indices]
            assert outputs_class.shape[-1] == 81, 'reserved shape NOT match 81'

        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'feats': hs[-1]}
        results = self.postprocessor(results, image_sizes)
        region_props = self.prepare_region_proposals(results)
        # with amp.autocast(enabled=True):
       #import pdb; pdb.set_trace()
        images_clip = nested_tensor_from_tensor_list(images_clip)

        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        logits, prior, bh, bo, objects, boxes, ed_logits, ing_logits, orig_feats, feats = self.compute_sim_scores(region_props,images_clip,targets, None )
        #boxes = [r['boxes'] for r in region_props]
        boxes = [torch.cat([region_props[i]['boxes'], boxes[i]], dim=0) if boxes[i] != None else region_props[i]['boxes'] for i in range(len(region_props))]
        #import pdb; pdb.set_trace()
        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets, ed_logits, ing_logits, orig_feats, feats)

            loss_dict = dict(
                interaction_loss=interaction_loss
            )

            # if self.args.local_rank == 0:
            #     wandb.log(loss_dict)

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
            scores = torch.sigmoid(lg[x, y])

            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], size=size
            ))

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
        config_path = "/hub_data1/taehoonsong/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-13b/snapshots/901a44b9113dea67b976e71f58d4e372cf9de81a/config.json"
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
        if args.linear:
            shortcut_weights = torch.load(
                "/hub_data1/taehoonsong/HOICLIP/training_linearshortcut/checkpoint_last.pth",
                map_location="cpu"
            )
        else:
            shortcut_weights = torch.load(
                "/hub_data1/taehoonsong/HOICLIP/training_nonlinearshortcut/10/30/checkpoint_last.pth",
                map_location="cpu"
            )
        self.linear_shortcut.load_state_dict(shortcut_weights['model'], strict=False)

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Assume input x is already on device
        vision_out = self.vision_tower(x, output_hidden_states=True)
        #import pdb; pdb.set_trace()
        hidden_states = vision_out.hidden_states[-2][:, 1:]  # take patch tokens
        #import pdb; pdb.set_trace()
        output = self.projector(hidden_states)
        #import pdb; pdb.set_trace()
        if self.args.linear_shortcut:
            output = self.linear_shortcut(output)
        #mport pdb; pdb.set_trace()
        return output


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

    # detr, _, postprocessors = build_model(args)
    if os.path.exists(args.pretrained):
        #if dist.get_rank() == 0:
        print(f"Load weights for the object detector from {args.pretrained}")
        if 'e632da11' in args.pretrained:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model']) 
        else:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])

    #clip_state_dict = torch.load(clip_model_path, map_location="cpu").state_dict()
    #clip_model = build_model(state_dict=clip_state_dict, use_adapter=args.use_insadapter, adapter_pos=args.adapter_pos, args=args)



    model_name = "llava13b" # or "blip7b"

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

    object_embedding = torch.load("/hub_data1/taehoonsong/HOICLIP/training_linearshortcut/obj_classifier_tensor.pt", "cpu")
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

