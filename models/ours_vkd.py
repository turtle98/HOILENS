"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>
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
import random

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

class VKD_backbone(nn.Module):
    def __init__(self,
        args,
        detector: nn.Module,
        postprocessor: nn.Module,
        model: nn.Module,
        vkd_shortcut: nn.Module,
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
        self.args = args
        self.vkd_shortcut = vkd_shortcut
        self.proj = MLP(2048, 1024, 1024, 2)

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

    def _reset_parameters(self):  ## xxx
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def compute_loss(self, output, tgt_layer, criterion): ### loss
        result = output
        tgt = tgt_layer

        loss = criterion(result, tgt) 
        #import pdb; pdb.set_trace()
        n_p = tgt.numel()  # batchpergpu

        if dist.is_initialized():
            # Sync n_p across all ranks
            n_p_tensor = torch.as_tensor([n_p], device=loss.device, dtype=torch.float32)
            dist.all_reduce(n_p_tensor, op=dist.ReduceOp.SUM)
            n_p = n_p_tensor.item()  # scalar

            # Sync the total loss across GPUs (keep gradient)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            # global average across GPUs
            loss = loss / n_p
        else:
            loss = loss / n_p

        #mport pdb; pdb.set_trace()
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
        criterion = torch.nn.MSELoss(reduction='sum')
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
        images_clip = nested_tensor_from_tensor_list(images_clip)

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
        #device = image.tensors.device
        #import pdb; pdb.set_trace()
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        # pairwise_tokens_collated = []
        all_logits_collated = []
        all_boxes_collated = []
        all_ing_logits_collated = []
        all_ed_logits_collated = []
        all_orig_feats_collated = []
        all_feats_collated = []
        all_mvid_collated = []
        all_zs_logits_collated = []

        batch_hidden_states = []
        batch_last_hidden_states = []
        batch_cls_proj = []
        with torch.inference_mode():
            for b_idx, props in enumerate(region_props):
                boxes = props['boxes']
                scores = props['scores']
                labels = props['labels']
                feats = props['feat']
                if self.args.cls_token:
                    text_prompt = "Provide 5 single word actions that can be visually identified between humans and objects in this image."
                else:
                    text_prompt = "."

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
                boxes_xyxy = boxes[x_keep]  # shape [K, 4]
                batch_indices = torch.zeros((boxes_xyxy.size(0),), dtype=torch.long, device=boxes_xyxy.device)  # shape [K]
                roi_boxes_h = torch.cat([batch_indices[:, None].float(), boxes_xyxy], dim=1)  # shape [K, 5]

                boxes_xyxy1 = boxes[y_keep]  # shape [K, 4]
                batch_indices1 = torch.zeros((boxes_xyxy1.size(0),), dtype=torch.long, device=boxes_xyxy1.device)  # shape [K]
                roi_boxes_o = torch.cat([batch_indices1[:, None].float(), boxes_xyxy1], dim=1)  # shape [K, 5]pe [K, 5]
                self.clip_head["model"].model.mm_projector.half()

                ho_feats, hidden_states, last_hidden_states, caption = run_llava_model(
                    self.clip_head,
                    self.clip_head['model_name'],
                    images_clip.decompose()[0][b_idx:b_idx + 1].half(),
                    (336,336),
                    self.clip_head['tokenizer'],
                    box_h=roi_boxes_h,
                    box_o=roi_boxes_o,
                    hidden_states=True,
                    text_prompt=text_prompt
                )
                self.clip_head["model"].model.mm_projector.float()
                #import pdb; pdb.set_trace()
                proj_out = self.proj(ho_feats.float())   # FP16
                proj_out = proj_out.float()              # convert to FP32 for fp32 projector
                ho_proj = self.clip_head["model"].model.mm_projector(proj_out)
                #import pdb; pdb.set_trace()

                

                batch_hidden_states.append(
                    hidden_states[self.args.layer].to(device)
                )
                batch_last_hidden_states.append(
                    last_hidden_states[self.args.layer].to(device).expand(ho_proj.shape[0], -1)
                )
                #last_hidden_states[self.args.layer].to(device).expand(ho_proj.shape[0], -1).shape
                batch_cls_proj.append(ho_proj.to(device))
        batch_last_hidden_states = torch.cat(batch_last_hidden_states, dim=0)
        batch_hidden_states = torch.cat(batch_hidden_states, dim=0)
        #batch_last_hidden_states = torch.stack(batch_last_hidden_states)
        batch_cls_proj = torch.cat(batch_cls_proj, dim=0)
        import pdb; pdb.set_trace()
        if self.args.cls_token: 
            src = torch.cat([batch_cls_proj,batch_hidden_states[:,0]], dim=1).float()
            #import pdb; pdb.set_trace()
            tgt = torch.cat([batch_last_hidden_states[:,self.args.layer].unsqueeze(1),batch_hidden_states[:,self.args.layer]], dim=1).float()
        else:
            src = batch_hidden_states[:,0].float()
            tgt = batch_hidden_states[:,self.args.layer].float()
        
        #import pdb; pdb.set_trace()
        result = self.vkd_shortcut(src)
        

        if self.training:
            mse_loss = self.compute_loss(result, tgt, criterion)

            loss_dict = dict(
               mse_loss = mse_loss
            )

            #import pdb; pdb.set_trace()
            return loss_dict
        return

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

class shortcut(nn.Module):
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


    def forward(self, hidden_states):
        out = self.probe(hidden_states)
        out = self.norm(out)
        return out
    

def build_detector(args, class_corr, object_n_verb_to_interaction, clip_model_path, rank):

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
    
    if args.num_classes == 117:
        classnames = hico_verbs_sentence
    elif args.num_classes == 24:
        classnames = vcoco_verbs_sentence
    elif args.num_classes == 600:
        classnames = list(hico_text_label.hico_text_label.values())
    else:
        raise NotImplementedError
    
    model_name = "llava13b" 

    if model_name.startswith("llava"):
        model = load_llava_state(rank)
    vkd_shortcut = shortcut(5120, args.linear)

    obj_class_names = [obj[1] for obj in hico_text_label.hico_obj_text_label]
    object_embedding = torch.load("/workspace/internal_objects_5_alllayers.pt", "cpu")
    object_embedding = object_embedding.clone().detach()

    detector = VKD_backbone(args,
        detr, postprocessors['bbox'], model, vkd_shortcut,
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

