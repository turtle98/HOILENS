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


class VKD_backbone(nn.Module):
    def __init__(self,
        args,
        model,
        vkd_shortcut
    ) -> None:
        super().__init__()

        self.clip_head = model
        self.args = args
        self.vkd_shortcut = vkd_shortcut


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
        batch_hidden_states = []
        batch_last_hidden_states = []
        batch_cls_proj = []
        with torch.inference_mode():
            for b_idx, props in enumerate(targets):
                if self.args.cls_token:
                    text_prompt = "Provide 5 single word actions that can be visually identified between humans and objects in this image."
                else:
                    text_prompt = "."
                cls_proj, hidden_states, last_hidden_states, caption = run_llava_model(
                    self.clip_head,
                    self.clip_head['model_name'],
                    images_clip.decompose()[0][b_idx:b_idx + 1].half(),
                    (336,336),
                    self.clip_head['tokenizer'],
                    hidden_states=True,
                    text_prompt=text_prompt
                )

                batch_hidden_states.append(
                    hidden_states.to(device).squeeze(1)
                )
                batch_last_hidden_states.append(
                    last_hidden_states.to(device).squeeze(1)
                )
                batch_cls_proj.append(cls_proj.to(device))

        batch_hidden_states = torch.stack(batch_hidden_states)
        batch_last_hidden_states = torch.stack(batch_last_hidden_states)
        batch_cls_proj = torch.stack(batch_cls_proj)
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
    

def build_detector(args, rank):
    model_name = "llava13b" 

    if model_name.startswith("llava"):
        model = load_llava_state(rank)
    vkd_shortcut = shortcut(5120, args.linear)

    detector = VKD_backbone(args,
        model, vkd_shortcut
    )

    return detector

