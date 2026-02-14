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

h_feats = torchvision.ops.roi_align(llava_features.permute(1, 2, 0).view(-1, 4096, 24, 24),  roi_boxes,
                                        output_size=(7, 7),
                                    spatial_scale=24 / 336, aligned=True)
o_feats = torchvision.ops.roi_align(llava_features.permute(1, 2, 0).view(-1, 4096, 24, 24),  roi_boxes1,
                                        output_size=(7, 7),
                                        spatial_scale=24 / 336, aligned=True)
ho_feats = torchvision.ops.roi_align(llava_features.permute(1, 2, 0).view(-1, 4096, 24, 24),  roi_boxes2,
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
import copy
from utils.hico_list import hico_verbs_sentence
from utils.vcoco_list import vcoco_verbs_sentence
from utils.hico_utils import reserve_indices
from utils.postprocessor import PostProcess
from utils.ops import binary_focal_loss_with_logits,binary_focal_loss, vectorized_bboxes_and_indices, bbox_to_token, compute_spatial_encodings
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
# decoder = create_decoder_from_llama_layer(llama_layer, num_heads=32)


from methods.llava_utils import retrieve_logit_lens_llava, load_llava_state, run_llava_model, compute_conditional_likelihood_llava, get_img_idx
from methods.attention import llama_modify

"""
body_parts = ["mouth", "eyes","arms","hand","feet","leg"]
"""


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
    def __init__(self, in_dim=4096, out_dim=256):
        super().__init__()
        self.encoder = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        z = self.encoder(x)  # projection
        z_normed = self.norm(z)
        return z_normed

class verbsteer(nn.Module):
    def __init__(self, in_dim=4096, out_dim=4096, rank=128):
        super().__init__()
        self.lora_up = Llavaproj(rank, 4096)
        self.norm = nn.LayerNorm(rank)
        self.decoder = TransformerDecoder(input_dim = 4096, embed_dim=rank, num_heads=8, num_layers=1)
        self.llava_proj =  Llavaproj(4096, rank)

    def forward(self, x, detr_feats, boxes, size, llava_feats, obj_embeds):
        x_down = self.norm(x)
        x_down = x + detr_feats + obj_embeds
        x_down = self.decoder(x_down.unsqueeze(0), self.llava_proj(llava_feats))
        out = self.lora_up(x_down)
        return out.squeeze(0), x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, attend_layers=[32]):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(input_dim,embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.attend_layers = attend_layers
        self.final_norm = RMSNorm(embed_dim)

    def forward(self, queries, keys_values):
        for i, layer in enumerate(self.layers):
            queries = layer(queries, keys_values[self.attend_layers[i]])
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
            hidden_states = hidden_states.to(layer.mlp.gate_proj.weight.dtype)
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

        self.verb_classifier_ho = torch.load("/home/taehoon/HOICLIP/training_linearshortcut/verb_classifier_weights_ho_7b.pt", "cpu").float()

        self.verb_projection_ho = nn.Linear(4096, 117, bias=False)
        self.verb_projection_ho.weight.data = self.verb_classifier_ho
        for param in self.verb_projection_ho.parameters():
            param.requires_grad = False

        self.ho_query_proj = MLP(512, 128, 4096, 2)
        self.h_query_proj = MLP(256, 128, 4096, 2)
        self.o_query_proj = MLP(256, 128, 4096, 2)

        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 4096), nn.ReLU(),
        ).to(torch.float32)

        if self.num_classes == 117:
            verb_forms = [
                'adjust', 'adjusting', 'adjusted',
                'assemble', 'assembling', 'assembled',
                'block', 'blocking', 'blocked',
                'blow', 'blowing', 'blew',
                'board', 'boarding', 'boarded',
                'break', 'breaking', 'broke',
                'brush with', 'brushing with', 'brushed with',
                'buy', 'buying', 'bought',
                'carry', 'carrying', 'carried',
                'catch', 'catching', 'caught',
                'chase', 'chasing', 'chased',
                'check', 'checking', 'checked',
                'clean', 'cleaning', 'cleaned',
                'control', 'controlling', 'controlled',
                'cook', 'cooking', 'cooked',
                'cut', 'cutting', 'cut',
                'cut with', 'cutting with', 'cut with',
                'direct', 'directing', 'directed',
                'drag', 'dragging', 'dragged',
                'dribble', 'dribbling', 'dribbled',
                'drink with', 'drinking with', 'drank with',
                'drive', 'driving', 'drove',
                'dry', 'drying', 'dried',
                'eat', 'eating', 'ate',
                'eat at', 'eating at', 'ate at',
                'exit', 'exiting', 'exited',
                'feed', 'feeding', 'fed',
                'fill', 'filling', 'filled',
                'flip', 'flipping', 'flipped',
                'flush', 'flushing', 'flushed',
                'fly', 'flying', 'flew',
                'greet', 'greeting', 'greeted',
                'grind', 'grinding', 'ground',
                'groom', 'grooming', 'groomed',
                'herd', 'herding', 'herded',
                'hit', 'hitting', 'hit',
                'hold', 'holding', 'held',
                'hop on', 'hopping on', 'hopped on',
                'hose', 'hosing', 'hosed',
                'hug', 'hugging', 'hugged',
                'hunt', 'hunting', 'hunted',
                'inspect', 'inspecting', 'inspected',
                'install', 'installing', 'installed',
                'jump', 'jumping', 'jumped',
                'kick', 'kicking', 'kicked',
                'kiss', 'kissing', 'kissed',
                'lasso', 'lassoing', 'lassoed',
                'launch', 'launching', 'launched',
                'lick', 'licking', 'licked',
                'lie on', 'lying on', 'lay on',
                'lift', 'lifting', 'lifted',
                'light', 'lighting', 'lighted',
                'load', 'loading', 'loaded',
                'lose', 'losing', 'lost',
                'make', 'making', 'made',
                'milk', 'milking', 'milked',
                'move', 'moving', 'moved',
                'no interaction', 'no interaction', 'no interaction',
                'open', 'opening', 'opened',
                'operate', 'operating', 'operated',
                'pack', 'packing', 'packed',
                'paint', 'painting', 'painted',
                'park', 'parking', 'parked',
                'pay', 'paying', 'paid',
                'peel', 'peeling', 'peeled',
                'pet', 'petting', 'petted',
                'pick', 'picking', 'picked',
                'pick up', 'picking up', 'picked up',
                'point', 'pointing', 'pointed',
                'pour', 'pouring', 'poured',
                'pull', 'pulling', 'pulled',
                'push', 'pushing', 'pushed',
                'race', 'racing', 'raced',
                'read', 'reading', 'read',
                'release', 'releasing', 'released',
                'repair', 'repairing', 'repaired',
                'ride', 'riding', 'rode',
                'row', 'rowing', 'rowed',
                'run', 'running', 'ran',
                'sail', 'sailing', 'sailed',
                'scratch', 'scratching', 'scratched',
                'serve', 'serving', 'served',
                'set', 'setting', 'set',
                'shear', 'shearing', 'sheared',
                'sign', 'signing', 'signed',
                'sip', 'sipping', 'sipped',
                'sit at', 'sitting at', 'sat at',
                'sit on', 'sitting on', 'sat on',
                'slide', 'sliding', 'slid',
                'smell', 'smelling', 'smelled',
                'spin', 'spinning', 'spun',
                'squeeze', 'squeezing', 'squeezed',
                'stab', 'stabbing', 'stabbed',
                'stand on', 'standing on', 'stood on',
                'stand under', 'standing under', 'stood under',
                'stick', 'sticking', 'stuck',
                'stir', 'stirring', 'stirred',
                'stop at', 'stopping at', 'stopped at',
                'straddle', 'straddling', 'straddled',
                'swing', 'swinging', 'swung',
                'tag', 'tagging', 'tagged',
                'talk on', 'talking on', 'talked on',
                'teach', 'teaching', 'taught',
                'text on', 'texting on', 'texted on',
                'throw', 'throwing', 'threw',
                'tie', 'tying', 'tied',
                'toast', 'toasting', 'toasted',
                'train', 'training', 'trained',
                'turn', 'turning', 'turned',
                'type on', 'typing on', 'typed on',
                'walk', 'walking', 'walked',
                'wash', 'washing', 'washed',
                'watch', 'watching', 'watched',
                'wave', 'waving', 'waved',
                'wear', 'wearing', 'wore',
                'wield', 'wielding', 'wielded',
                'zip', 'zipping', 'zipped',
            ]
            tokenizer = self.clip_head['tokenizer']
            num_forms = 3  # base, -ing, past
            num_verbs = len(verb_forms) // num_forms
            verb_token_ids = []
            for vi in range(num_verbs):
                forms = verb_forms[vi * num_forms : (vi + 1) * num_forms]
                ids = []
                for form in forms:
                    ids.extend(tokenizer.encode(form)[1:])  # skip BOS
                ids = list(dict.fromkeys(ids))  # deduplicate preserving order
                verb_token_ids.append(ids)

            # Expand with LLaVA-generated synonyms
            # import json, os
            # synonyms_path = os.path.join(os.path.dirname(__file__), "..", "verb_synonyms_3.json")
            # if os.path.exists(synonyms_path):
            #     with open(synonyms_path) as f:
            #         verb_synonyms = json.load(f)
            #     # Build -ing form list for synonym lookup (index 1 in each group)
            #     verbs_ing = [verb_forms[vi * num_forms + 1] for vi in range(num_verbs)]
            #     for i, v in enumerate(verbs_ing):
            #         if v in verb_synonyms:
            #             for word in verb_synonyms[v]:
            #                 syn_ids = tokenizer.encode(word.lower())[1:]  # skip BOS
            #                 verb_token_ids[i].extend(syn_ids)
            #             verb_token_ids[i] = list(dict.fromkeys(verb_token_ids[i]))

            max_tokens = max(len(ids) for ids in verb_token_ids)
            # Pad with 0 and create mask
            padded = torch.zeros(num_verbs, max_tokens, dtype=torch.long)
            mask = torch.zeros(num_verbs, max_tokens, dtype=torch.bool)
            for i, ids in enumerate(verb_token_ids):
                padded[i, :len(ids)] = torch.tensor(ids)
                mask[i, :len(ids)] = True
            self.register_buffer('verb_token_ids', padded)      # [117, max_tokens]
            self.register_buffer('verb_token_mask', mask)        # [117, max_tokens]

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


        start_layer = 31
        num_layers = 1

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

        #Clone layers for HO (human + object union)
        cloned_layers_ho = nn.ModuleList([
            copy.deepcopy(self.clip_head["model"].model.layers[i])
            for i in range(start_layer, start_layer + num_layers)
        ])
        self.cross_attend_lora_ho = CrossAttendWithLoRA(cloned_layers_ho, lora_rank=16, lora_alpha=16)

        final_norm = self.clip_head["model"].model.norm
        self.final_norm = copy.deepcopy(final_norm)
        for param in self.final_norm.parameters():
            param.requires_grad = False

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
        #self.clip_head.eval()
        for b_idx, props in enumerate(region_props):
            # local_features = features[b_idx]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            feats = props['feat']

    
            with torch.no_grad():
                hidden_states, output, generated_ids, _= run_llava_model(
                    self.clip_head,
                    self.clip_head['model_name'],
                    image.decompose()[0][b_idx:b_idx + 1].to(torch.bfloat16),
                    (336,336),
                    self.clip_head['tokenizer'],
                    hidden_states=True,
                    text_prompt="."
                )

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            llava_features = hidden_states[self.args.layer].float()
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

            pairwise_spatial = compute_spatial_encodings(
                    [boxes[x.flatten()], ], [boxes[y.flatten()], ], [(336,336), ]
            ).to(torch.float32)
            pairwise_spatial = self.spatial_head(pairwise_spatial)
            pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)
            pairwise_spatial_reshaped = pairwise_spatial_reshaped[x_keep, y_keep]

            ho_detr_feats = self.ho_query_proj(torch.cat([feats[x_keep],feats[y_keep]],dim=-1))
            h_detr_feats = self.h_query_proj(feats[h_unique_indices])
            o_detr_feats = self.o_query_proj(feats[o_unique_indices])

           # import pdb; pdb.set_trace()
#             F.normalize(feats[h_unique_indices], dim=-1) @ F.normalize(feats[h_unique_indices], dim=-1).T
#             F.normalize(h_detr_feats, dim=-1) @ F.normalize(h_detr_feats, dim=-1).T

#             F.normalize(pairwise_spatial_reshaped, dim=-1) @ F.normalize(pairwise_spatial_reshaped, dim=-1).T

# pairwise_spatial_reshaped

            #text_2_query = self.text_2_queries(self.object_embedding)
            h_text = self.object_embedding[labels[h_unique_indices]]
            o_text = self.object_embedding[labels[o_unique_indices]]
            ho_text = self.object_embedding[labels[h_inverse_indices]] + self.object_embedding[labels[o_inverse_indices]]
           
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

            h_feats0 = torchvision.ops.roi_align(llava_features.view(1, 24, 24, 4096).permute(0, 3, 1, 2),  roi_boxes,
                                                    output_size=(7, 7),
                                                spatial_scale=24 / 336, aligned=True)
            o_feats0 = torchvision.ops.roi_align(llava_features.view(1, 24, 24, 4096).permute(0, 3, 1, 2),  roi_boxes1,
                                                    output_size=(7, 7),
                                                    spatial_scale=24 / 336, aligned=True)
            ho_feats0 = torchvision.ops.roi_align(llava_features.view(1, 24, 24, 4096).permute(0, 3, 1, 2),  roi_boxes2,
                                        output_size=(7, 7),
                                        spatial_scale=24 / 336, aligned=True) # (B, D)


            ho_feats0 = ho_feats0.flatten(2).mean(-1)
            h_feats0 = h_feats0.flatten(2).mean(-1)
            o_feats0 = o_feats0.flatten(2).mean(-1)

            bool_o_inverse = bbox_2_tokens[o_inverse_indices]  # [num_unique_o, 576]
            bool_h_inverse = bbox_2_tokens[h_inverse_indices]  # [num_unique_o, 576]
           # import pdb; pdb.set_trace()
            h_feats0 = h_feats0[h_inverse_indices].unsqueeze(0)
            o_feats0 = o_feats0[o_inverse_indices].unsqueeze(0)


            h_out  = self.cross_attend_lora_h(h_detr_feats[h_inverse_indices].unsqueeze(0), llava_features, ) 
            o_out = self.cross_attend_lora_o(o_detr_feats[o_inverse_indices].unsqueeze(0), llava_features, ) 
            ho_out = self.cross_attend_lora_o(ho_detr_feats.unsqueeze(0) + pairwise_spatial_reshaped.unsqueeze(0), llava_features, ) 
            #qimport pdb; pdb.set_trace()
            # ho_logits = self.verb_projection_ho(ho_tokens) #/ math.sqrt(4096)
            # h_logits = self.verb_projection_ho(h_tokens) #/ math.sqrt(4096)
            # o_logits = self.verb_projection_ho(o_tokens) #/ math.sqrt(4096)

            normed_h = self.final_norm(h_out.squeeze(0)) + h_feats0.squeeze(0)# - lm_head_t.mean(1, keepdim=True).T #+ self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16) # [N, 4096]+ self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16)
            normed_o = self.final_norm(o_out.squeeze(0)) + o_feats0.squeeze(0)#- lm_head_t.mealon(1, keepdim=True).T
            normed_ho = self.final_norm(ho_out.squeeze(0)) + ho_feats0# 

            # F.normalize(o_tokens.squeeze(0), dim= -1 ) @ F.normalize(o_tokens.squeeze(0), dim= -1 ).T
            # normed_h = h_feats0# - lm_head_t.mean(1, keepdim=True).T #+ self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16) # [N, 4096]+ self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16)
            # normed_o = o_feats0#- lm_head_t.mean(1, keepdim=True).T
            # normed_ho = ho_feats0# 

            lm_head_t = self.lm_head_embeddings.T.to(hidden_states.device).float().detach() 
            centered_lm_head = lm_head_t #- lm_head_t.mean(1, keepdim=True)
            h_vocab = normed_h @ centered_lm_head
            o_vocab = normed_o @ centered_lm_head
            ho_vocab = (normed_ho) @ centered_lm_head
            #diffs_h = torch.log(1 + torch.sigmoid(topk_vocab.unsqueeze(2) - logits_h.unsqueeze(1)).sum(1)) / log_v
            # diffs = torch.empty_like(logits_ho)
            # if self.num_classes == 117:
            #     #import pdb; pdb.set_trace()
            #     h_verb_scores = h_vocab[:, self.verb_token_ids] #- global_verb_scores # [N, 117, max_tokens] - global_verb_logits.unsqueeze(0)
            #     h_verb_scores = h_verb_scores.masked_fill(~self.verb_token_mask.unsqueeze(0), 0.0)

            #     logits_h = h_verb_scores.sum(dim=-1) / self.verb_token_mask.sum(dim=-1).unsqueeze(0) #+ lens_per_pair[-1] # [N, 117]


            #     #logits_h = h_verb_scores.min(dim=-1).values 
            #     o_verb_scores = o_vocab[:, self.verb_token_ids] #- global_verb_scores # [N, 117, max_tokens]
            #     o_verb_scores = o_verb_scores.masked_fill(~self.verb_token_mask.unsqueeze(0), 0.0)
            #     #logits_o = o_verb_scores.min(dim=-1).values 
            #     logits_o = o_verb_scores.sum(dim=-1) / self.verb_token_mask.sum(dim=-1).unsqueeze(0) #+ lens_per_pair[-1] # [N, 117] + lens_per_pair
                
            #     ho_verb_scores = ho_vocab[:, self.verb_token_ids] #- global_verb_scores # [N, 117, max_tokens]
            #     ho_verb_scores = ho_verb_scores.masked_fill(~self.verb_token_mask.unsqueeze(0), 0.0)
            #     #logits_o = o_verb_scores.min(dim=-1).values 
            #     logits_ho = ho_verb_scores.sum(dim=-1) / self.verb_token_mask.sum(dim=-1).unsqueeze(0) #+ lens_
            # if self.num_classes == 600:
            #     h_hoi_scores = h_vocab[:, self.hoi_token_ids]  # [N, 117, max_tokens]
            #     h_hoi_scores = h_hoi_scores.masked_fill(~self.hoi_token_mask.unsqueeze(0), 0.0)
            #     logits_h = h_hoi_scores.sum(dim=-1) / self.hoi_token_mask.sum(dim=-1).unsqueeze(0) #+ lens_per_pair[-1] # [N, 117]

            #     o_hoi_scores = o_vocab[:, self.hoi_token_ids]  # [N, 117, max_tokens]
            #     o_hoi_scores = o_hoi_scores.masked_fill(~self.hoi_token_mask.unsqueeze(0), 0.0)
            #     logits_o = o_hoi_scores.sum(dim=-1) / self.hoi_token_mask.sum(dim=-1).unsqueeze(0)
            # chunk_size = 4  # only 4 pairs × 32000 vocab in memory at a time
            # probs_dist_ho = F.softmax(ho_vocab, dim=-1)
            # entropy_ho = -(probs_dist_ho * torch.log(probs_dist_ho + 1e-10)).sum(dim=-1) 
            # probs_dist_h = F.softmax(h_vocab, dim=-1)
            # entropy_h = -(probs_dist_h * torch.log(probs_dist_h + 1e-10)).sum(dim=-1) 
            # probs_dist_o = F.softmax(o_vocab, dim=-1)
            # entropy_o = -(probs_dist_o * torch.log(probs_dist_o + 1e-10)).sum(dim=-1) 

            # w_ho = 1.0 / (1.0 + entropy_ho)'

            topk_ho_vocab = torch.topk(ho_vocab, k=1024, dim=-1).values
            topk_h_vocab = torch.topk(h_vocab, k=1024, dim=-1).values
            topk_o_vocab = torch.topk(o_vocab, k=1024, dim=-1).values
            # probs_dist_ho = F.softmax(topk_ho_vocab, dim=-1)
            # entropy_ho = -(probs_dist_ho * torch.log(probs_dist_ho + 1e-10)).sum(dim=-1) 
            # probs_dist_h = F.softmax(topk_h_vocab , dim=-1)
            # entropy_h = -(probs_dist_h * torch.log(probs_dist_h + 1e-10)).sum(dim=-1) 
            # probs_dist_o = F.softmax(topk_o_vocab , dim=-1)
            # entropy_o = -(probs_dist_o * torch.log(probs_dist_o + 1e-10)).sum(dim=-1) 

            # confidence_h = (1 - entropy_h / math.log(32000))
            # confidence_ho = (1 - entropy_ho / math.log(32000))
            # confidence_o = (1 - entropy_o / math.log(32000))
            # confidence_h = torch.exp(-entropy_h)
            # confidence_ho = torch.exp(-entropy_ho)
            # confidence_o = torch.exp(-entropy_o)

           # import pdb; pdb.set_trace()
            logits_ho = torch.sigmoid(self.verb_projection_ho(normed_ho).unsqueeze(-1) - topk_ho_vocab.unsqueeze(1)).mean(-1) #-  #- normed_ho @ self.mu.T #- global_logit
            logits_h = torch.sigmoid(self.verb_projection_ho(normed_h).unsqueeze(-1)  - topk_h_vocab.unsqueeze(1)).mean(-1) #torch.log(torch.sigmoid(self.verb_projection_ho(normed_h).unsqueeze(-1) - topk_h_vocab.unsqueeze(1)).sum(-1)) / torch.log(torch.tensor(100.0)) #-  #- normed_ho @ self.mu.T #- global_logit #- normed_h @ self.mu.T  #- global_logit
            logits_o = torch.sigmoid(self.verb_projection_ho(normed_o).unsqueeze(-1)  - topk_o_vocab.unsqueeze(1)).mean(-1) #torch.log(torch.sigmoid(self.verb_projection_ho(normed_o).unsqueeze(-1) - topk_o_vocab.unsqueeze(1)).sum(-1)) / torch.log(torch.tensor(100.0)) #-  #- normed_ho @ self.mu.T #- global_logit #- normed_o @ self.mu.T #- global_logitb.unsqueeze(1)).mean(-1) #torch.log(torch.sigmoid(self.verb_projection_ho(normed_o).unsqueeze(-1) - topk_o_vocab.unsqueeze(1)).sum(-1)) / torch.log(torch.tensor(100.0)) #-  #- normed_ho @ self.mu.T #- global_logit #- normed_o @ self.mu.T #- global_logit

            # logits_ho = 1 - diffs_ho
            # logits_o = 1 - diffs_o
            # logits_h = 1 - diffs_h #+ logits_h[h_inverse_indices] + logits_o[o_inverse_indices]) /3

            logits = (logits_h[h_inverse_indices] + logits_o[o_inverse_indices] + logits_ho)/3

            #logits = (ho_logits + h_logits[h_inverse_indices] + o_logits[o_inverse_indices])/3
            #asd = h_out @  self.lm_head_embeddings.T.to(hidden_states.device)
            #print(self.clip_head['tokenizer'].decode(torch.topk(ho_vocab[1], 100, dim=-1)[1]))
            #print(self.clip_head['tokenizer'].decode(torch.topk(asd[0][0], 10, dim=-1)[1]))
            
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            all_logits_collated.append(logits)
            import pdb; pdb.set_trace()

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
        # loss = binary_focal_loss_with_logits(
        # torch.log(
        #     prior / (1 + torch.exp(-logits) - prior) + 1e-8
        # ), labels, reduction='sum',
        # alpha=self.alpha, gamma=self.gamma
        # )
        #import pdb; pdb.set_trace()
        loss = binary_focal_loss(
            prior * logits,
            labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )
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
            #scores = torch.sigmoid(lg[x, y])
            scores = lg[x, y]
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

