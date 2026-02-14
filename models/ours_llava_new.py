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


from methods.llava_utils import retrieve_logit_lens_llava, load_llava_state, run_llava_model, compute_conditional_likelihood_llava, get_img_idx
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

def calculate_js_divergence_vectorized(logits):
    """Memory-efficient JS divergence calculation per token."""
    num_layers = logits.shape[0]
    device = logits.device
    dtype = logits.dtype
    
    js_divs = torch.zeros(num_layers - 1, logits.shape[1], device=device, dtype=dtype)
    eps = 1e-10
    
    for i in range(num_layers - 1):
        with torch.no_grad():
            pi = F.softmax(logits[i].float(), dim=-1)
            pj = F.softmax(logits[i + 1].float(), dim=-1)
            
            A = (pi + pj) / 2
            
            js_divs[i] = 0.5 * (
                (pi * (torch.log(pi + eps) - torch.log(A + eps))).sum(dim=-1) +
                (pj * (torch.log(pj + eps) - torch.log(A + eps))).sum(dim=-1)
            )
            
            del pi, pj, A
            torch.cuda.empty_cache()
    
    return js_divs  # [32, 576]


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

        # self.h_steer = verbsteer(in_dim=4096, out_dim=4096, rank=args.adapt_dim)
        # self.o_steer = verbsteer(in_dim=4096, out_dim=4096, rank=args.adapt_dim)
        # self.ho_steer = verbsteer(in_dim=4096, out_dim=4096, rank=args.adapt_dim)
        self.lm_head_embeddings = torch.load("/home/taehoon/HOICLIP/training_linearshortcut/lm_head_embedding_7b.pt", "cpu")
        self.verb_classifier_ho = torch.load("/home/taehoon/HOICLIP/training_linearshortcut/verb_classifier_weights_ho_7b.pt", "cpu").to(torch.bfloat16)
       # self.verb_classifier_ho = F.normalize(self.verb_classifier_ho, p=2, dim=1)
        self.verb_projection_ho = nn.Linear(4096, 117, bias=False).to(torch.bfloat16)
        self.verb_projection_ho.weight.data = self.verb_classifier_ho
        for param in self.verb_projection_ho.parameters():
            param.requires_grad = False

 #/  self.lm_head_embeddings.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        # Precompute token indices for each verb (117 verbs x 3 forms each)
        #asd2 = self.verb_classifier_ho[76] @ self.lm_head_embeddings.T
        # For multi-token verbs, we store all sub-token ids and take max at runtime
        if self.num_classes == 117:
            # 117 verbs × 3 forms each: (base, -ing, past)
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
        
        # self.text_2_queries = MLP(4096, 128, args.adapt_dim, 2)
        # self.ho_llava_2_queries = nn.Linear(4096, args.adapt_dim, bias = False)
        # self.h_llava_2_queries = nn.Linear(4096, args.adapt_dim, bias = False)
        # self.o_llava_2_queries = nn.Linear(4096, args.adapt_dim, bias = False)
        self.ho_query_proj = MLP(512, 128, 4096, 2).to(torch.bfloat16)
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


        #self.alpha_logit = nn.Parameter(torch.tensor(0.0)) 

    #     import nltk
    #     from nltk.corpus import wordnet as wn
    #     nltk.download('wordnet')
    #     nltk.download('omw-1.4')

    #     # 1. Get all unique verbs from WordNet
    #     all_verbs = set()
    #     for synset in wn.all_synsets(pos=wn.VERB):
    #         for lemma in synset.lemma_names():
    #             all_verbs.add(lemma.replace('_', ' '))

    #     print(f"Found {len(all_verbs)} unique verbs")  # ~6,000+


    #    # import pdb; pdb.set_trace()


    #     verb_embeddings = []
    #     verb_names = []
    #     with torch.no_grad():
    #         for verb in all_verbs:
    #             token_ids = self.clip_head['tokenizer'].encode(verb)[1:]  # skip BOS
    #             if len(token_ids) == 0:
    #                 continue
    #             emb = self.lm_head_embeddings[token_ids].float().mean(dim=0)
    #             verb_embeddings.append(emb)
    #             verb_names.append(verb)
    #     verb_matrix = torch.stack(verb_embeddings)  # [~6000, 4096]
    #     print(f"Verb matrix shape: {verb_matrix.shape}")
    #     mu = verb_matrix.mean(dim=0, keepdim=True)
    #     self.register_buffer('mu', mu.to(torch.bfloat16)) 
    #     verb_centered = verb_matrix - mu
        anchor = torch.load("/home/taehoon/HOILENS/anchor_256.pt", "cpu")
        self.register_buffer('anchor', anchor.to(torch.bfloat16)) 
        mu = torch.load("/home/taehoon/HOILENS/verb_mean.pt", "cpu")
        self.register_buffer('mu', mu.to(torch.bfloat16)) 
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

        #Clone layers for HO (human + object union)
        cloned_layers_ho = nn.ModuleList([
            copy.deepcopy(self.clip_head["model"].model.layers[i])
            for i in range(start_layer, start_layer + num_layers)
        ])
        self.cross_attend_lora_ho = CrossAttendWithLoRA(cloned_layers_ho, lora_rank=16, lora_alpha=16)

        # self.logit_scale_ho = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        #self.logit_scale_h = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.logit_scale_o = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 4096), nn.ReLU(),
        ).to(torch.bfloat16)

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
        all_llava_logits_collated = []
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
            # Valid human-object pairs

            #import pdb; pdb.set_trace()



            #import pdb; pdb.set_trace()
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)

            
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            #import pdb; pdb.set_trace()
            h_unique_indices, h_inverse_indices = torch.unique(x_keep, return_inverse=True)
            o_unique_indices, o_inverse_indices = torch.unique(y_keep, return_inverse=True)


            #human = labels[x_keep] 
            ho_detr_feats = self.ho_query_proj(torch.cat([feats[x_keep],feats[y_keep]],dim=-1).to(torch.bfloat16))
            h_detr_feats = self.h_query_proj(feats[h_unique_indices].to(torch.bfloat16))
            o_detr_feats = self.o_query_proj(feats[o_unique_indices].to(torch.bfloat16))
            #objects = targets[0]["object"]#.unique()

            pairwise_spatial = compute_spatial_encodings(
                    [boxes[x.flatten()], ], [boxes[y.flatten()], ], [(336,336), ]
            ).to(torch.bfloat16)
            pairwise_spatial = self.spatial_head(pairwise_spatial)
            pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)
            pairwise_spatial_reshaped = pairwise_spatial_reshaped[x_keep, y_keep]

            objects = labels[y_keep].unique()
            # candidate_texts = []
            # for obj_idx in objects.cpu().tolist():
            #     candidates = [
            #         ((verb_idx, object_idx), text.replace("a photo of a person ", ""))
            #         for (verb_idx, object_idx), text in hico_text_label.hico_text_label.items()
            #         if object_idx == obj_idx and verb_idx in self.seen_verb_idxs
            #     ]
            #     candidate_texts.extend(candidates)

            gt_bx_h = self.recover_boxes(targets[0]['boxes_h'], targets[0]['size'])
            gt_bx_o = self.recover_boxes(targets[0]['boxes_o'], targets[0]['size'])
            
            bbox_2_tokens = bbox_to_token((336,336),boxes, 24)
            bool_h = bbox_2_tokens[x_keep]
            bool_o = bbox_2_tokens[y_keep]
            bool_union = bool_h | bool_o


            #img_start_idx, img_end_idx, prefix_prompt_end_idx = get_img_idx(self.clip_head, self.clip_head['model_name'], self.clip_head['tokenizer'],  "Describe this image in detail")
            # llama_modify(
            #     self.clip_head['model'],
            #     self.args.start_idx,
            #     self.args.end_idx,
            #     True,
            #     0.5,
            #     False,
            #     img_start_idx,
            #     img_end_idx,
            #     bool_union[0],
            #     prefix_prompt_end_idx,
            #     None,
            #     None,
            #     self.args.focus
            # )
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


            # tokens = word_tokenize(output)
            # tagged = pos_tag(tokens)
            # verbs = [word for word, tag in tagged if tag.startswith("VB")]
            # nouns = [word for word, tag in tagged if tag.startswith("NN")]
            # #Sort by probability (descending order)
            # #import pdb; pdb.set_trace()
            # #generate_ids[0]
            # #self.clip_head['tokenizer'].decode(generated_ids[0][19])
            # #results_per_object_sorted = sorted(results_per_object, key=lambda x: x['probs'], reverse=True)
            # cumulative_img_set = set()
            # gen_ids_list = set(generated_ids[0].tolist())
            # for layer_idx in range(len(hidden_states)):
            #     llava_features = hidden_states[layer_idx]
            #     img_tokens = llava_features @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
            #     img_decoded_ids = img_tokens.topk(5, dim=-1).indices
            #     layer_set = set(img_decoded_ids.reshape(-1).tolist())
            #     cumulative_img_set |= layer_set 
                
            #     content_overlap = 0
            #     content_total = 0
            #     content_overlap_tokens = []
            #     for tid in gen_ids_list:
            #         tok = self.clip_head['tokenizer'].decode([tid], skip_special_tokens=True).strip().lower()
            #         # if len(tok) <= 2 or tok in {'the', 'a', 'an', 'is', 'are', 'was', 'in', 'on', 'of', 'and', 'to', 'with', 'for'}:
            #         #     continue
            #         content_total += 1
            #         if tid in layer_set:
            #             content_overlap += 1
            #             content_overlap_tokens.append(self.clip_head['tokenizer'].decode([tid]))
            #         # if tid in cumulative_img_set:
            #         #     content_overlap += 1
            #         #     content_overlap_tokens.append(self.clip_head['tokenizer'].decode([tid]))
                
            #     recall = content_overlap / content_total if content_total > 0 else 0
            #     print(f"layer {layer_idx:2d} | recall: {recall:.3f} ({content_overlap}/{content_total}) | {content_overlap_tokens}")



            #import pdb; pdb.set_trace()
            # img_set = set(img_decoded_ids[0].tolist())
            # gen_set = set(generated_ids[0].tolist())
            # overlap = img_set & gen_set
            # recall = len(overlap) / len(gen_set) if len(gen_set) > 0 else 0  # fraction of generated tokens found in image tokens
            # precision = len(overlap) / len(img_set) if len(img_set) > 0 else 0
            # print(self.clip_head['tokenizer'].decode(torch.topk(img_tokens[0], 10, dim=-1)[1]))
            # ho_detr_feats = self.ho_query_proj(torch.cat([feats[x_keep],feats[y_keep]],dim=-1))
            # h_detr_feats = self.h_query_proj(feats[h_unique_indices])
            # o_detr_feats = self.o_query_proj(feats[o_unique_indices])
           
            # text_2_query = self.text_2_queries(self.object_embedding.float())
            # #ing_dir = self.text_2_queries(self.ing.to(device))
            # h_text = text_2_query[labels[h_unique_indices]] #+ ing_dir.unsqueeze(0)
            # o_text = text_2_query[labels[o_unique_indices]] #+ ing_dir.unsqueeze(0)
            # #ho_text = h_text[h_inverse_indices] + o_text[o_inverse_indices]
            # ho_text = self.ho_text_query_proj(torch.cat([h_text[h_inverse_indices],o_text[o_inverse_indices]],dim=-1)) #+ ing_dir.unsqueeze(0)
            # bbox_2_tokens = bbox_to_token((336,336),boxes, 24)


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


            # roi_align requires float32
            #llava_features = hidden_states[self.args.layer]
          

            llava_features = hidden_states[self.args.layer]
            #llava_features = llava_features - llava_features.mean(1)


            #llava_features = hidden_states[self.args.layer]
            #asd = hidden_states[-1] @ centered_lm_head

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


            # llava_features_final = hidden_states[-1]
            #F.normalize(ho_feats0, dim=-1).squeeze(0) @  F.normalize(ho_feats0, dim=-1).squeeze(0).T
            # #llava_features = hidden_states[self.args.layer]

            # llava_feat_for_roi = llava_features_final.float().view(1, 24, 24, 4096).permute(0, 3, 1, 2)
            # h_feats_final = torchvision.ops.roi_align(llava_feat_for_roi, roi_boxes,
            #                                         output_size=(7, 7),
            #                                         spatial_scale=24 / 336, aligned=True)
            # o_feats_final = torchvision.ops.roi_align(llava_feat_for_roi, roi_boxes1,
            #                                         output_size=(7, 7),
            #                                         spatial_scale=24 / 336, aligned=True)

            # ho_feats_final = torchvision.ops.roi_align(llava_feat_for_roi, roi_boxes2,
            #                                         output_size=(7, 7),
            #                                         spatial_scale=24 / 336, aligned=True)

            # # Convert back to original dtype
            # ho_feats_final = ho_feats_final.flatten(2).mean(-1).unsqueeze(0).to(llava_features.dtype)
            # h_feats_final  = h_feats_final.flatten(2).mean(-1).unsqueeze(0).to(llava_features.dtype)
            # o_feats_final = o_feats_final.flatten(2).mean(-1).unsqueeze(0).to(llava_features.dtype)
     


            # llava_feat_for_roi = hidden_states.float().view(-1, 24, 24, 4096).permute(0, 3, 1, 2)
            # roi_boxes_exp = expand_boxes_for_layers(roi_boxes, num_layers)
            # roi_boxes1_exp = expand_boxes_for_layers(roi_boxes1, num_layers)
            # roi_boxes2_exp = expand_boxes_for_layers(roi_boxes2, num_layers)

            # # Single roi_align call for all layers
            # h_feats_all = torchvision.ops.roi_align(llava_feat_for_roi, roi_boxes_exp,
            #                                         output_size=(7, 7),
            #                                         spatial_scale=24 / 336, aligned=True)
            # o_feats_all = torchvision.ops.roi_align(llava_feat_for_roi, roi_boxes1_exp,
            #                                         output_size=(7, 7),
            #                                         spatial_scale=24 / 336, aligned=True)
            # ho_feats_all = torchvision.ops.roi_align(llava_feat_for_roi, roi_boxes2_exp,
            #                                         output_size=(7, 7),
            #                                         spatial_scale=24 / 336, aligned=True)

            # # Reshape: (num_layers * num_boxes, 4096, 7, 7) -> (num_layers, num_boxes, 4096)
            # h_feats_stacked = h_feats_all.flatten(2).sum(-1).view(num_layers, roi_boxes.shape[0], -1)
            # o_feats_stacked = o_feats_all.flatten(2).sum(-1).view(num_layers, roi_boxes1.shape[0], -1)
            # ho_feats_stacked = ho_feats_all.flatten(2).sum(-1).view(num_layers, roi_boxes2.shape[0] , -1)

            # # Convert dtype
            # original_dtype = hidden_states[0].dtype
            # h_feats_stacked = h_feats_stacked.to(original_dtype)
            # o_feats_stacked = o_feats_stacked.to(original_dtype)
            # ho_feats_stacked = ho_feats_stacked.to(original_dtype)

            # logits2 = (hidden_states.squeeze(1)) @ self.verb_classifier_ho.T.to(hidden_states.device).to(torch.bfloat16)
            # logits1 = (hidden_states.squeeze(1)) @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
            #aggr_logits = hidden_states.sum(dim=0) @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
            #asd = logits[:,bool_h[0]].amax(dim=(0, 1))
            #delta = h_out 
            #logits2 = h_feats0 @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
            # js_divs = calculate_js_divergence_vectorized(logits1)
            # torch.save(js_divs, 'js_divergence.pt')
            # print(js_divs)  # Shape: [32]
            # import pdb; pdb.set_trace()
            # asd = F.softmax(logits1, dim =1)[32]
            # entropy = -(asd * asd.log()).sum(dim=1)
        #     norms = torch.norm(hidden_states, dim=-1).squeeze(1)

        #     mean = norms[0].mean()

            # for i in range(asd3.shape[0]):
            #     print(i)
            #     print(self.clip_head['tokenizer'].decode(torch.topk(asd3[i], 20, dim=-1)[1]))
            # #asd = logits[22][bool_union[3]].mean(0)
            # import pdb; pdb.set_trace()
            # Cross-attend ho_feats to llava_features
            #import pdb; pdb.set_trace()
            #comb = ho_feats0.half() + self.object_embedding[labels[o_inverse_indices]].half() + self.object_embedding[labels[h_inverse_indices]].half()
            #asd = self.verb_classifier_ho.squeeze(0).float().to(hidden_states.device) @ self.lm_head_embeddings.T.to(hidden_states.device).float()
            # ho_out = self.cross_attend(ho_feats0.half(), llava_features.half(), roi_mask=bool_o)  # [1, N, 4096]
            # #ho_out = self.cross_attend(self.verb_classifier_ho.half().unsqueeze(0).to(hidden_states.device), llava_features.half())  # [1, N, 4096]
            # ho_out = ho_out.squeeze(0)  # [N, 4096]
            # logits = ho_out.float() @ self.lm_head_embeddings.T.to(hidden_states.device).float()
            # logits1 = ho_feats0.squeeze(0).float() @ self.lm_head_embeddings.T.to(hidden_states.device).float()
            # for i in range(ho_out.shape[0]):
            #     #print(self.clip_head['tokenizer'].decode(torch.topk(asd[86], 10, dim=-1)[1]))
            #     print(self.clip_head['tokenizer'].decode(torch.topk(logits[i], 10, dim=-1)[1]))
            # print("orig")
            # for i in range(ho_out.shape[0]):
            #     print(self.clip_head['tokenizer'].decode(torch.topk(logits1[i], 10, dim=-1)[1]))
            # #self.clip_head['tokenizer'].decode(topk_ids)
            # import pdb; pdb.set_trace()
            # ho_feats0 = ho_feats_stacked[self.args.layer].unsqueeze(0)
            # o_feats0 = o_feats_stacked[self.args.layer].unsqueeze(0)
            # h_feats0 = h_feats_stacked[self.args.layer].unsqueeze(0)

            #llava_features = hidden_states[self.args.layer]
            #asd = ho_feats_stacked[:,1].unsqueeze(1) - hidden_states.squeeze(1).mean(1, keepdim=True)

            # Cross-attend for H, O, and HO
            #h_out @ 
            # h_feats0/o_feats0 use unique indices, so use corresponding masks
            # bool_h_unique = bbox_2_tokens[h_unique_indices]  # [num_unique_h, 576]
            # bool_o_unique = bbox_2_tokens[o_unique_indices]  # [num_unique_o, 576]
            bool_o_inverse = bbox_2_tokens[o_inverse_indices]  # [num_unique_o, 576]
            bool_h_inverse = bbox_2_tokens[h_inverse_indices]  # [num_unique_o, 576]
           # import pdb; pdb.set_trace()
            h_feats0 = h_feats0.squeeze(0)[h_inverse_indices].unsqueeze(0)
            o_feats0 = o_feats0.squeeze(0)[o_inverse_indices].unsqueeze(0)
            #h_out = self.cross_attend_lora_h(h_feats0 + self.object_embedding[labels[h_unique_indices]].to(torch.bfloat16), llava_features, roi_mask=bool_h_unique)
            #o_out = self.cross_attend_lora_o(o_feats0 + self.object_embedding[labels[o_unique_indices]].to(torch.bfloat16), llava_features, roi_mask=bool_o_unique)
            #ho_out = self.cross_attend_lora_ho(ho_feats0 + self.object_embedding[labels[o_inverse_indices]].to(torch.bfloat16) + self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16) , llava_features, roi_mask=bool_union)

            #h_out, attn_weights_h = self.cross_attend_lora_h(h_feats0 + self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16), llava_features, roi_mask=bool_o_inverse, return_attn_weights=True)
            #o_out, attn_weights_o = self.cross_attend_lora_o(o_feats0 + self.object_embedding[labels[o_inverse_indices]].to(torch.bfloat16), llava_features, roi_mask=bool_h_inverse, return_attn_weights=True)
            
            h_out  = self.cross_attend_lora_h(h_feats0 + h_detr_feats[h_inverse_indices], llava_features, ) 
            o_out = self.cross_attend_lora_o(o_feats0 + o_detr_feats[o_inverse_indices], llava_features, ) 
            ho_out = self.cross_attend_lora_o(ho_feats0 + ho_detr_feats + pairwise_spatial_reshaped, llava_features, ) 

            # h_out  = self.cross_attend_lora_h(h_feats0 + h_detr_feats[h_inverse_indices], llava_features) 
            # o_out = self.cross_attend_lora_o(o_feats0 + o_detr_feats[o_inverse_indices], llava_features) 
            # ho_out = self.cross_attend_lora_o(ho_feats0 + ho_detr_feats, llava_features) 
            # h_out  = self.cross_attend_lora_h(h_feats0 , all_hidden_states[-2], roi_mask=bool_o_inverse) 
            # o_out = self.cross_attend_lora_o(o_feats0 , all_hidden_states[-2], roi_mask=bool_h_inverse) 
            #import pdb; pdb.set_trace()
            #h_out = self.cross_attend_lora_h(h_feats0 , llava_features, roi_mask=bool_o_inverse) #+ h_feats0
            # o_out = self.cross_attend_lora_o(o_feats0 , llava_features, roi_mask=bool_o_unique) #+ o_feats0
            #ho_out = self.cross_attend_lora_ho(ho_feats0 , llava_features, roi_mask=bool_union) #+ ho_feats0
            # h_feats1 = self.h_llava_2_queries(h_feats0.squeeze(0))
            # o_feats1 = self.o_llava_2_queries(o_feats0.squeeze(0))
            # ho_feats1 = self.ho_llava_2_queries(ho_feats0.squeeze(0))
            # ho_out1 = self.final_norm(ho_out)
            # ho_out2 = self.final_norm(ho_feats0)
            #logits_base_123 = ho_feats0 @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)

            # ho_feats_final = ho_feats_final.flatten(2).mean(-1).unsqueeze(0).to(llava_features.dtype)
            # h_feats_final  = h_feats_final.flatten(2).mean(-1).unsqueeze(0).to(llava_features.dtype)
            # o_feats_final = o_feats_final.flatten(2).mean(-1).unsqueeze(0).to(llava_features.dtype)


            #logits_reg = ho_feats_final @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
            normed_h = h_out.squeeze(0)# - lm_head_t.mean(1, keepdim=True).T #+ self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16) # [N, 4096]+ self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16)
            normed_o = o_out.squeeze(0) #- lm_head_t.mean(1, keepdim=True).T
            normed_ho = ho_out.squeeze(0)# - lm_head_t.mean(1, keepdim=True).T  #+ self.object_embedding[labels[o_inverse_indices]].to(torch.bfloat16) # [N, 4096]

            #normed_ho= rms_normalize(normed_ho)
            #normed_h= rms_normalize(normed_h)
            #normed_o= rms_normalize(normed_o)
            #F.normalize(normed_ho, dim =-1) @  F.normalize(self.verb_classifier_ho, dim =-1).T.to(hidden_states.device)
            #lm_head_t = self.lm_head_rms_normed.T.to(device).to(torch.bfloat16)
            lm_head_t = self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16).detach() 
            centered_lm_head = lm_head_t #- lm_head_t.mean(1, keepdim=True)
            # Logit lens across all layers: [33, 1, 576, 32000] -> [33, 576, 32000]
            # all_layer_logits = (h_feats0 @ lm_head_t).squeeze(1)


            
            # # Per-verb scores across all layers: [33, 576, 117, max_tokens]
            # all_layer_logits = (hidden_states @ lm_head_t).squeeze(1)
            # lens_verb = all_layer_logits[:, :, self.verb_token_ids]
            # lens_verb = lens_verb.masked_fill(~self.verb_token_mask, 0.0)
            # lens_verb_scores = lens_verb.sum(dim=-1) / self.verb_token_mask.sum(dim=-1)  # [33, 576, 117]

            # # Pool verb scores per union region per pair
            # # bool_union: [N, 576], lens_verb_scores: [33, 576, 117]
            # #bu = bool_union.float()

            #h_feats0 @ lm_head_t

            # bu = bool_union.to(torch.bfloat16)
            # bu_count = bu.sum(1, keepdim=True).clamp(min=1)  # [N, 1]
            # # [N, 576] @ [33, 576, 117] -> einsum -> [33, N, 117]
            # lens_per_pair = torch.einsum('np,lpv->lnv', bu, lens_verb_scores) / bu_count.unsqueeze(0)  # [33, N, 117]

            # h_vocab = h_feats0.squeeze(0) @ lm_head_t

            # #all_out = hidden_states[-1] @ lm_head_t
            # o_vocab = o_feats0.squeeze(0) @ lm_head_t

            # h_vocab = self.final_norm(h_feats0.squeeze(0)) @ lm_head_t
            # o_vocab = self.final_norm(o_feats0.squeeze(0)) @ lm_head_t

            #llava_features @ 
            #o_vocab[:,generated_ids[0]].sum(-1) / len(generated_ids[0])
            #asd1 = hidden_states[-1][0].mean(0) @  lm_head_t
            #log_ref = F.logsigmoid(asd1)
            # global_verb_logits = asd1[self.verb_token_ids]lm_head_t.mean(1, keepdim=True)
            # global_verb_scores = global_verb_logits.unsqueeze(0)
            # global_verb_scores = global_verb_logits.masked_fill(~self.verb_token_mask.unsqueeze(0), 0.0)
            # global_logit = global_verb_scores.sum(dim=-1) / self.verb_token_mask.sum(dim=-1).unsqueeze(0)
            #obj_vocab = self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16) @ lm_head_t # [N, 4096]+ self.object_embedding[labels[h_inverse_indices]].to(torch.bfloat16)
            h_vocab = normed_h @ centered_lm_head
            #h_reconstructed = sparse_probs @ centered_lm_head.T
            #h_reconstructed = h_probs @ centered_lm_head.T 

            o_vocab = normed_o @ centered_lm_head

            #o_probs = sparsemax(o_vocab, dim=-1) 

            ho_vocab = normed_ho @ centered_lm_head
            #ho_probs = sparsemax(ho_vocab, dim=-1) 
            #ho_reconstructed = ho_probs @ centered_lm_head.T 
            


            #h_vocab1 = h_feats0 @ centered_lm_head

            #normed_ho @  lm_head_t
            #h_vocab1 = (normed_ho - lm_head_t.mean(1, keepdim=True).T) @ centered_lm_head
            # global_logit = self.verb_projection_ho(hidden_states[-1][0].mean(0))
            topk_ho_vocab = torch.topk(ho_vocab, k=200, dim=-1).values
            topk_h_vocab = torch.topk(h_vocab, k=200, dim=-1).values
            topk_o_vocab = torch.topk(o_vocab, k=200, dim=-1).values
            # logits_ho = torch.sigmoid(self.verb_projection_ho(normed_ho).unsqueeze(-1) - topk_ho_vocab.unsqueeze(1)).mean(-1) #-  #- normed_ho @ self.mu.T #- global_logit
            # logits_h = torch.sigmoid(self.verb_projection_ho(normed_h).unsqueeze(-1) - topk_h_vocab.unsqueeze(1)).mean(-1) #torch.log(torch.sigmoid(self.verb_projection_ho(normed_h).unsqueeze(-1) - topk_h_vocab.unsqueeze(1)).sum(-1)) / torch.log(torch.tensor(100.0)) #-  #- normed_ho @ self.mu.T #- global_logit #- normed_h @ self.mu.T  #- global_logit
            # logits_o = torch.sigmoid(self.verb_projection_ho(normed_o).unsqueeze(-1) - topk_o_vocab.unsqueeze(1)).mean(-1) #torch.log(torch.sigmoid(self.verb_projection_ho(normed_o).unsqueeze(-1) - topk_o_vocab.unsqueeze(1)).sum(-1)) / torch.log(torch.tensor(100.0)) #-  #- normed_ho @ self.mu.T #- global_logit #- normed_o @ self.mu.T #- global_logit
            #logits_ho = self.verb_projection_ho(ho_reconstructed) #- normed_ho @ self.mu.T #- global_logit
            #logits_h = self.verb_projection_ho(h_reconstructed)#- normed_h @ self.mu.T  #- global_logit
            #logits_o = self.verb_projection_ho(o_reconstructed)# - normed_o @ self.mu.T #- global_logit
            
            #.unsqueeze(-1) - global_verb_logits.unsqueeze(0)what if 
            #import pdb; pdb.set_trace()

            #vocab_mean = h_vocab.mean(dim=-1, keepdim=True)
            #vocab_std = h_vocab.std(dim=-1, keepdim=True)

            # Standardize the entire vocab space, then slice the verbs
            #h_vocab = (h_vocab - vocab_mean) / (vocab_std + 1e-7)

            #vocab_mean = o_vocab.mean(dim=-1, keepdim=True)
            #vocab_std = o_vocab.std(dim=-1, keepdim=True)

            #o_vocab = (o_vocab - vocab_mean) / (vocab_std + 1e-7)
            ##o_vocab1 = o_feats0 @ lm_head_t
            # Lookup verb token scores and take max across sub-tokens
            # verb_token_ids: [117, max_tokens], verb_token_mask: [117, max_tokens]
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

            probs_ho = torch.sigmoid(self.verb_projection_ho(normed_ho).unsqueeze(-1) - topk_ho_vocab.unsqueeze(1)).mean(-1) #-  #- normed_ho @ self.mu.T #- global_logit
            probs_h = torch.sigmoid(self.verb_projection_ho(normed_h).unsqueeze(-1) - topk_h_vocab.unsqueeze(1)).mean(-1) #torch.log(torch.sigmoid(self.verb_projection_ho(normed_h).unsqueeze(-1) - topk_h_vocab.unsqueeze(1)).sum(-1)) / torch.log(torch.tensor(100.0)) #-  #- normed_ho @ self.mu.T #- global_logit #- normed_h @ self.mu.T  #- global_logit
            probs_o = torch.sigmoid(self.verb_projection_ho(normed_o).unsqueeze(-1) - topk_o_vocab.unsqueeze(1)).mean(-1) #torch.log(torch.sigmoid(self.verb_projection_ho(normed_o).unsqueeze(-1) - topk_o_vocab.unsqueeze(1)).sum(-1)) / torch.log(torch.tensor(100.0)) #-  #- normed_ho @ self.mu.T #- global_logit #- normed_o @ self.mu.T #- global_logit

            # Normalize against top-k vocab
            # topk_vocab_ho, _ = ho_vocab.detach().topk(1024, dim=-1)
            # topk_vocab_h, _ = h_vocab.detach().topk(1024, dim=-1)

            # topk_vocab_o, _ = o_vocab.detach().topk(1024, dim=-1)
            #normed_o @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16).detach() 
            probs_dist_ho = F.softmax(ho_vocab, dim=-1)
            entropy_ho = -(probs_dist_ho * torch.log(probs_dist_ho + 1e-10)).sum(dim=-1) 
            # probs_dist_h = F.softmax(h_vocab, dim=-1)
            # entropy_h = -(probs_dist_h * torch.log(probs_dist_h + 1e-10)).sum(dim=-1) 
            # probs_dist_o = F.softmax(o_vocab, dim=-1)
            # entropy_o = -(probs_dist_o * torch.log(probs_dist_o + 1e-10)).sum(dim=-1) 

            # # w_ho = 1.0 / (1.0 + entropy_ho)'
            # confidence_h = (1 - entropy_h / math.log(32000))
            # confidence_ho = (1 - entropy_ho / math.log(32000))
            # confidence_o = (1 - entropy_o / math.log(32000))
            # # w_h  = 1.0 / (1.0 + entropy_h)
            # #import pdb; pdb.set_trace()
            # w_o  = 1.0 / (1.0 + entropy_o)
            #logits = ((1 + confidence_ho.unsqueeze(1))* logits_ho + (1 + confidence_h.unsqueeze(1)) * logits_h + (1 + confidence_o.unsqueeze(1)) *logits_o)/3# / #100 # / (w_ho + w_h + w_o)
            logits = (probs_ho + probs_h + probs_o) /3
            # log_v = torch.log(torch.tensor(1024, device=logits_h.device))
            # diffs_h = torch.log(1 + torch.sigmoid(topk_vocab_h.unsqueeze(2) - logits_h.unsqueeze(1)).sum(1)) / log_v
            # diffs_o = torch.log(1 + torch.sigmoid(topk_vocab_o.unsqueeze(2) - logits_o.unsqueeze(1)).sum(1)) / log_v
            # diffs_ho = torch.log(1 + torch.sigmoid(topk_vocab_ho.unsqueeze(2) - logits_ho.unsqueeze(1)).sum(1)) / log_v
            #probs_h = logits_h/h_vocab.max(dim=-1, keepdim=True)[0]
            #probs_o = logits_o/o_vocab.max(dim=-1, keepdim=True)[0]
            #confidence = 1.0 / (1.0 + entropy)
            # probs_ho = 1 - diffs_ho
            # probs_h = 1 - diffs_h
            # probs_o = 1 - diffs_o
            #logits =  (probs_ho + probs_h + probs_o)/3
            #logits = (probs_h + probs_o)/2
            #lens_per_pair[-1]logits.shape
            #logits = logits_h
            #asd1 =  logits[20][bool_h[0]]
            #asd = ho_feats0 - llava_features.mean(1)
        # import pdb; pdb.set_trace()∂
        #     verb_indices = torch.tensor([r['candidates'][0][0] for r in results_per_object_sorted], device=device, dtype=torch.long)
        #     obj_indices = torch.tensor([r['candidates'][0][1] for r in results_per_object_sorted], device=device, dtype=torch.long)
        #     probs = torch.tensor([float(r['probs'][0]) for r in results_per_object_sorted], device=device, dtype=torch.float32)

        #     # Get object labels for each pair
        #     obj_labels = labels[y_keep]  # Shape: [N]

        #     # Create target probability distribution from LLaVA for each pair
        #     N = logits.shape[0]
        #     num_verbs = 117
        #     llava_target = torch.zeros((N, num_verbs), device=device, dtype=torch.float32)
        #    #import pdb; pdb.set_trace()
        #     match_mask = (obj_labels.unsqueeze(1) == obj_indices.unsqueeze(0))  # [N, M]

        #     # Get pair indices and result indices where they match
        #     pair_idx, result_idx = torch.where(match_mask)
            #logits1 = (ho_feats0.float() @ self.lm_head_embeddings.T.to(hidden_states.device).float())[0]
        #     # Assign probabilities
        #     bool_union[1].nonzero()
        # #     llava_target[pair_idx, verb_indices[result_idx]] = probs[result_idx]self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
        #     logits = asd @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
        #     logits = hidden_states.squeeze(1) @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
        #     logits1 = h_feats_stacked @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
            # for i in range(ho_out.shape[1]):
            #     #print(self.clip_head['tokenizer'].decode(torch.topk(asd[86], 10, dim=-1)[1]))
            #     print(self.clip_head['tokenizer'].decode(torch.topk(logits1[2], 10, dim=-1)[1]))
            # logits2 = o_out.squeeze(0) @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
            # print("object")
            # for i in range(o_out.shape[1]):
            #     #print(self.clip_head['tokenizer'].decode(torch.topk(asd1, 50, dim=-1)[1]))
            #     print(self.clip_head['tokenizer'].decode(torch.topk(logits1[25][3], 10, dim=-1)[1]))
                 # print(self.clip_head['tokenizer'].decode(torch.topk(logits[25][0], 10, dim=-1)[1]))
            # logits3 = h_out.squeeze(0) @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
            # print("object")
            #for i in range(h_out.shape[1]):
            #     #print(self.clip_head['tokenizer'].decode(torch.topk(ho_vocab[7], 200, dim=-1)[1]))
            #print(self.clip_head['tokenizer'].decode(torch.topk(asd2, 20, dim=-1)[1]))
            #print(self.clip_head['tokenizer'].decode(torch.topk(o_vocab[5], 30, dim=-1)[1]))
            #print(self.clip_head['tokenizer'].decode(torch.topk(all_out[0][468], 10, dim=-1)[1]))
            #     print(self.clip_head['tokenizer'].decode(torch.topk(logits[i], 10, dim=-1)[1]))
            #import pdb; pdb.set_trace()
            #asd = llava_features[0] @ lm_head_t
            #print(self.clip_head['tokenizer'].decode(torch.topk(asd[469], 10, dim=-1)[1]))

            #asd  =  self.final_norm(hidden_states[-1].mean(1)) @ self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)  # [chunk_size, 32000]
            #asd1 = hidden_states[-1][0].mean(0) @  self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
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
            # print(self.clip_head['tokenizer'].decode(torch.topk(h_vocab1[0][5], 200, dim=-1)[1]))
            #bbox_2_tokens[o_inverse_indices][1].nonzero()
            import pdb; pdb.set_trace()

        return all_logits_collated, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated, #all_llava_logits_collated

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

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets, llava_logits): ### loss
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

        # unseen_mask = torch.zeros(self.num_classes, dtype=torch.bool, device=logits.device)
        # unseen_mask[self.filtered_hoi_idx] = True
        # unseen = unseen_mask[y]
        # # labels[unseen] = 0
        # # valid_mask = llava_logits > 0
        # # n_valid = valid_mask.sum()
        # import pdb; pdb.set_trace()
        n_p = len(torch.nonzero(labels))
        # if dist.is_initialized():
        #     n_valid_tensor = torch.as_tensor([n_valid], device='cuda')
        #     dist.barrier()
        #     dist.all_reduce(n_valid_tensor)
        #     n_valid_global = n_valid_tensor.item()

        # if n_valid_global > 0:
        #     # Only compute BCE on valid (non-zero) entries
        #     kl_loss = F.binary_cross_entropy(
        #         model_probs[valid_mask], 
        #         llava_logits[valid_mask], 
        #         reduction='sum'
        #     ) / n_valid_global

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
        loss = binary_focal_loss(
            prior * logits,
            labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )
        #import pdb; pdb.set_trace()

        #kl_weight = 1.0
        #import pdb; pdb.set_trace()
        return (loss / n_p) #+ kl_weight * kl_loss

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
        logits, prior, bh, bo, objects = self.compute_sim_scores(region_props,images_clip,targets, None )
        boxes = [r['boxes'] for r in region_props]
        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets, None)

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

