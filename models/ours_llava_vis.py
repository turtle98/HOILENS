"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>
The Australian National University
Australian Centre for Robotic Vision
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

import numpy as np
import torchvision 
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
#import wandb

from utils.hico_list import hico_verbs_sentence, hico_verb_object_list, hico_verbs, hico_objects
from utils.vcoco_list import vcoco_verbs_sentence
from utils.hico_utils import reserve_indices, HOI_IDX_TO_ACT_IDX
from utils.postprocessor import PostProcess
from utils.ops import binary_focal_loss_with_logits, binary_focal_loss, vectorized_bboxes_and_indices, bbox_to_token
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

# Clone a LLaMA layer directly

# Usage:
# llama_layer = model["model"].model.layers[15]  # pick a layer
# decoder = create_decoder_from_llama_layer(llama_layer, num_heads=32)


from methods.llava_utils import retrieve_logit_lens_llava, load_llava_state, run_llava_model, compute_conditional_likelihood_llava, get_img_idx
from methods.attention import llama_modify
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import MaxAbsScaler
from matplotlib.colors import to_hex
from mpl_toolkits.mplot3d import Axes3D

def farthest_point_sampling(embeddings, k=256):
    """Select k diverse anchors from LM head embeddings via FPS."""
    N, D = embeddings.shape
    embeddings = F.normalize(embeddings, dim=-1)  # cosine distance
    
    selected = [torch.randint(N, (1,)).item()]
    min_dists = torch.full((N,), float('inf'), device=embeddings.device)
    
    for _ in range(k - 1):
        last = embeddings[selected[-1]]
        dists = 1 - (embeddings @ last)  # cosine distance
        min_dists = torch.min(min_dists, dists)
        selected.append(min_dists.argmax().item())
    
    return torch.tensor(selected)


def data_processing(data_array):
    """
    Process the data for t-SNE.

    Args:
        data_array: Data array to be processed.
        
    Returns:
        tuple: A tuple containing the processed data and labels.
    """

    # Flatten the data for t-SNE. Here the feature dimension is assumed to be 4096.
    data_flat = data_array.reshape(-1, 4096)

    # Normalize data using MaxAbsScaler.
    scaler = MaxAbsScaler()
    data_flat_scaled = scaler.fit_transform(data_flat)

    # Create flat labels for plotting.
    # Example: first 2 points are label 0, next 16 are label 1, then 13 for label 2, and 1 for label 3.
    base_labels = np.concatenate([
        np.repeat(0, 2),
        np.repeat(1, 16),
        np.repeat(2, 13),
        np.repeat(3, 1),
    ])
    labels_flat = np.tile(base_labels, 117)

    return data_flat_scaled, labels_flat

def tsne_plot(data_flat_scaled, labels_flat):
    """
    Create a t-SNE plot.

    Args:
        data_flat_scaled: Scaled data for t-SNE.
        labels_flat: Labels for the data points.
    """
    # Create and fit a t-SNE model.
    tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, n_iter=2000, random_state=0)
    data_tsne = tsne.fit_transform(data_flat_scaled)

    # Define label names and a list of colors.
    label_names = [
        'the first two layers',
        '3rd layer to 18th layer',
        '19th layer to 31st layer',
        '32nd layer'
    ]
    # colors = generate_colors('Blues_r', len(np.unique(labels_flat)))
    colors = ['#FF6347', '#3CB371', '#4682B4', '#9370DB', '#FFA07A', '#FFDAB9', '#87CEFA', '#FFB6C1']

    # Create the t-SNE scatter plot.
    plt.figure(figsize=(12, 10))
    for i in range(len(np.unique(labels_flat))):
        plt.scatter(
            data_tsne[labels_flat == i, 0],
            data_tsne[labels_flat == i, 1],
            color=colors[i],
            label=label_names[i],
            s=25,
            alpha=0.7
        )

    # Remove axis ticks and labels.
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        left=False,
        right=False,
        labelleft=False
    )

    # Customize the plot spines (borders)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
        spine.set_linewidth(0.5)

    # Add legend and grid.
    plt.legend(loc='upper left', prop={'size': 22, 'weight': 'bold'})
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0.1)
    plt.savefig('./tsne.png', dpi=600, bbox_inches='tight')
    # Show the plot.
    #plt.show()

def tsne_plot_verb_vs_lmhead(embeddings, lm_head_embeddings, n_vocab_sample=32000, save_path='./tsne_finalnormed_vs_lmhead.png', label='embeddings'):
    """
    t-SNE plot of embeddings (NxD) and a random subset of lm_head_embeddings (32000xD)
    on the same figure to see if they are separable.
    """
    verb_np = embeddings.float().cpu().numpy()  # (N, 4096)
    lm_np = lm_head_embeddings.float().cpu().numpy()  # (32000, 4096)

    # Subsample lm_head to keep t-SNE tractable
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(lm_np.shape[0], size=min(n_vocab_sample, lm_np.shape[0]), replace=False)
    lm_sampled = lm_np[sample_idx]

    combined = np.concatenate([verb_np, lm_sampled], axis=0)

    scaler = MaxAbsScaler()
    combined_scaled = scaler.fit_transform(combined)

    perp = min(40, combined_scaled.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perp, learning_rate=200, n_iter=2000, random_state=0)
    embedded = tsne.fit_transform(combined_scaled)

    n_verb = verb_np.shape[0]

    plt.figure(figsize=(12, 10))
    plt.scatter(embedded[n_verb:, 0], embedded[n_verb:, 1],
                color='#4682B4', label=f'lm_head embeddings (n={lm_sampled.shape[0]})',
                s=15, alpha=0.4)
    plt.scatter(embedded[:n_verb, 0], embedded[:n_verb, 1],
                color='#FF6347', label=f'{label} (n={n_verb})',
                s=40, alpha=0.9, edgecolors='black', linewidths=0.3)

    plt.tick_params(axis='both', which='both',
                    bottom=False, top=False, labelbottom=False,
                    left=False, right=False, labelleft=False)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
        spine.set_linewidth(0.5)

    plt.legend(loc='upper left', prop={'size': 18, 'weight': 'bold'})
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0.1)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")

def svd_analysis(lm_head_embeddings, verb_token_ids=None, verb_token_mask=None,
                  n_components=128, save_path='./svd_analysis.png'):
    """
    SVD analysis of LM head embeddings and optionally the verb subspace.

    Returns:
        full_basis: PCA basis of full vocab [n_components, 4096]
        verb_basis: PCA basis of verb tokens [n_components, 4096] (if verb_token_ids provided)
    """
    embeds = lm_head_embeddings.float()

    # --- Full vocab SVD ---
    mean_full = embeds.mean(0)
    U_full, S_full, Vh_full = torch.linalg.svd(embeds - mean_full, full_matrices=False)

    cumvar_full = (S_full ** 2).cumsum(0) / (S_full ** 2).sum()
    rank_90 = (cumvar_full < 0.90).sum().item() + 1
    rank_95 = (cumvar_full < 0.95).sum().item() + 1
    rank_99 = (cumvar_full < 0.99).sum().item() + 1

    print(f"=== Full Vocab SVD ({embeds.shape[0]} tokens x {embeds.shape[1]}D) ===")
    print(f"  90% variance: {rank_90} components")
    print(f"  95% variance: {rank_95} components")
    print(f"  99% variance: {rank_99} components")
    print(f"  Top {n_components} explain: {cumvar_full[n_components-1]:.1%}")

    verb_basis = None
    mean_verb = None
    S_verb = None
    cumvar_verb = None

    if verb_token_ids is not None and verb_token_mask is not None:
        # --- Verb subspace SVD ---
        verb_ids = verb_token_ids[verb_token_mask].unique().cpu()
        verb_embeds = embeds[verb_ids]
        mean_verb = verb_embeds.mean(0)
        U_verb, S_verb, Vh_verb = torch.linalg.svd(verb_embeds - mean_verb, full_matrices=False)

        cumvar_verb = (S_verb ** 2).cumsum(0) / (S_verb ** 2).sum()
        n_verb_tokens = verb_embeds.shape[0]
        v_rank_90 = (cumvar_verb < 0.90).sum().item() + 1
        v_rank_95 = (cumvar_verb < 0.95).sum().item() + 1
        v_rank_99 = (cumvar_verb < 0.99).sum().item() + 1

        print(f"\n=== Verb Subspace SVD ({n_verb_tokens} tokens) ===")
        print(f"  90% variance: {v_rank_90} components")
        print(f"  95% variance: {v_rank_95} components")
        print(f"  99% variance: {v_rank_99} components")
        n_comp_verb = min(n_components, n_verb_tokens)
        print(f"  Top {n_comp_verb} explain: {cumvar_verb[n_comp_verb-1]:.1%}")

        verb_basis = Vh_verb[:n_comp_verb]

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Singular value spectrum
    ax = axes[0]
    ax.semilogy(S_full[:200].cpu().numpy(), label='Full vocab', color='#4682B4')
    if S_verb is not None:
        ax.semilogy(S_verb[:min(200, len(S_verb))].cpu().numpy(), label='Verb tokens', color='#FF6347')
    ax.set_xlabel('Component index')
    ax.set_ylabel('Singular value (log)')
    ax.set_title('Singular Value Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Cumulative explained variance
    ax = axes[1]
    ax.plot(cumvar_full[:500].cpu().numpy(), label='Full vocab', color='#4682B4')
    if cumvar_verb is not None:
        ax.plot(cumvar_verb[:min(500, len(cumvar_verb))].cpu().numpy(), label='Verb tokens', color='#FF6347')
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95%')
    ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, label='99%')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative explained variance')
    ax.set_title('Explained Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Projection of verb tokens onto top-2 full vocab PCs
    ax = axes[2]
    coords_full = (embeds - mean_full) @ Vh_full[:2].T
    ax.scatter(coords_full[:, 0].cpu().numpy(), coords_full[:, 1].cpu().numpy(),
               s=3, alpha=0.1, color='#4682B4', label='All vocab')
    if verb_token_ids is not None and verb_token_mask is not None:
        verb_ids = verb_token_ids[verb_token_mask].unique().cpu()
        verb_coords = coords_full[verb_ids]
        ax.scatter(verb_coords[:, 0].cpu().numpy(), verb_coords[:, 1].cpu().numpy(),
                   s=30, alpha=0.9, color='#FF6347', edgecolors='black', linewidths=0.3, label='Verb tokens')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Top-2 PCs: Vocab vs Verbs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved SVD analysis plot to {save_path}")

    return {
        'full_basis': Vh_full[:n_components],
        'full_mean': mean_full,
        'full_singular_values': S_full,
        'verb_basis': verb_basis,
        'verb_mean': mean_verb,
    }


def dataset_from_hidden_states(hidden_states):
    """
    Build a dataset by extracting hidden states from each time step and layer.
    For the first time step, extract the hidden state of the last token.

    Args:
        hidden_states: Hidden states from the model output.
        
    Returns:
        array: An array of datasets for each time step.
    """
    dataset = []
    for layer in range(hidden_states.shape[0]):
            dataset.append(np.squeeze(hidden_states[layer].float().cpu().numpy()))

    # Convert the dataset to a NumPy array and remove the first layer (embedding layer)
    data_array = np.array(dataset).transpose(1, 0, 2)[:,1:,:]
    return data_array

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

    def forward(self, queries, context, roi_mask=None):
        hidden_states = queries
        
        for idx, layer in enumerate(self.cloned_layers):
            attn = layer.self_attn
            bsz, q_len, _ = hidden_states.shape
            kv_len = context.shape[1]

            # Pre-norm and residual
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            normed_q = layer.input_layernorm(hidden_states)
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

            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            
            # Output projection with LoRA
            attn_out = self.lora_o[idx](attn_out, attn.o_proj)

            hidden_states = residual + attn_out

            # FFN (frozen, no LoRA)
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states

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

    def forward(self, x, detr_feats, boxes, size, llava_feats, obj_embeds, attn_mask=None):
        x = self.norm(x)
        x_down = x + detr_feats + obj_embeds
        x_down = self.decoder(x_down.unsqueeze(0), self.llava_proj(llava_feats), attn_mask)
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
        #self.out_proj = Llavaproj(embed_dim, input_dim) 

    def forward(self, queries, keys_values, attn_mask):
        for layer in self.layers:
            queries = layer(queries, keys_values, attn_mask)
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

    def forward(self, queries, keys_values, attn_mask = None):
        q = self.norm1(queries)
        cross_attn_mask = None
        if attn_mask is not None:
            # attn_mask: True = can attend, False = block
            # nn.MHA attn_mask: True = block, False = attend (inverted!)
            cross_attn_mask = ~attn_mask  # Invert: now True = block

        self_attn_output, _ = self.self_attn(q, q, q)  # Self-attention
        queries = queries + self.dropout1(self_attn_output)  # Residual
        q = self.norm2(queries)
        #import pdb; pdb.set_trace()
        #cross_attn_output, _ = self.cross_attn(self.cross_attn_q_proj(q), self.cross_attn_k_proj(keys_values), self.cross_attn_v_proj(keys_values))
        cross_attn_output, _ = self.cross_attn(q, keys_values, keys_values,  attn_mask=cross_attn_mask)
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
        # #self.verb_classifier_ho = self.clip_head.linear_shortcut(self.verb_classifier_ho)
        #self.verb_classifier_ho = F.normalize(self.verb_classifier_ho, p=2, dim=1)
        self.verb_projection_ho = nn.Linear(4096, 117, bias=False)
        self.verb_projection_ho.weight.data = self.verb_classifier_ho
        for param in self.verb_projection_ho.parameters():
            param.requires_grad = False

        # Precompute verb token ids for logit lens
        verbs_ing = [
            "adjusting", "assembling", "blocking", "blowing", "boarding", "breaking",
            "brushing with", "buying", "carrying", "catching", "chasing", "checking",
            "cleaning", "controlling", "cooking", "cutting", "cutting with", "directing",
            "dragging", "dribbling", "drinking with", "driving", "drying", "eating",
            "eating at", "exiting", "feeding", "filling", "flipping", "flushing",
            "flying", "greeting", "grinding", "grooming", "herding", "hitting",
            "holding", "hopping on", "hosing", "hugging", "hunting", "inspecting",
            "installing", "jumping", "kicking", "kissing", "lassoing", "launching",
            "licking", "lying on", "lifting", "lighting", "loading", "losing",
            "making", "milking", "moving", "no interaction", "opening", "operating",
            "packing", "painting", "parking", "paying", "peeling", "petting",
            "picking", "picking up", "pointing", "pouring", "pulling", "pushing",
            "racing", "reading", "releasing", "repairing", "riding", "rowing",
            "running", "sailing", "scratching", "serving", "setting", "shearing",
            "signing", "sipping", "sitting at", "sitting on", "sliding", "smelling",
            "spinning", "squeezing", "stabbing", "standing on", "standing under",
            "sticking", "stirring", "stopping at", "straddling", "swinging", "tagging",
            "talking on", "teaching", "texting on", "throwing", "tying", "toasting",
            "training", "turning", "typing on", "walking", "washing", "watching",
            "waving", "wearing", "wielding", "zipping",
        ]
        tokenizer = self.clip_head['tokenizer']
        verb_token_ids = [tokenizer.encode(v)[1:] for v in verbs_ing]
        max_tokens = max(len(ids) for ids in verb_token_ids)
        padded = torch.zeros(len(verbs_ing), max_tokens, dtype=torch.long)
        mask = torch.zeros(len(verbs_ing), max_tokens, dtype=torch.bool)
        for i, ids in enumerate(verb_token_ids):
            padded[i, :len(ids)] = torch.tensor(ids)
            mask[i, :len(ids)] = True
        self.register_buffer('verb_token_ids', padded)
        self.register_buffer('verb_token_mask', mask)

        # self.text_2_queries = MLP(4096, 128, args.adapt_dim, 2)
        # self.ho_llava_2_queries = nn.Linear(4096, args.adapt_dim, bias = False)
        # self.h_llava_2_queries = nn.Linear(4096, args.adapt_dim, bias = False)
        # self.o_llava_2_queries = nn.Linear(4096, args.adapt_dim, bias = False)
        # self.ho_query_proj = MLP(512, 128, args.adapt_dim, 2)
        # self.h_query_proj = MLP(256, 128, args.adapt_dim, 2)
        # self.o_query_proj = MLP(256, 128, args.adapt_dim, 2)
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

        start_layer = 31
        num_layers = 1

        # Clone layers for H (human)
        # cloned_layers_h = nn.ModuleList([
        #     copy.deepcopy(self.clip_head["model"].model.layers[i])
        #     for i in range(start_layer, start_layer + num_layers)
        # ])
        # self.cross_attend_lora_h = CrossAttendWithLoRA(cloned_layers_h, lora_rank=8, lora_alpha=8)

        # # Clone layers for O (object)
        # cloned_layers_o = nn.ModuleList([
        #     copy.deepcopy(self.clip_head["model"].model.layers[i])
        #     for i in range(start_layer, start_layer + num_layers)
        # ])
        # self.cross_attend_lora_o = CrossAttendWithLoRA(cloned_layers_o, lora_rank=8, lora_alpha=8)

        # Clone layers for HO (human + object union)
        cloned_layers_ho = nn.ModuleList([
            copy.deepcopy(self.clip_head["model"].model.layers[i])
            for i in range(start_layer, start_layer + num_layers)
        ])
        self.cross_attend_lora_ho = CrossAttendWithLoRA(cloned_layers_ho, lora_rank=8, lora_alpha=8)

        # self.logit_scale_ho = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        #self.logit_scale_h = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.logit_scale_o = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        final_norm = self.clip_head["model"].model.norm
        self.final_norm = copy.deepcopy(final_norm)
        for param in self.final_norm.parameters():
            param.requires_grad = False

        anchor_ids = farthest_point_sampling(self.lm_head_embeddings, k=256)
        self.register_buffer('anchor_ids', anchor_ids)
        self.register_buffer('anchor_embeds', self.lm_head_embeddings[anchor_ids])
        import pdb; pdb.set_trace()

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
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)

            
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            #import pdb; pdb.set_trace()
            h_unique_indices, h_inverse_indices = torch.unique(x_keep, return_inverse=True)
            o_unique_indices, o_inverse_indices = torch.unique(y_keep, return_inverse=True)


            #human = labels[x_keep]
            #objects = targets[0]["object"]#.unique()
            objects = labels[y_keep].unique()
            candidate_texts = []
            for obj_idx in objects.cpu().tolist():
                candidates = [
                    ((verb_idx, object_idx), text.replace("a photo of a person ", ""))
                    for (verb_idx, object_idx), text in hico_text_label.hico_text_label.items()
                    if object_idx == obj_idx and verb_idx in self.seen_verb_idxs
                ]
                candidate_texts.extend(candidates)

            gt_bx_h = self.recover_boxes(targets[0]['boxes_h'], targets[0]['size'])
            gt_bx_o = self.recover_boxes(targets[0]['boxes_o'], targets[0]['size'])
            
            bbox_2_tokens = bbox_to_token((336,336),boxes, 24)
            bool_h = bbox_2_tokens[x_keep]
            bool_o = bbox_2_tokens[y_keep]
            bool_union = bool_h | bool_o


            img_start_idx, img_end_idx, prefix_prompt_end_idx = get_img_idx(self.clip_head, self.clip_head['model_name'], self.clip_head['tokenizer'],  "A photo of a person ")
            # results_per_object = []
            # for candidates in candidate_texts:
            #     #import pdb; pdb.set_trace()
            #     log_probs, probs, hidden_states, tgt_hidden_states = compute_conditional_likelihood_llava(
            #         self.clip_head,
            #         self.clip_head['model_name'],
            #         image.decompose()[0][b_idx:b_idx + 1].half(),
            #         (336,336),
            #         self.clip_head['tokenizer'],
            #         "Provide the correct human-object interaction in the image: a photo of a person ",
            #         #f"Provide the correct interaction between the person and the {hico_objects[candidates[0][-1]]}",
            #         #"Provide the correct human-object interaction that goes in "a photo of a person <interaction> an object",
            #         #"A photo of a person ",
            #         candidates[-1],   # Now just ["person boarding an airplane", "person directing an airplane", ...]
            #     )
            #     results_per_object.append({
            #         'candidates': candidates,
            #         'probs': probs,
            #     })
            with torch.no_grad():
                hidden_states, output, generated_ids, _= run_llava_model(
                    self.clip_head,
                    self.clip_head['model_name'],
                    image.decompose()[0][b_idx:b_idx + 1].to(torch.bfloat16),
                    (336,336),
                    self.clip_head['tokenizer'],
                    hidden_states=True,
                    text_prompt="Please describe this image in detail."
                )
            
            # Sort by probability (descending order)
            #results_per_object_sorted = sorted(results_per_object, key=lambda x: x['probs'], reverse=True)


            #llava_features = hidden_states[self.args.layer]  # keep native dtype (bfloat16)

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
            llava_features = hidden_states[31]
            #tsne_plot_verb_vs_lmhead(self.anchor_embeds, self.lm_head_embeddings)
            svd_results = svd_analysis(
                self.lm_head_embeddings,
                verb_token_ids=self.verb_token_ids,
                verb_token_mask=self.verb_token_mask,
                n_components=128,
                save_path='./svd_verb_analysis.png'
            )
            import pdb; pdb.set_trace()

            asd  = self.final_norm(llava_features) @  self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
            asd1 = llava_features @  self.lm_head_embeddings.T.to(hidden_states.device).to(torch.bfloat16)
            print(self.clip_head['tokenizer'].decode(torch.topk(asd1[0][450], 50, dim=-1)[1]))
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            #gt_labels = torch.ones([N])
            all_logits_collated.append(logits)
            #all_llava_logits_collated.append(llava_target)
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
        # valid_mask = llava_logits > 0
        # n_valid = valid_mask.sum()
        
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
            scores = torch.sigmoid(lg[x, y])
            #scores = lg[x, y]
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

