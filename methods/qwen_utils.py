"""
Qwen2.5-VL-3B utility functions for HOILENS.

Drop-in replacement for methods/llava_utils.py — mirrors the same API so that
ours_llava_new_old.py (and all other model files) can switch to Qwen2.5-VL-3B
with minimal code changes.

Key architectural differences vs. LLaVA-v1.5-7B that callers must adapt:
  ┌────────────────────┬──────────────────────┬─────────────────────────┐
  │                    │ LLaVA-v1.5-7B        │ Qwen2.5-VL-3B           │
  ├────────────────────┼──────────────────────┼─────────────────────────┤
  │ hidden_size        │ 4096                 │ 2048 (QWEN_HIDDEN_SIZE) │
  │ num_layers         │ 32                   │ 28   (QWEN_NUM_LAYERS)  │
  │ image grid (336²)  │ 24×24 = 576 tokens   │ 12×12 = 144 tokens      │
  │ spatial_scale      │ 24 / 336             │ 12 / 336                │
  │ recommended start  │ start_layer = 30     │ start_layer = 26        │
  │ attention          │ MHA (32 Q heads)     │ GQA (16 Q / 8 KV heads) │
  │ image input        │ pre-processed tensor │ PIL Image               │
  └────────────────────┴──────────────────────┴─────────────────────────┘

Integration checklist for ours_qwen_new_old.py:
  1. Swap import:
       from methods.llava_utils import ... → from methods.qwen_utils import ...
  2. Swap all llava function names → qwen equivalents (same signatures).
  3. Pass PIL images to run_qwen_model instead of pre-processed tensors.
       Option A – add im[2] (PIL) to the dataset tuple.
       Option B – call tensor_to_pil(images_orig_tensor) on the fly.
  4. Update hardcoded shape constants in compute_sim_scores:
       .view(1, 24, 24, 4096)  →  .view(1, QWEN_GRID_H, QWEN_GRID_W, QWEN_HIDDEN_SIZE)
       spatial_scale=24/336    →  spatial_scale=QWEN_GRID_H/QWEN_IMAGE_SIZE
  5. Change self.start_layer = QWEN_START_LAYER  (26 instead of 30).
  6. Load Qwen-specific pre-computed tensors (lm_head_embedding_3b.pt etc.).
       See get_lm_head_embeddings_qwen() below to produce them.
  7. In CrossAttendWithLoRA, handle GQA:
       k_proj / v_proj output dim = QWEN_KV_HIDDEN_SIZE (1024), not 2048.
       Replace the single hidden_size used for all LoRA adapters with:
         q_dim = layer.self_attn.q_proj.weight.shape[0]   # 2048
         kv_dim = layer.self_attn.k_proj.weight.shape[0]  # 1024
       and use kv_dim for lora_k / lora_v.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration, 
    HfArgumentParser, 
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError(
        "qwen-vl-utils is required.  Install with:  pip install qwen-vl-utils"
    )

# ---------------------------------------------------------------------------
# Architecture constants for Qwen2.5-VL-3B-Instruct at 336×336 resolution
# ---------------------------------------------------------------------------
QWEN_MODEL_ID         = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_HIDDEN_SIZE      = 2048        # LLM hidden dim      (LLaVA 7B: 4096)
QWEN_NUM_LAYERS       = 28          # transformer layers  (LLaVA 7B: 32)
QWEN_IMAGE_SIZE       = 448         # fixed input resolution
QWEN_PATCH_SIZE       = 14         # ViT patch size in pixels
QWEN_MERGE_SIZE       = 2           # spatial merge factor (2×2 → 4 patches / token)
QWEN_GRID_H           = QWEN_IMAGE_SIZE // (QWEN_PATCH_SIZE * QWEN_MERGE_SIZE)  # 16
QWEN_GRID_W           = QWEN_IMAGE_SIZE // (QWEN_PATCH_SIZE * QWEN_MERGE_SIZE)  # 16
QWEN_NUM_IMAGE_TOKENS = QWEN_GRID_H * QWEN_GRID_W                               # 256
QWEN_START_LAYER      = 26          # analogous to LLaVA start_layer = 30
QWEN_NUM_HEADS        = 16          # query attention heads
QWEN_NUM_KV_HEADS     = 8           # key/value heads (GQA)
QWEN_HEAD_DIM         = QWEN_HIDDEN_SIZE // QWEN_NUM_HEADS  # 128
QWEN_KV_HIDDEN_SIZE   = QWEN_NUM_KV_HEADS * QWEN_HEAD_DIM  # 1024


# ---------------------------------------------------------------------------
# Vocabulary / LM-head embeddings
# ---------------------------------------------------------------------------

def get_vocab_embeddings_qwen(model, tokenizer, device="cuda"):
    """
    Get token embeddings from the Qwen embedding table.
    Returns [1, vocab_size, hidden_size].
    Mirrors get_vocab_embeddings_llava().
    """
    vocab = tokenizer.get_vocab()
    llm_tokens = (
        torch.tensor(list(vocab.values()), dtype=torch.long)
        .unsqueeze(0)
        .to(device)
    )
    return model.get_input_embeddings()(llm_tokens)


def get_lm_head_embeddings_qwen(model, device="cuda"):
    """
    Extract the LM head weight matrix [vocab_size, hidden_size].

    Use this to generate lm_head_embedding_3b.pt, which replaces
    lm_head_embedding_7b.pt in HOILLAVA.__init__:

        self.lm_head_embeddings = get_lm_head_embeddings_qwen(
            self.clip_head["model"], device
        )
    """
    return model.lm_head.weight.detach().to(device)


# ---------------------------------------------------------------------------
# Input preparation helpers
# ---------------------------------------------------------------------------

def _build_qwen_messages(pil_image, text_prompt, target_h=QWEN_IMAGE_SIZE, target_w=QWEN_IMAGE_SIZE):
    """Compose a Qwen-style chat message list."""
    if pil_image is not None:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image,
                        # Snap to the nearest multiple of (patch_size * merge_size) = 28
                        "resized_height": (target_h // (QWEN_PATCH_SIZE * QWEN_MERGE_SIZE)) * (QWEN_PATCH_SIZE * QWEN_MERGE_SIZE),
                        "resized_width":  (target_w // (QWEN_PATCH_SIZE * QWEN_MERGE_SIZE)) * (QWEN_PATCH_SIZE * QWEN_MERGE_SIZE),
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
    return [{"role": "user", "content": text_prompt}]


def _build_qwen_inputs(processor, pil_image, text_prompt, device, target_h=QWEN_IMAGE_SIZE, target_w=QWEN_IMAGE_SIZE):
    """
    Run the Qwen2.5-VL processor and move all tensors to `device`.
    Returns a dict ready to unpack into the model.
    """
    messages    = _build_qwen_messages(pil_image, text_prompt, target_h, target_w)
    text        = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Pass the PIL image directly to the processor — this is more robust than
    # relying on process_vision_info, which may not handle PIL in all versions.
    if pil_image is not None:
        inputs = processor(text=[text], images=[pil_image], return_tensors="pt")
    else:
        inputs = processor(text=[text], return_tensors="pt")

    return {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}


def _get_qwen_grid(image_sizes):
    """Given (H, W), return (grid_h, grid_w) after snapping to 28-pixel stride."""
    _stride = QWEN_PATCH_SIZE * QWEN_MERGE_SIZE  # 28
    H, W = image_sizes
    return H // _stride, W // _stride


def _get_image_token_range(input_ids, processor):
    """
    Find the contiguous block of <|image_pad|> tokens in input_ids.

    Returns:
        img_start (int): index of the first image-pad token
        img_end   (int): index one past the last image-pad token
    """
    tok             = processor.tokenizer
    vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
    image_pad_id    = tok.convert_tokens_to_ids("<|image_pad|>")

    ids     = input_ids[0].tolist()
    vs_idx  = ids.index(vision_start_id) + 1          # skip <|vision_start|> itself
    num_img = sum(1 for t in ids[vs_idx:] if t == image_pad_id)
    return vs_idx, vs_idx + num_img


# ---------------------------------------------------------------------------
# Image conversion utility
# ---------------------------------------------------------------------------

def tensor_to_pil(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Convert a normalised image tensor [1, 3, H, W] → PIL Image.

    Use this when only the normalised tensor is available from the data
    pipeline and you need a PIL image to pass to run_qwen_model.
    The default mean/std match COCO / torchvision ImageNet normalisation.
    """
    t = tensor.squeeze(0).float().cpu().clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t   = t.clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Core inference: forward pass
# ---------------------------------------------------------------------------

def run_qwen_model(
    model,
    model_name,
    pil_image,
    image_sizes,
    tokenizer,
    text_prompt=None,
    hidden_states=False,
):
    """
    Forward pass through Qwen2.5-VL-3B.  Mirrors run_llava_model().

    Args:
        model        : state dict returned by load_qwen_state()
        model_name   : ignored — kept for API compatibility
        pil_image    : PIL.Image for the input image.
                       If you only have a normalised tensor, call tensor_to_pil() first.
        image_sizes  : (H, W) — kept for API compatibility, not used by Qwen
        tokenizer    : kept for API compatibility (processor is used internally)
        text_prompt  : optional string (defaults to ".")
        hidden_states: kept for API compatibility

    Returns:
        img_hidden_states : Tensor [num_layers+1, 1, 144, 2048]
                            (LLaVA equivalent: [layers+1, 1, 576, 4096])
        output            : raw model output dict
        None, None        : padding for API compatibility
    """
    if text_prompt is None:
        text_prompt = "."

    processor  = model["processor"]
    qwen_model = model["model"]
    device     = next(qwen_model.parameters()).device

    target_h, target_w = image_sizes if image_sizes is not None else (QWEN_IMAGE_SIZE, QWEN_IMAGE_SIZE)
    inputs    = _build_qwen_inputs(processor, pil_image, text_prompt, device, target_h, target_w)
    input_ids = inputs["input_ids"]
    #import pdb; pdb.set_trace()
    with torch.inference_mode():
        output = qwen_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # hidden_states: tuple of (num_layers+1) tensors, each [1, seq_len, hidden_size]
    hidden                = torch.stack(output.hidden_states)  # [L+1, 1, seq, dim]
    img_start, img_end    = _get_image_token_range(input_ids, processor)
    img_hidden            = hidden[:, :, img_start:img_end, :]  # [L+1, 1, 144, 2048]

    #import pdb; pdb.set_trace()

    return img_hidden, output, None, None


# ---------------------------------------------------------------------------
# Core inference: generation
# ---------------------------------------------------------------------------

def generate_qwen_model(
    model,
    model_name,
    pil_image,
    image_sizes,
    tokenizer,
    text_prompt=None,
    hidden_states=False,
):
    """
    Text generation with Qwen2.5-VL-3B.  Mirrors generate_llava_model().

    Args:
        pil_image : PIL.Image or None (text-only mode)
        (other)   : same as run_qwen_model

    Returns:
        img_hidden_states : [L+1, 1, 144, 2048]  (None if text-only)
        generated_text    : decoded string
        generated_ids     : token id tensor
        scores            : stacked generation scores
    """
    if text_prompt is None:
        text_prompt = "Write a detailed description."

    processor  = model["processor"]
    qwen_model = model["model"]
    device     = next(qwen_model.parameters()).device

    inputs    = _build_qwen_inputs(processor, pil_image, text_prompt, device)
    input_ids = inputs["input_ids"]

    with torch.inference_mode():
        output = qwen_model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
        )

    # Hidden states of the prompt tokens (from the first generation step)
    hidden = torch.stack(output.hidden_states[0])  # [L+1, 1, prompt_len, dim]

    if pil_image is not None:
        img_start, img_end = _get_image_token_range(input_ids, processor)
        img_hidden         = hidden[:, :, img_start:img_end, :]
    else:
        img_hidden = None

    generated_ids  = output.sequences
    trimmed        = generated_ids[:, input_ids.shape[1]:]   # strip prompt tokens
    generated_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    return img_hidden, generated_text, generated_ids, torch.stack(output.scores, dim=0).squeeze(1)


# ---------------------------------------------------------------------------
# Conditional likelihood scoring
# ---------------------------------------------------------------------------

def compute_conditional_likelihood_qwen(
    model,
    model_name,
    pil_image,
    image_sizes,
    tokenizer,
    prefix_prompt,
    target_text,
):
    """
    Average log-likelihood of target_text given image + prefix.
    Mirrors compute_conditional_likelihood_llava().

    Returns:
        log_probs            : [avg_log_prob]
        probs                : [exp(avg_log_prob)]
        visual_hidden_states : [L+1, 1, 144, 2048]
        tgt_hidden_states    : [L+1, 1, target_len, 2048]
    """
    processor  = model["processor"]
    qwen_model = model["model"]
    device     = next(qwen_model.parameters()).device

    inputs      = _build_qwen_inputs(processor, pil_image, prefix_prompt, device)
    input_ids   = inputs["input_ids"]
    prefix_len  = input_ids.shape[1]

    target_ids = processor.tokenizer.encode(target_text, add_special_tokens=False)
    target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)

    full_input_ids        = torch.cat([input_ids, target_ids.unsqueeze(0)], dim=1)
    full_inputs           = dict(inputs)
    full_inputs["input_ids"]      = full_input_ids
    full_inputs["attention_mask"] = torch.ones_like(full_input_ids)

    with torch.inference_mode():
        outputs = qwen_model(
            **full_inputs,
            return_dict=True,
            output_hidden_states=True,
        )

    logits            = outputs.logits
    hidden_all        = torch.stack(outputs.hidden_states)  # [L+1, 1, seq, dim]

    img_start, img_end    = _get_image_token_range(input_ids, processor)
    visual_hidden         = hidden_all[:, :, img_start:img_end, :]

    target_len      = len(target_ids)
    target_logits   = logits[0, prefix_len - 1 : prefix_len - 1 + target_len, :]
    tgt_hidden      = hidden_all[:, :, prefix_len - 1 : prefix_len - 1 + target_len, :]

    log_probs_dist  = F.log_softmax(target_logits, dim=-1)
    token_log_probs = log_probs_dist[range(target_len), target_ids]
    avg_log_prob    = token_log_probs.mean().item()

    return [avg_log_prob], [np.exp(avg_log_prob)], visual_hidden, tgt_hidden


def compute_binary_conditional_likelihood_qwen(
    model,
    model_name,
    pil_image,
    image_sizes,
    tokenizer,
    prefix_prompt,
    target_text,
):
    """
    Yes / No probability for a binary question.
    Mirrors compute_binary_conditional_likelihood_llava().

    Returns:
        yes_prob, no_prob : floats
    """
    processor  = model["processor"]
    qwen_model = model["model"]
    device     = next(qwen_model.parameters()).device

    inputs = _build_qwen_inputs(processor, pil_image, prefix_prompt, device)
    with torch.inference_mode():
        outputs = qwen_model(**inputs, return_dict=True)

    log_probs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)

    yes_ids  = processor.tokenizer("Yes", add_special_tokens=False).input_ids
    no_ids   = processor.tokenizer("No",  add_special_tokens=False).input_ids

    return log_probs[0, yes_ids].exp().item(), log_probs[0, no_ids].exp().item()


# ---------------------------------------------------------------------------
# Image token index helper
# ---------------------------------------------------------------------------

def get_img_idx_qwen(model, model_name, tokenizer, text_prompt):
    """
    Return (img_start_idx, img_end_idx, actual_prefix_len).
    Mirrors get_img_idx() from llava_utils.

    For Qwen2.5-VL-3B at 336×336 the image region is always 144 tokens.
    To get exact positions for a real input, use _get_image_token_range()
    on the output of _build_qwen_inputs().
    """
    img_start = 1                                    # right after <|vision_start|>
    img_end   = img_start + QWEN_NUM_IMAGE_TOKENS    # 145
    return img_start, img_end, img_end


# ---------------------------------------------------------------------------
# Logit-lens retrieval
# ---------------------------------------------------------------------------

def retrieve_logit_lens_qwen(state, pil_image, args, text_prompt=None):
    """
    Retrieve intermediate hidden states for logit-lens analysis.
    Mirrors retrieve_logit_lens_llava().

    Returns:
        cls_proj         : None  (Qwen has no CLS token)
        img_hidden_states: [L+1, 1, 144, 2048]
        last_hidden_states: [L+1, 1, hidden_size]
    """
    img_hidden, output, _, _ = run_qwen_model(
        state,
        state["model_name"],
        pil_image,
        (QWEN_IMAGE_SIZE, QWEN_IMAGE_SIZE),
        state["processor"].tokenizer,
        text_prompt=text_prompt,
        hidden_states=True,
    )
    hidden_all         = torch.stack(output.hidden_states)
    last_hidden_states = hidden_all[:, :, -1, :]
    return None, img_hidden, last_hidden_states


# ---------------------------------------------------------------------------
# Hidden text embedding
# ---------------------------------------------------------------------------

def _get_hidden_text_embedding_qwen(
    target_word, model, vocab_embeddings, tokenizer, layer=5, device="cuda"
):
    token_ids = tokenizer.encode(target_word, add_special_tokens=False)
    input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model.model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = torch.stack(out.hidden_states)   # [L+1, 1, seq, dim]
    return hidden[layer, 0, -1].unsqueeze(0)


# ---------------------------------------------------------------------------
# Model loading — mirrors load_llava_state()
# ---------------------------------------------------------------------------

def load_qwen_state(rank):
    """
    Load Qwen2.5-VL-3B-Instruct and return a state dict with the same
    keys as load_llava_state() for seamless integration.

    Extra keys (not in llava_utils):
        'processor'         : Qwen2_5_VLProcessor (required for image pre-proc)
        'hidden_size'       : 2048
        'grid_h', 'grid_w'  : 12, 12
        'num_image_tokens'  : 144
        'start_layer'       : 26
        'num_layers'        : 28

    Typical usage in build_detector():
        model = load_qwen_state(rank)
    """
    device_str = f"cuda:{rank}"

    try:
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device_str,
        )
    except Exception:
        # Fallback if flash-attn is unavailable
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=device_str,
        )
    qwen_model = qwen_model.to(torch.bfloat16)

    # Allow native resolution — actual size is driven by resized_height/resized_width
    # in _build_qwen_messages, so min/max_pixels are just a fallback safety bound.
    _stride = QWEN_PATCH_SIZE * QWEN_MERGE_SIZE  # 28
    processor = AutoProcessor.from_pretrained(
        QWEN_MODEL_ID,
        min_pixels=_stride * _stride,          # at least 1 token per dim
        max_pixels=1344 * 1344,                # generous upper bound (~48×48 tokens)
    )
    tokenizer = processor.tokenizer

    vocab            = tokenizer.get_vocab()
    vocab_embeddings = get_vocab_embeddings_qwen(qwen_model, tokenizer, device=device_str)

    _state_ref = {
        "model": qwen_model,
        "processor": processor,
        "model_name": QWEN_MODEL_ID,
    }

    register_hook     = lambda hook, layer: qwen_model.model.language_model.layers[layer].register_forward_hook(hook)
    register_pre_hook = lambda pre_hook, layer: qwen_model.model.language_model.layers[layer].register_forward_pre_hook(pre_hook)

    hidden_layer_embedding = lambda text, layer: _get_hidden_text_embedding_qwen(
        text, qwen_model, vocab_embeddings, tokenizer, layer, device=device_str
    )

    return {
        # ── Keys matching llava_utils.load_llava_state() ──────────────────
        "vocabulary":             vocab,
        "vocab_embeddings":       vocab_embeddings,
        "tokenizer":              tokenizer,
        "execute_model":          lambda pil_image, text_prompt=None, **kw: run_qwen_model(
            _state_ref, QWEN_MODEL_ID,
            pil_image, (QWEN_IMAGE_SIZE, QWEN_IMAGE_SIZE),
            tokenizer, text_prompt=text_prompt,
        ),
        "register_hook":          register_hook,
        "register_pre_hook":      register_pre_hook,
        "hidden_layer_embedding": hidden_layer_embedding,
        "model":                  qwen_model,
        "model_name":             QWEN_MODEL_ID,
        "image_processor":        processor,   # alias — llava_utils stores CLIPImageProcessor here
        # ── Qwen-specific metadata ────────────────────────────────────────
        "processor":              processor,
        "hidden_size":            QWEN_HIDDEN_SIZE,
        "grid_h":                 QWEN_GRID_H,
        "grid_w":                 QWEN_GRID_W,
        "num_image_tokens":       QWEN_NUM_IMAGE_TOKENS,
        "start_layer":            QWEN_START_LAYER,
        "num_layers":             QWEN_NUM_LAYERS,
    }


# ---------------------------------------------------------------------------
# Shared projection / embedding utilities  (identical to llava_utils.py)
# ---------------------------------------------------------------------------

def get_device_from_module(module):
    return next(module.parameters()).device

