"""
qwen_utils_joint.py — utilities for the joint single-pass HOI architecture.

Uses Qwen2.5-VL's proper embedding API (following qwen-vl-series-finetune):
  - get_input_embeddings()(input_ids)           → text embeddings
  - get_image_features(pixel_values, grid_thw)  → ViT image embeddings
  - masked_scatter to merge image into sequence
  - get_rope_index(...)                          → proper 3D position_ids

All constants and functions from qwen_utils.py are re-exported so callers
only need to import from this file.
"""

import torch

# ---------------------------------------------------------------------------
# Re-export everything from qwen_utils so callers only need one import
# ---------------------------------------------------------------------------
from methods.qwen_utils import (
    # constants
    QWEN_MODEL_ID,
    QWEN_HIDDEN_SIZE,
    QWEN_NUM_LAYERS,
    QWEN_IMAGE_SIZE,
    QWEN_PATCH_SIZE,
    QWEN_MERGE_SIZE,
    QWEN_GRID_H,
    QWEN_GRID_W,
    QWEN_NUM_IMAGE_TOKENS,
    QWEN_START_LAYER,
    QWEN_NUM_HEADS,
    QWEN_NUM_KV_HEADS,
    QWEN_HEAD_DIM,
    QWEN_KV_HIDDEN_SIZE,
    # helpers
    _build_qwen_inputs,
    _get_image_token_range,
    _get_qwen_grid,
    tensor_to_pil,
    get_lm_head_embeddings_qwen,
    get_vocab_embeddings_qwen,
    # inference
    run_qwen_model,
    generate_qwen_model,
    compute_conditional_likelihood_qwen,
    retrieve_logit_lens_qwen,
    get_img_idx_qwen,
    # loading
    load_qwen_state,
)


# ---------------------------------------------------------------------------
# Joint-pass helper: build full-sequence inputs_embeds via proper Qwen API
# ---------------------------------------------------------------------------

def get_qwen_inputs_embeds(model_state, pil_image, image_sizes, text_prompt="."):
    """
    Build full-sequence inputs_embeds using Qwen2.5-VL's native embedding API,
    mirroring the approach in qwen-vl-series-finetune/monkey_patch_forward.py
    (qwen2_5_mixed_modality_forward).

    Steps:
      1. Run processor to get input_ids, pixel_values, image_grid_thw
      2. get_input_embeddings()(input_ids)  →  text embeddings  [1, seq, 2048]
      3. get_image_features(pixel_values, image_grid_thw)  →  ViT embeddings
      4. get_placeholder_mask(...)  →  boolean mask over image-pad positions
      5. masked_scatter to splice ViT embeddings into the sequence
      6. get_rope_index(...)  →  3D position_ids  [3, 1, seq]

    Args:
        model_state  : state dict from load_qwen_state()
        pil_image    : PIL.Image
        image_sizes  : (H, W) for resizing inside the processor
        text_prompt  : appended text (default ".")

    Returns:
        inputs_embeds  : Tensor [1, seq_len, QWEN_HIDDEN_SIZE]  (detached)
        position_ids   : Tensor [3, 1, seq_len]  (detached) — 3D mrope
        attention_mask : Tensor [1, seq_len]
        img_start      : int — first image-pad token index in the sequence
        img_end        : int — one past the last image-pad token
    """
    processor  = model_state["processor"]
    qwen_model = model_state["model"]
    vlm        = qwen_model.model          # Qwen2_5_VLModel
    device     = next(qwen_model.parameters()).device

    target_h, target_w = image_sizes
    inputs         = _build_qwen_inputs(
        processor, pil_image, text_prompt, device, target_h, target_w
    )
    input_ids      = inputs["input_ids"]
    pixel_values   = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    attention_mask = inputs.get(
        "attention_mask",
        torch.ones_like(input_ids),
    )

    with torch.no_grad():
        # 1. Text embeddings via embed_tokens
        inputs_embeds = vlm.get_input_embeddings()(input_ids)   # [1, seq, 2048]

        # 2-4. Merge ViT image features into the sequence
        if pixel_values is not None:
            image_embeds = vlm.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = vlm.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # 5. Compute 3D position_ids via mrope
        position_ids, _ = vlm.get_rope_index(
            input_ids,
            image_grid_thw,
            None,                   # video_grid_thw
            attention_mask=attention_mask,
        )                           # [3, 1, seq_len]

    img_start, img_end = _get_image_token_range(input_ids, processor)

    return inputs_embeds.detach(), position_ids.detach(), attention_mask, img_start, img_end
