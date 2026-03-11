"""
hoi_prompt_utils.py — prompt helpers for the HOI SFT fine-tuning pipeline.

Defines the special <hoi_feat> token and utilities to:
  - Build the user message containing the image placeholder and N <hoi_feat> tokens
  - Build the target answer string from ground-truth HOI indices
  - Retrieve the token id of <hoi_feat> from an already-updated tokenizer
"""

HOI_FEAT_TOKEN = "<hoi_feat>"
HOI_SEP_TOKEN  = "<hoi>"
HOI_NEG_TOKEN  = "<hoi_neg>"
ANSWER_SEP     = ", "   # separator between HOI descriptions in the target answer


# Imported lazily inside functions to avoid circular imports when this module
# is used as a pure-Python utility without the full HOILENS package installed.
def _get_hico_verb_object_list():
    from utils.hico_list import hico_verb_object_list
    return hico_verb_object_list


def _build_obj_to_hoi_map():
    """Returns dict: obj_idx (0-79) → sorted list of HOI indices for that object."""
    from utils.hico_utils import HOI_IDX_TO_OBJ_IDX
    obj_to_hoi: dict[int, list[int]] = {}
    for hoi_idx, obj_idx in enumerate(HOI_IDX_TO_OBJ_IDX):
        obj_to_hoi.setdefault(obj_idx, []).append(hoi_idx)
    return obj_to_hoi


# Module-level cache (built once on first call)
_OBJ_TO_HOI: dict[int, list[int]] | None = None

def _get_obj_to_hoi():
    global _OBJ_TO_HOI
    if _OBJ_TO_HOI is None:
        _OBJ_TO_HOI = _build_obj_to_hoi_map()
    return _OBJ_TO_HOI


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_hoi_text_prompt(
    n_pairs: int,
    caption: str | None = None,
    force_caption: bool = False,
) -> tuple[str, bool]:
    """
    Return (prompt_text, caption_in_prompt).

    If caption is provided, randomly selects between:
      (A) "<caption>. Provide all human-object interactions in this image <hoi_feat>..."
          → caption_in_prompt=True  (no gen_loss; caption already visible)
      (B) "The interaction features are <hoi_feat>... Provide all human-object
           interactions in this image."  (original)
          → caption_in_prompt=False (gen_loss on caption as answer)
    If caption is None, always uses (B).
    Set force_caption=True (e.g. at test time) to always use format (A) when a
    caption is available, skipping the random coin flip.
    """
    import random
    feat_str = HOI_FEAT_TOKEN * n_pairs
    #import pdb; pdb.set_trace()
    return (
        f"{caption}. "
        f"Provide all human-object interactions in this image {feat_str}.",
        True,
    )
    # return (
    #     f"The interaction features are {feat_str}. "
    #     "Provide all human-object interactions in this image.",
    #     False,
    # )


def build_hoi_messages(pil_image, n_pairs: int) -> list:
    """
    Build a Qwen2.5-VL chat message list (content-dict format) for the HOI task.

    The message contains:
      1. An image entry (PIL image)
      2. A text prompt with exactly n_pairs occurrences of HOI_FEAT_TOKEN

    Compatible with processor.apply_chat_template() when called with
    ``tokenize=False``, then passed to processor(text=..., images=...).

    Args:
        pil_image : PIL.Image
        n_pairs   : number of H-O pairs

    Returns:
        list[dict] — a single-turn user message in Qwen content-dict format
    """
    from methods.qwen_utils import QWEN_IMAGE_SIZE, QWEN_PATCH_SIZE, QWEN_MERGE_SIZE
    _stride = QWEN_PATCH_SIZE * QWEN_MERGE_SIZE  # 28
    # Snap to nearest stride multiple
    snapped = (QWEN_IMAGE_SIZE // _stride) * _stride

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                    "resized_height": snapped,
                    "resized_width":  snapped,
                },
                {"type": "text", "text": build_hoi_text_prompt(n_pairs)},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Target text builder
# ---------------------------------------------------------------------------

def build_target_text(hoi_indices) -> str:
    """
    Build the SFT target answer string from a list/tensor of HOI class indices
    (0-indexed, HICO-DET 600-class scheme).

    Each index maps to a (verb, object) pair from hico_verb_object_list.
    Duplicates are removed while preserving order of first occurrence.

    Args:
        hoi_indices: iterable of ints (HOI class indices 0..599)

    Returns:
        str: comma-space-separated descriptions, e.g.
             "person riding a bicycle, person holding a bottle"
        If hoi_indices is empty returns "no interaction".
    """
    hico_verb_object_list = _get_hico_verb_object_list()
    seen  = set()
    parts = []
    for idx in hoi_indices:
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        verb, obj = hico_verb_object_list[idx]
        parts.append(f"person {verb} {obj}")
    return ANSWER_SEP.join(parts) if parts else "no interaction"


def build_target_text_detailed(hoi_indices) -> str:
    """
    Build a numbered, bracketed target answer string for SFT.

    Each unique HOI is listed as:
        [N] person {verb} {object}

    This is the structured fallback format used when no pre-extracted
    Qwen caption is available. It is more explicit than the flat
    comma-separated format and mirrors the bracket notation expected
    by the extraction prompt (see scripts/extract_hoi_captions.py).

    Args:
        hoi_indices: iterable of ints (HOI class indices 0..599)

    Returns:
        str: e.g.
             "[1] person riding a bicycle\n[2] person holding a bottle"
        If hoi_indices is empty returns "none".
    """
    hico_verb_object_list = _get_hico_verb_object_list()
    seen  = set()
    parts = []
    for idx in hoi_indices:
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        verb, obj = hico_verb_object_list[idx]
        parts.append(f"[{len(parts)+1}] person {verb} {obj}")
    return "\n".join(parts) if parts else "none"


def build_target_text_with_boxes(
    target: dict,
    unseen_ids: set[int] | None = None,
) -> str:
    """
    Build target text with bounding boxes for each HOI instance (no deduplication).

    Positive entries (GT HOIs) end with <hoi>.
    Negative entries (seen-but-not-GT HOIs sharing the same object category and
    boxes) end with <hoi_neg>. Unseen HOI classes are never used as negatives.

    Format per line:
        Person [x1,y1,x2,y2] {verb} {obj} [x1,y1,x2,y2]<hoi>      ← positive
        Person [x1,y1,x2,y2] {verb2} {obj} [x1,y1,x2,y2]<hoi_neg> ← negative

    Boxes are converted from normalized cxcywh (post-transform) to Qwen-style
    relative coordinates in range [0, 1000].

    Args:
        target:     dict with keys 'hoi' [N], 'object' [N], 'boxes_h' [N,4],
                    'boxes_o' [N,4]. Boxes in normalized cxcywh format.
        unseen_ids: set of HOI indices held out (not seen) during training.
                    These are never added as negatives. Pass None or empty set
                    for the default (fully supervised) setting.

    Returns:
        str: newline-separated lines, one per positive or negative entry.
        If empty returns "none".
    """
    import torch
    hico_verb_object_list = _get_hico_verb_object_list()
    obj_to_hoi            = _get_obj_to_hoi()

    if unseen_ids is None:
        unseen_ids = set()

    hoi_indices = target.get("hoi", torch.zeros(0, dtype=torch.long))
    obj_indices = target.get("object", torch.zeros(0, dtype=torch.long))
    boxes_h     = target.get("boxes_h", torch.zeros(0, 4))
    boxes_o     = target.get("boxes_o", torch.zeros(0, 4))

    n = min(len(hoi_indices), len(boxes_h), len(boxes_o), len(obj_indices))
    if n == 0:
        return "none"
    hoi_indices = hoi_indices[:n]
    obj_indices = obj_indices[:n]
    boxes_h     = boxes_h[:n]
    boxes_o     = boxes_o[:n]

    def cxcywh_to_xyxy_qwen(box):
        cx, cy, w, h = box.tolist()
        x1 = int(round((cx - w / 2) * 1000))
        y1 = int(round((cy - h / 2) * 1000))
        x2 = int(round((cx + w / 2) * 1000))
        y2 = int(round((cy + h / 2) * 1000))
        return x1, y1, x2, y2

    parts = []
    for i, idx in enumerate(hoi_indices):
        idx_int = int(idx)
        verb, obj = hico_verb_object_list[idx_int]
        h_box = cxcywh_to_xyxy_qwen(boxes_h[i])
        o_box = cxcywh_to_xyxy_qwen(boxes_o[i])
        h_str = f"[{h_box[0]},{h_box[1]},{h_box[2]},{h_box[3]}]"
        o_str = f"[{o_box[0]},{o_box[1]},{o_box[2]},{o_box[3]}]"
        parts.append(f"Person {h_str} {verb} {obj} {o_str}{HOI_SEP_TOKEN}")

    return "\n".join(parts) if parts else "none"


# ---------------------------------------------------------------------------
# Token id helper
# ---------------------------------------------------------------------------

def get_hoi_token_id(tokenizer) -> int:
    """
    Return the token id of HOI_FEAT_TOKEN from a tokenizer that already has it
    registered as a special token.

    Raises ValueError if the token is unknown (unk_token_id returned).
    """
    tok_id = tokenizer.convert_tokens_to_ids(HOI_FEAT_TOKEN)
    if tok_id == tokenizer.unk_token_id:
        raise ValueError(
            f"{HOI_FEAT_TOKEN!r} not found in tokenizer vocabulary. "
            "Call tokenizer.add_special_tokens({'additional_special_tokens': "
            "['<hoi_feat>']}) and model.resize_token_embeddings(len(tokenizer)) first."
        )
    return tok_id
