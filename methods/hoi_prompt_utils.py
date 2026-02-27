"""
hoi_prompt_utils.py — prompt helpers for the HOI SFT fine-tuning pipeline.

Defines the special <hoi_feat> token and utilities to:
  - Build the user message containing the image placeholder and N <hoi_feat> tokens
  - Build the target answer string from ground-truth HOI indices
  - Retrieve the token id of <hoi_feat> from an already-updated tokenizer
"""

HOI_FEAT_TOKEN = "<hoi_feat>"
ANSWER_SEP     = ", "   # separator between HOI descriptions in the target answer


# Imported lazily inside functions to avoid circular imports when this module
# is used as a pure-Python utility without the full HOILENS package installed.
def _get_hico_verb_object_list():
    from utils.hico_list import hico_verb_object_list
    return hico_verb_object_list


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_hoi_text_prompt(n_pairs: int) -> str:
    """
    Return the text portion of the user question containing n_pairs <hoi_feat>
    placeholder tokens.

    Args:
        n_pairs: number of detected H-O pairs (one <hoi_feat> token per pair)

    Returns:
        str — e.g.
          "The interaction features are <hoi_feat><hoi_feat>. Provide all
           human-object interactions in this image."
    """
    feat_str = HOI_FEAT_TOKEN * n_pairs
    return (
        f"The interaction features are {feat_str}. "
        "Provide all human-object interactions in this image."
    )


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
