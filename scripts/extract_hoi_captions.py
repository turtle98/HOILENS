"""
scripts/extract_hoi_captions.py — Offline pre-extraction of detailed HOI captions
using the base Qwen2.5-VL-3B-Instruct model (no fine-tuning / LoRA).

For each training image the script:
  1. Reads the GT HOI annotation (verb, object, bounding boxes) from the JSON.
  2. Builds a spatially-grounded prompt that tells Qwen:
       - which interactions are present (verb + object label)
       - WHERE each person and object are (normalized bounding boxes)
  3. Generates one detailed description per interaction with the base model
     using greedy decoding (reproducible and fast).
  4. Saves results incrementally to a JSON file:
       { filename: caption_string, ... }

The saved captions are used by HOISFTDataset (--caption-file) as richer SFT
supervision signals instead of the plain "person VERB OBJECT" format.

Usage (single GPU):
    python scripts/extract_hoi_captions.py \\
        --data-root /path/to/hicodet \\
        --partition train2015 \\
        --output     outputs/hico_train_captions.json \\
        --device     cuda:0 \\
        --max-new-tokens 200

For multi-GPU parallelism, shard by index range:
    # GPU 0 handles first half
    python scripts/extract_hoi_captions.py ... --start 0     --end 19000 --device cuda:0 &
    # GPU 1 handles second half
    python scripts/extract_hoi_captions.py ... --start 19000 --end 38118 --device cuda:1 &
    # Then merge the two JSON files afterward.

Output format (one entry per image):
    {
      "HICO_train2015_00000001.jpg": "[1] person riding a bicycle: The person...",
      ...
    }
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Allow project-root imports when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.hico_list import hico_verb_object_list


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def load_annotation(data_root: str, partition: str) -> dict:
    """Load HICO-DET annotation JSON and return the raw dict."""
    anno_path = os.path.join(data_root, f"instances_{partition}.json")
    with open(anno_path, "r") as f:
        return json.load(f)


def _norm_box(box, img_w: float, img_h: float) -> tuple:
    """Normalize [x1,y1,x2,y2] pixel coords → (0-1) floats, clipped."""
    x1, y1, x2, y2 = box
    return (
        round(max(0.0, x1 / img_w), 3),
        round(max(0.0, y1 / img_h), 3),
        round(min(1.0, x2 / img_w), 3),
        round(min(1.0, y2 / img_h), 3),
    )


def build_hoi_entries(anno_entry: dict, img_w: float, img_h: float) -> list[dict]:
    """
    Deduplicate HOI types in one annotation entry and attach representative boxes.

    Returns a list of dicts, one per unique HOI type:
        {
          "hoi_idx": int,
          "verb": str,
          "object": str,
          "box_h": (x1, y1, x2, y2),   # normalized
          "box_o": (x1, y1, x2, y2),   # normalized
        }
    """
    seen_hoi = {}  # hoi_idx → first occurrence index
    for k, hoi_idx in enumerate(anno_entry["hoi"]):
        if hoi_idx not in seen_hoi:
            seen_hoi[hoi_idx] = k

    entries = []
    for hoi_idx, k in seen_hoi.items():
        verb, obj = hico_verb_object_list[hoi_idx]
        if verb == "no interaction":
            continue
        bh = _norm_box(anno_entry["boxes_h"][k], img_w, img_h)
        bo = _norm_box(anno_entry["boxes_o"][k], img_w, img_h)
        entries.append({
            "hoi_idx": hoi_idx,
            "verb":    verb,
            "object":  obj,
            "box_h":   bh,
            "box_o":   bo,
        })
    return entries


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_extraction_prompt(entries: list[dict]) -> str:
    return "Describe all human-object interactions visible in this image."


#


# ---------------------------------------------------------------------------
# Caption extractor
# ---------------------------------------------------------------------------

class QwenCaptionExtractor:
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(self, device: str = "cuda:0", max_new_tokens: int = 200):
        self.device = device
        self.max_new_tokens = max_new_tokens

        print(f"Loading {self.MODEL_ID} on {device} …")
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map=device,
            )
        except Exception:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
        self.model = model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            min_pixels=28 * 28,
            max_pixels=1344 * 1344,
        )
        self.tokenizer = self.processor.tokenizer
        print("Model loaded.")

    @torch.no_grad()
    def generate(self, pil_image: Image.Image, entries: list[dict]) -> str:
        """
        Generate a detailed HOI caption for one image.

        Args:
            pil_image : PIL RGB image (full scene)
            entries   : HOI entries with verb/object/boxes from build_hoi_entries()

        Returns:
            Caption string with one bracketed line per interaction.
            Returns empty string on failure.
        """
        if not entries:
            return "no interaction"

        user_text = build_extraction_prompt(entries)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image,
                        "resized_height": 448,
                        "resized_width":  448,
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[chat_text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,           # greedy — reproducible
            repetition_penalty=1.1,   # mild penalty to avoid looping
        )

        # Decode only the newly generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        new_ids    = out_ids[0, prompt_len:]
        result = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        print(f"[DEBUG] Generated caption: {result}", flush=True)
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-extract detailed HOI captions with the base Qwen2.5-VL model"
    )
    p.add_argument(
        "--data-root", required=True,
        help="Root directory of HICO-DET (must contain instances_<partition>.json "
             "and hico_20160224_det/images/<partition>/)"
    )
    p.add_argument(
        "--partition", default="train2015",
        choices=["train2015", "test2015"],
    )
    p.add_argument(
        "--output", required=True,
        help="Path for the output JSON file"
    )
    p.add_argument("--device",         default="cuda:0")
    p.add_argument("--max-new-tokens", default=200, type=int)
    p.add_argument(
        "--start", default=0,    type=int,
        help="First dataset index to process (inclusive). For multi-GPU sharding."
    )
    p.add_argument(
        "--end",   default=-1,   type=int,
        help="Last dataset index to process (exclusive). -1 means process all."
    )
    p.add_argument(
        "--save-every", default=100, type=int,
        help="Save incrementally every N images (crash-safe)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Load existing results (resume support)
    results: dict[str, str] = {}
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            results = json.load(f)
        print(f"Resuming — {len(results)} captions already in {args.output}")

    # Load annotation JSON directly (avoids pocket/DataFactory dependencies)
    anno = load_annotation(args.data_root, args.partition)
    filenames   = anno["filenames"]   # list[str]
    annotations = anno["annotation"]  # list[dict]  — parallel to filenames
    sizes       = anno["size"]        # list[[w, h]] or [[h, w]] — check below
    empty_set   = set(anno["empty"])  # image indices with no annotations

    image_dir = os.path.join(
        args.data_root, "hico_20160224_det", "images", args.partition
    )

    # Build list of valid (non-empty) indices in [start, end)
    n_total = len(filenames)
    end_idx = args.end if args.end > 0 else n_total
    valid_indices = [
        i for i in range(args.start, min(end_idx, n_total))
        if i not in empty_set
    ]
    print(f"Processing {len(valid_indices)} images (indices {args.start}–{end_idx})")

    extractor = QwenCaptionExtractor(
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    for count, i in enumerate(tqdm(valid_indices, desc="Extracting captions")):
        filename = filenames[i]
        if filename in results:
            continue  # already done (resume)

        # Image size from annotation — stored as [width, height]
        img_w, img_h = float(sizes[i][0]), float(sizes[i][1])

        # Build per-HOI entries with boxes
        anno_entry = annotations[i]
        entries    = build_hoi_entries(anno_entry, img_w, img_h)

        # Load PIL image
        img_path = os.path.join(image_dir, filename)
        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open {img_path}: {e}")
            results[filename] = ""
            continue

        # Generate caption
        try:
            caption = extractor.generate(pil_image, entries)
        except Exception as e:
            print(f"[WARN] Generation failed for {filename}: {e}")
            caption = ""

        results[filename] = caption

        # Incremental save
        if (count + 1) % args.save_every == 0:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_nonempty = sum(1 for v in results.values() if v)
    print(f"\nDone. {len(results)} entries saved ({n_nonempty} non-empty) → {args.output}")


if __name__ == "__main__":
    main()
