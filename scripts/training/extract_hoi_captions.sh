#!/usr/bin/env bash
# Pre-extract detailed HOI captions using the base Qwen2.5-VL-3B model.
#
# Runs two shards in parallel (one per GPU) then merges into a single JSON.
# Adjust CUDA_VISIBLE_DEVICES / --start / --end to match your GPU setup.
#
# Usage:
#   bash scripts/training/extract_hoi_captions.sh


DATA_ROOT=./hicodet
PARTITION=train2015
OUTPUT_DIR=./outputs/captions
MAX_NEW_TOKENS=200
SAVE_EVERY=100

mkdir -p "$OUTPUT_DIR"

N_IMAGES=38118   # total filenames in instances_train2015.json
MID=$(( N_IMAGES / 2 ))   # ~19059

CUDA_VISIBLE_DEVICES=0 python scripts/extract_hoi_captions.py \
    --data-root   "$DATA_ROOT" \
    --partition   "$PARTITION" \
    --output      "$OUTPUT_DIR/captions_shard0.json" \
    --device      cuda:0 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --start       0 \
    --end         "$MID" \
    --save-every  "$SAVE_EVERY" &