#!/usr/bin/env bash
# Evaluation: Qwen2.5-VL-3B on HICO-DET
#
# Usage:
#   ZS_TYPE=unseen_object  CKPT=checkpoints/qwen3b/UO/ckpt_XXXXX_25.pt  bash scripts/eval/qwen_eval.sh
#   ZS_TYPE=unseen_verb    CKPT=checkpoints/qwen3b/UV/ckpt_XXXXX_25.pt  bash scripts/eval/qwen_eval.sh
#   ZS_TYPE=non_rare_first CKPT=checkpoints/qwen3b/NF-UC/ckpt_XXXXX_25.pt bash scripts/eval/qwen_eval.sh
#   ZS_TYPE=rare_first     CKPT=checkpoints/qwen3b/RF-UC/ckpt_XXXXX_25.pt bash scripts/eval/qwen_eval.sh
#
# For fully-supervised (no zero-shot), omit --zs and ZS_TYPE:
#   CKPT=checkpoints/qwen3b/full/ckpt_XXXXX_25.pt ZS_TYPE="" bash scripts/eval/qwen_eval.sh
#
# Environment variables (all optional — defaults shown below):
#   CUDA_ID    GPU index(es)           default: 0
#   CKPT       path to checkpoint      default: "" (runs from scratch, sanity check only)
#   ZS_TYPE    zero-shot split name    default: unseen_object
#   OUT_DIR    output dir              default: same folder as checkpoint, or checkpoints/qwen3b/eval

set -e

CUDA_ID="${CUDA_ID:-0}"
CKPT="${CKPT:-}"
ZS_TYPE="${ZS_TYPE:-unseen_object}"
OUT_DIR="${OUT_DIR:-}"

if [ -n "$CKPT" ]; then
    CKPT_DIR=$(dirname "$CKPT")
    OUT_DIR="${OUT_DIR:-$CKPT_DIR}"
    RESUME_FLAG="--resume $CKPT"
else
    OUT_DIR="${OUT_DIR:-checkpoints/qwen3b/eval}"
    RESUME_FLAG=""
fi

ZS_FLAGS=""
if [ -n "$ZS_TYPE" ]; then
    ZS_FLAGS="--zs --zs_type $ZS_TYPE"
fi

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=1

echo "=== Qwen2.5-VL-3B Evaluation ==="
echo "    checkpoint : ${CKPT:-none (random init)}"
echo "    zs_type    : ${ZS_TYPE:-none (fully supervised)}"
echo "    output dir : $OUT_DIR"
echo "================================="

CUDA_VISIBLE_DEVICES=$CUDA_ID torchrun \
    --rdzv_id $id --rdzv_backend=c10d \
    --nproc_per_node=$gpu_num \
    --rdzv_endpoint=127.0.0.1:$port \
    main_qwen_training.py \
        --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth \
        --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt \
        --output-dir "$OUT_DIR" \
        --dataset hicodet \
        $ZS_FLAGS \
        --num_classes 117 \
        --num-workers 4 \
        --epochs 1 \
        --adapt_dim 128 \
        --batch-size 1 \
        --print-interval 100 \
        --linear_shortcut \
        --eval \
        --per_class_ap \
        $RESUME_FLAG
