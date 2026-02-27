#!/usr/bin/env bash
# Training: Qwen2.5-VL-3B  —  Rare First / Unseen Composition (RF-UC)

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=1
WANDB__SERVICE_WAIT=300

CUDA_VISIBLE_DEVICES=0 torchrun \
    --rdzv_id $id --rdzv_backend=c10d \
    --nproc_per_node=$gpu_num \
    --rdzv_endpoint=127.0.0.1:$port \
    main_qwen_training.py \
        --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth \
        --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt \
        --output-dir checkpoints/qwen3b/RF-UC \
        --dataset hicodet \
        --zs --zs_type rare_first \
        --num_classes 117 \
        --num-workers 4 \
        --epochs 25 \
        --lr-head 1e-3 \
        --lr-lora 1e-4 \
        --lr-drop 20 \
        --adapt_dim 128 \
        --batch-size 2 \
        --print-interval 100 \
        --linear_shortcut
