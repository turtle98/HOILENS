#!/usr/bin/env bash
# Training: Qwen2.5-VL-3B  —  Unseen Verb (UV) zero-shot split
export TOKENIZERS_PARALLELISM=false
let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=1
WANDB__SERVICE_WAIT=300

CUDA_VISIBLE_DEVICES=2 torchrun \
    --rdzv_id $id --rdzv_backend=c10d \
    --nproc_per_node=$gpu_num \
    --rdzv_endpoint=127.0.0.1:$port \
    train_hoi_sft.py \
        --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth \
        --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt \
        --output-dir checkpoints/hoi_sft_withhofusion_gt \
        --dataset hicodet \
        --data-root ./hicodet \
        --zs --zs_type unseen_verb \
        --num_classes 117 \
        --num-workers 4 \
        --epochs 25 \
        --lr-drop 20 \
        --batch-size 2 \
        --lr-head 1e-4 \
        --lr-lora 1e-4 \
        --sft-loss-weight 0.5 \
        --lora-rank 8 \
        --per_class_ap 
