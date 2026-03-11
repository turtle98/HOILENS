#!/usr/bin/env bash
# Training: Qwen2.5-VL-3B  —  Unseen Verb (UV) zero-shot split
export TOKENIZERS_PARALLELISM=false
let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=4
WANDB__SERVICE_WAIT=300

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --rdzv_id $id --rdzv_backend=c10d \
    --nproc_per_node=$gpu_num \
    --rdzv_endpoint=127.0.0.1:$port \
    train_hoi_sft.py \
        --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth \
        --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt \
        --output-dir checkpoints/NRF/hoi_sft_withhofusion_withhospatialfusion_withattn_with6verbforms_captionloss_leaveoneout_448_256_withinterloss10onhoqueries_fixed_twostage \
        --dataset hicodet \
        --data-root ./hicodet \
        --zs --zs_type non_rare_first \
        --num_classes 117 \
        --num-workers 4 \
        --epochs 55 \
        --stage1-epochs 30 \
        --lr-drop 100 \
        --lr-drop-stage2 10 \
        --batch-size 4 \
        --lr-head 1e-4 \
        --lr-lora 1e-4 \
        --sft-loss-weight 0.5 \
        --lora-rank 8 \
        --per_class_ap \
        --use-img-cross-attn
        #--eval \
        #--resume /home/taehoonsong/HOILENS/checkpoints/NRF/hoi_sft_withhofusion_with6verbforms_captionloss_leaveoneout_448_256_withinterloss10onhoqueries_fixed_twostage/ckpt_02079_01.pt
        #--eval \
        #--resume /home/taehoonsong/HOILENS/checkpoints/NRF/hoi_sft_withhofusion_with6verbforms_captionloss_leaveoneout_512_1e3_alpha2/ckpt_10395_05.pt