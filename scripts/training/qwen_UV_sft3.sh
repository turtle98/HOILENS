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
        --output-dir checkpoints/NRF/non_gt/topk64_notrainlmhead_448_lora_attnonly_interlossinprior_withcaption \
        --dataset hicodet \
        --data-root ./hicodet \
        --zs --zs_type non_rare_first \
        --num_classes 117 \
        --num-workers 4 \
        --epochs 30 \
        --stage1-epochs 0 \
        --lr-drop 10 \
        --lr-drop-stage2 5 \
        --batch-size 4 \
        --lr-head 1e-4 \
        --lr-lora 1e-4 \
        --sft-loss-weight 1.0 \
        --lora-rank 8 \
        --per_class_ap \
        --no-gen-loss 
        #--eval \
        #--resume /home/taehoonsong/HOILENS/checkpoints/NRF/non_gt/top64_notrainlmhead_448_lora_attnonly_interlossinprior_nogenloss_sumform/ckpt_01039_01.pt

        #--resume /home/taehoonsong/HOILENS/checkpoints/NRF/non_gt/hoi_sft_withhofusion_withhospatialfusion_additionhandofeats_nocrossattn_genloss_noattnloss_bceloss_joint_256_loranewmodules_rank8_1e4_learnabletemp5_cosineanneal/ckpt_01039_01.pt \
        #--eval
        #--resume /home/taehoonsong/HOILENS/checkpoints/NRF/hoi_sft_withhofusion_withhospatialfusion_additionhandofeats_nocrossattn_genloss_noattnloss_bceloss_joint_256_loraattnonly_rank64_1e4_sftweight1_maxverbs/ckpt_05195_05.pt \
    #