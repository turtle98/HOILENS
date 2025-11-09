#NCCL_SOCKET_IFNAME=eno2

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=1
WANDB__SERVICE_WAIT=300

CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
         main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /hub_data1/taehoonsong/LAIN/checkpoints/CVPR/orig_llava/no_text_tokens/1108/NRF/30/epoch15_lr1e3_drop10_scale0_nosigmoid_addzse_backtoours_objnorm_batch8_2gpu \
         --dataset hicodet --zs --zs_type non_rare_first --num_classes 117 --num-workers 4 \
         --epochs 15 --lr-head 1e-3 --lr-drop 10 \
         --adapt_dim 128 --batch-size 8 \
         --print-interval 100 --layer 30 --zse --resume /hub_data1/taehoonsong/LAIN/checkpoints/CVPR/orig_llava/no_text_tokens/NRF/30/epoch15_lr1e3_drop10_nosigmoid_backtoours_objnorm_batch8_2gpu_1106/ckpt_01039_01.pt
