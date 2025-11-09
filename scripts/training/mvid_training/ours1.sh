#NCCL_SOCKET_IFNAME=eno2

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=1
WANDB__SERVICE_WAIT=300

CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
         main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /hub_data1/taehoonsong/LAIN/checkpoints/CVPR/MVID_training/nores_noln/no_cls_token/30/UV/vkd_epoch1/VKD_30_inferencetimeaddition \
         --dataset hicodet --zs --zs_type unseen_verb --num_classes 117 --num-workers 4 \
         --epochs 20 --lr-head 1e-4 --lr-drop 10 \
         --adapt_dim 128 --batch-size 8 \
         --print-interval 100 --linear_shortcut #--eval --resume /hub_data1/taehoonsong/LAIN/checkpoints/CVPR/MVID_training/nores_noln/no_cls_token/30/UV/VKD_30/ckpt_02222_01.pt
