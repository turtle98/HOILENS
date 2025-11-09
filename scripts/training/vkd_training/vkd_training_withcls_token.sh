#NCCL_SOCKET_IFNAME=eno2

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=1
WANDB__SERVICE_WAIT=300

CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
         main_vkd_training.py --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /hub_data1/taehoonsong/LAIN/checkpoints/CVPR/VKD_training_nores_noln/wCLS_token \
         --dataset hicodet --num_classes 117 --num-workers 4 \
         --epochs 10 --lr-head 1e-4 --lr-drop 11 \
         --batch-size 4 \
         --print-interval 100 --cls_token #--eval --resume /hub_data1/taehoonsong/LAIN/checkpoints/UV/ckpt_55550_25.pt
