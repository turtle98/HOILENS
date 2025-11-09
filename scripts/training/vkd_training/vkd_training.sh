#NCCL_SOCKET_IFNAME=eno2

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=1
WANDB__SERVICE_WAIT=300

# CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
#          main_vkd_training.py --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /hub_data1/taehoonsong/LAIN/checkpoints/CVPR/VKD_training_nores_withln/no_cls_token/30 \
#          --dataset hicodet --num_classes 117 --num-workers 4 \
#          --epochs 10 --lr-head 1e-4 --lr-drop 11 \
#          --batch-size 4 \
#          --print-interval 100 --layer 30 #--eval --resume /hub_data1/taehoonsong/LAIN/checkpoints/UV/ckpt_55550_25.pt

CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
         main_vkd_training.py --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /hub_data1/taehoonsong/LAIN/checkpoints/CVPR/VKD_training_nores_withln/no_cls_token/20 \
         --dataset hicodet --num_classes 117 --num-workers 4 \
         --epochs 10 --lr-head 1e-4 --lr-drop 11 \
         --batch-size 4 \
         --print-interval 100 --layer 20

CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
         main_vkd_training.py --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /hub_data1/taehoonsong/LAIN/checkpoints/CVPR/VKD_training_nores_withln/no_cls_token/10 \
         --dataset hicodet --num_classes 117 --num-workers 4 \
         --epochs 10 --lr-head 1e-4 --lr-drop 11 \
         --batch-size 4 \
         --print-interval 100 --layer 10

CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
         main_vkd_training.py --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /hub_data1/taehoonsong/LAIN/checkpoints/CVPR/VKD_training_nores_withln/no_cls_token/40 \
         --dataset hicodet --num_classes 117 --num-workers 4 \
         --epochs 10 --lr-head 1e-4 --lr-drop 11 \
         --batch-size 4 \
         --print-interval 100 --layer 40
# CUDA_VISIBLE_DEVICES=4 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
#          main_vkd_training.py --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /hub_data1/taehoonsong/LAIN/checkpoints/CVPR/VKD_training_nores_noln/no_cls_token/20 \
#          --dataset hicodet --num_classes 117 --num-workers 4 \
#          --epochs 10 --lr-head 1e-4 --lr-drop 11 \
#          --batch-size 4 --layer 20 \
#          --print-interval 100 #--eval --resume /hub_data1/taehoonsong/LAIN/checkpoints/UV/ckpt_55550_25.pt