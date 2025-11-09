#NCCL_SOCKET_IFNAME=eno2

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
WANDB__SERVICE_WAIT=300
gpu_num=1

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
            main_clip.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir checkpoints/zsCLIP \
         --dataset hicodet --zs --zs_type unseen_verb --num_classes 117 --num-workers 4 \
         --epochs 20  --use_exp --N_CTX 0 --batch-size 4 --use_prompt --adapt_dim 128 \
         --print-interval 100 --fps