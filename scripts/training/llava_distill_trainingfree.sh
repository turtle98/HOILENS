#NCCL_SOCKET_IFNAME=eno2

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=1
WANDB__SERVICE_WAIT=300
export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=4 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
         main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /home/taehoon/LAIN/checkpoints/LLavazeroshot_origfeat \
         --dataset hicodet --zs --zs_type unseen_object --num_classes 117 --num-workers 4 \
         --epochs 20 --use_hotoken --use_prompt --use_exp --CSC --N_CTX 36 \
         --use_insadapter --adapt_dim 128 --use_prior --adapter_alpha 1. --batch-size 4 \
         --print-interval 100 --eval --linear_shortcut # --resume /home/taehoon/LAIN/check

