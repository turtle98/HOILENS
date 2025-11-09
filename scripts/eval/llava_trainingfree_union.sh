#NCCL_SOCKET_IFNAME=eno2

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=4
WANDB__SERVICE_WAIT=300
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python main_llava_cropped.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /home/taehoon/LAIN/checkpoints/LLavazeroshot_union \
         --dataset hicodet --num_classes 117 --num-workers 4 \
         --epochs 1  --adapt_dim 128 --adapter_alpha 1. --batch-size 1 \
         --print-interval 100 --linear_shortcut --layer 30 --eval #--save_dir /home/taehoon/HOICLIP/data/features_llava  # --resume /home/taehoon/LAIN/check

# CUDA_VISIBLE_DEVICES=4,5 python main_llava.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /home/taehoon/LAIN/checkpoints/LLavazeroshot_origfeat \
#          --dataset hicodet --zs --zs_type unseen_object --num_classes 117 --num-workers 4 \
#          --epochs 20 --use_hotoken --use_prompt --use_exp --CSC --N_CTX 36 \
#          --use_insadapter --adapt_dim 128 --use_prior --adapter_alpha 1. --batch-size 4 \
#          --print-interval 100 --eval --linear_shortcut # --resume /home/taehoon/LAIN/check

