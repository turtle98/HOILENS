#NCCL_SOCKET_IFNAME=eno2

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=4
WANDB__SERVICE_WAIT=300

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
         main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /home/taehoon/LAIN/checkpoints2/default/ours_h_o_u_all_seperate_128_decoder_all_seperate_layer_1_sub_obj_feat_add_withzeroshotconvexcomb_dropout_noselfattn_conveccomb \
         --dataset hicodet --num_classes 117 --num-workers 4 \
         --epochs 20 --use_hotoken --use_prompt --use_exp --CSC --N_CTX 36 \
         --use_insadapter --adapt_dim 128 --use_prior --adapter_alpha 1. --batch-size 4 \
         --print-interval 100 --linear_shortcut  # --resume #/home/taehoon/LAIN/checkpoints2/UO_ours_h_o_u_seperate_128_top10_decoder_all_seperate_layer_1_sub_obj_feat_add_withzeroshot_dropout/ckpt_03894_02.pt