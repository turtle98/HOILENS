#NCCL_SOCKET_IFNAME=eno2

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=4
WANDB__SERVICE_WAIT=300
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
         main_llava_training.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /home/taehoonsong/HOILENS/checkpoints/ECCV/train/0223/UV/ours_new_all3branch_twolayer_lora_rank16_31_withdetr_withspatialho_withattnmaskho_weightedvocabmeanforall_noklloss_epoch30_drop15_noqueryselfattnadded_objclassadded_maxverbform3 \
         --dataset hicodet --num_classes 117 --num-workers 4 \
         --epochs 30  --zs --zs_type unseen_verb --lr-head 1e-3 --lr-lora 1e-4 --lr-drop 15 \
         --adapt_dim 128 --batch-size 4 --start_idx 1 --end_idx 32 \
         --print-interval 100 --layer 30 --per_class_ap #--eval --resume /home/taehoonsong/HOILENS/checkpoints/ECCV/train/0222/UV/ours_new_all3branch_twolayer_lora_rank16_31_withdetr_withspatialho_withattnmaskho_weightedmeanforallvocab_noklloss_epoch30_drop15_noqueryselfattnadded_objclassadded/ckpt_37774_17.pt
# CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
#          main_llava_training.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /home/taehoon/HOILENS/checkpoints/ECCV/zeroshot_newsubset_block_attn_midtolate_layers_loglikelihood_gt \
#          --dataset hicodet --num_classes 117 --num-workers 4 \q
#          --epochs 10 --lr-head 1e-4 --lr-drop 5 \
#          --adapt_dim 128 --batch-size 1 --start_idx 11 --end_idx 32 \
#          --print-interval 100 --eval --few_shot --attn_mod

# CUDA_VISIBLE_DEVICES=3,5 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
#          main_llava_training.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /hub_data1/taehoonsong/LAIN/checkpoints/CVPR/orig_llava/no_text_tokens/UV/30_epoch20/hub_data1/taehoonsong/LAIN/checkpoints/CVPR/orig_llava/no_text_tokens/UV/30_epoch20_lr1e4_drop10_nosigmoid_new_1104 \
#          --dataset hicodet --zs --zs_type unseen_verb --num_classes 117 --num-workers 4 \
#          --epochs 20 --lr-head 1e-4 --lr-drop 10 \
#          --adapt_dim 128 --batch-size 8 \
#          --print-interval 100 --eval --zse --resume /hub_data1/taehoonsong/LAIN/checkpoints/CVPR/orig_llava/no_text_tokens/UV/30_epoch20_lr1e4_drop10_nosigmoid_new_1104/ckpt_11110_05.pt




# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
#          main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /home/taehoon/LAIN/checkpoints/UO/best_orighoverb_steplr_lr1e3to1e4_epoch15\
#          --dataset hicodet --zs --zs_type unseen_object --
# num_classes 117 --num-workers 4 \
#          --epochs 15 --lr-head 1e-3 --lr-drop 10 \
#          --adapt_dim 128 --batch-size 4 \
#          --print-interval 100 --linear_shortcut 

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
#          main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /home/taehoon/LAIN/checkpoints/UV/best_orighoverb_steplr_lr1e3to1e4_epoch15\
#          --dataset hicodet --zs --zs_type unseen_verb --num_classes 117 --num-workers 4 \
#          --epochs 30 --lr-head 1e-3 --lr-drop 10 \
#          --adapt_dim 128 --batch-size 4 \
#          --print-interval 100 --linear_shortcut 

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
#          main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /home/taehoon/LAIN/checkpoints/NRF/best_precomputedllavafeats_orighoverb_coslr_lr1e4_epoch5 \
#          --dataset hicodet --zs --zs_type unseen_object --num_classes 117 --num-workers 4 \
#          --epochs 5 \
#          --adapt_dim 128 --batch-size 4 \
#          --print-interval 100 --linear_shortcut 


# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
#          main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir /home/taehoon/LAIN/checkpoints/NRF/best_precomputedllavafeats_orighoverb_coslr_lr1e4_epoch5 \
#          --dataset hicodet --zs --zs_type unseen_verb --num_classes 117 --num-workers 4 \
#          --epochs 5 \
#          --adapt_dim 128 --batch-size 4 \
#          --print-interval 100 --linear_shortcut 

 #-resume /home/taehoon/LAIN/checkpoints/UO/128_justcrossattn_noalpha_30_lrdrop20/ckpt_58410_30.pt
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
#          main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir checkpoints/UV/128_justnormalplusingforho_justho \
#          --dataset hicodet --zs --zs_type unseen_verb --num_classes 117 --num-workers 4 \
#          --epochs 20 --use_hotoken --use_prompt --use_exp --CSC --N_CTX 36 \
#          --use_insadapter --adapt_dim 128 --use_prior --adapter_alpha 1. --batch-size 4 \
#          --print-interval 100 --linear_shortcut  # --resume #/home/taehoon/LAIN/checkpoints2/UO_ours_h_o_u_seperate_128_top10_decoder_all_seperate_layer_1_sub_obj_feat_add_withzeroshot_dropout/ckpt_03894_02.pt

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
#          main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir checkpoints/UV/128_noaddingoriginalembeddingatlast_sharedprojfortextandvis_nobias \
#          --dataset hicodet --zs --zs_type unseen_verb --num_classes 117 --num-workers 4 \
#          --epochs 20 --use_prompt --use_exp --CSC --N_CTX 24 --batch-size 4 --linear_shortcut \
#          --adapt_dim 256 \
#          --print-interval 100
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port 
#          main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir checkpoints/NF_UC/128_noaddingoriginalembeddingatlast_sharedprojfortextandvis_nobias \
#          --dataset hicodet --zs --zs_type non_rare_first --num_classes 117 --num-workers 4 \
#          --epochs 20 --use_prompt --use_exp --CSC --N_CTX 24 --batch-size 4 --linear_shortcut \
#          --adapt_dim 64 \
#          --print-interval 100

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
#          main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --output-dir checkpoints/NF_UC/ours_fixed_withmax \
#          --dataset hicodet --zs --zs_type non_rare_first --num_classes 117 --num-workers 4 \
#          --epochs 20 --use_prompt --use_exp --CSC --N_CTX 24 --batch-size 4 \
#          --adapt_dim 128 \
#          --print-interval 100