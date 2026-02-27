
import os
import json
import datetime
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from models.ours_qwen_new_old import build_detector  # [QWEN]
from utils.args import get_args
from engine import CustomisedDLE
from datasets import DataFactory, custom_collate
from utils.hico_text_label import hico_unseen_index
from utils.hico_list import hico_verb_object_list

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(rank, args):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=60),  # Qwen loading can be slow
    )

    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.set_device(rank)

    # Resolve CLIP model name (used by DataFactory even though we don't use CLIP
    # as backbone; DataFactory still needs it to build image transforms)
    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16'
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'

    # -----------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------
    trainset = DataFactory(
        name=args.dataset,
        partition=args.partitions[0],
        data_root=args.data_root,
        clip_model_name=args.clip_model_name,
        zero_shot=args.zs,
        zs_type=args.zs_type,
        num_classes=args.num_classes,
        args=args,
    )
    if args.few_shot:
        testset = DataFactory(
            name=args.dataset,
            partition=args.partitions[0],
            data_root=args.data_root,
            clip_model_name=args.clip_model_name,
            args=args,
        )
    else:
        testset = DataFactory(
            name=args.dataset,
            partition=args.partitions[1],
            data_root=args.data_root,
            clip_model_name=args.clip_model_name,
            args=args,
        )

    train_sampler = DistributedSampler(
        trainset, num_replicas=args.world_size, rank=rank
    )
    test_sampler = DistributedSampler(testset, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        sampler=test_sampler,
        shuffle=(test_sampler is None),
    )

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    args.human_idx = 0
    object_n_verb_to_interaction = trainset.dataset.object_n_verb_to_interaction
    object_to_target             = trainset.dataset.object_class_to_target_class

    print(f"[INFO] num_classes = {args.num_classes}")

    model = build_detector(
        args,
        object_to_target,
        object_n_verb_to_interaction=object_n_verb_to_interaction,
        clip_model_path=args.clip_dir_vit,
        rank=rank,
    )

    # After build, swap object_class_to_target_class for eval set
    if args.dataset == 'hicodet' and args.eval:
        model.object_class_to_target_class = testset.dataset.object_class_to_target_class

    # -----------------------------------------------------------------------
    # Checkpoint resume
    # -----------------------------------------------------------------------
    checkpoint = None
    if os.path.exists(args.resume):
        print(f"===>>> Rank {rank}: resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print(f"=> Rank {rank}: starting from scratch")

    # -----------------------------------------------------------------------
    # Engine
    # -----------------------------------------------------------------------
    engine = CustomisedDLE(
        model,
        train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
        test_loader=test_loader,
        args=args,
    )

    # -----------------------------------------------------------------------
    # Cache / eval-only paths
    # -----------------------------------------------------------------------
    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    if args.eval:
        model.eval()
        if args.dataset == 'vcoco':
            import eval_vcoco
            ret = engine.cache_vcoco(test_loader)
            vsrl_annot_file = 'vcoco/data/vcoco/vcoco_test.json'
            coco_file       = 'vcoco/data/instances_vcoco_all_2014.json'
            split_file      = 'vcoco/data/splits/vcoco_test.ids'
            vcocoeval       = eval_vcoco.VCOCOeval(vsrl_annot_file, coco_file, split_file)
            ap              = vcocoeval._do_eval(ret, ovr_thresh=0.5)
            print(ap)
            return

        ap       = engine.test_hico(test_loader, args)
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare     = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        print(
            f"The mAP is {ap.mean()*100:.2f},"
            f" rare: {ap[rare].mean()*100:.2f},"
            f" none-rare: {ap[non_rare].mean()*100:.2f},"
        )

        log_lines = [
            f"The mAP is {ap.mean()*100:.2f}, "
            f"rare: {ap[rare].mean()*100:.2f}, "
            f"non-rare: {ap[non_rare].mean()*100:.2f}"
        ]

        if args.zs:
            zs_hoi_idx = hico_unseen_index[args.zs_type]
            print(f">>> zero-shot setting ({args.zs_type})")
            ap_unseen = torch.as_tensor(
                [v for i, v in enumerate(ap) if i in zs_hoi_idx]
            ).mean()
            ap_seen = torch.as_tensor(
                [v for i, v in enumerate(ap) if i not in zs_hoi_idx]
            ).mean()
            print(
                f"full mAP: {ap.mean()*100:.2f}",
                f"unseen: {ap_unseen*100:.2f}",
                f"seen: {ap_seen*100:.2f}",
            )
            log_lines.append(
                f"full mAP: {ap.mean()*100:.2f}, "
                f"unseen: {ap_unseen*100:.2f}, "
                f"seen: {ap_seen*100:.2f}"
            )

        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, "eval_log.txt")
        with open(log_path, "a") as f:
            f.write("Evaluation Results (Qwen2.5-VL-3B):\n")
            for line in log_lines:
                f.write(line + "\n")

        if args.per_class_ap:
            per_class_log = os.path.join(args.output_dir, "per_class_ap_eval.txt")
            with open(per_class_log, "w") as f:
                f.write("Per-class AP — Qwen2.5-VL-3B\n")
                f.write(f"{'idx':>4}  {'verb':<20}  {'object':<20}  {'AP':>8}\n")
                f.write("-" * 58 + "\n")
                for i, (verb, obj) in enumerate(hico_verb_object_list):
                    f.write(
                        f"{i:>4}  {verb:<20}  {obj:<20}  "
                        f"{ap[i].item()*100:>8.2f}\n"
                    )
            print(f"Per-class AP saved to {per_class_log}")
        return

    # -----------------------------------------------------------------------
    # Training setup
    # -----------------------------------------------------------------------
    # Freeze DETR
    for p in model.detector.parameters():
        p.requires_grad = False

    # Freeze the Qwen backbone (model weights inside clip_head["model"]);
    # only LoRA adapters and projection heads are trained.
    for name, p in model.clip_head["model"].named_parameters():
        p.requires_grad = False

    param_dicts = [
        {   # LoRA adapters — separate (usually lower) lr
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'lora_' in n
            ],
            "lr": args.lr_lora,
        },
        {   # Everything else that is trainable (query projections, spatial head, …)
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'lora_' not in n
            ],
            "lr": args.lr_head,
        },
    ]

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] trainable params: {n_trainable:,}")

    optim        = torch.optim.AdamW(param_dicts, lr=args.lr_head, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)

    if checkpoint is not None:
        optim.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        engine.update_state_key(
            optimizer=optim,
            lr_scheduler=lr_scheduler,
            epoch=checkpoint['epoch'],
            iteration=checkpoint['iteration'],
            scaler=scaler,
        )
    else:
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    engine(args.epochs)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = get_args()
    print(args)

    if args.debug:
        os.environ['WANDB_MODE'] = 'disabled'
        os.environ["MASTER_PORT"] = args.port
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    print(f"WORLD_SIZE {os.environ.get('WORLD_SIZE', 1)}")

    os.environ["MASTER_ADDR"] = "localhost"
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    args.local_rank  = local_rank
    args.world_size  = int(os.environ.get("WORLD_SIZE", 1))

    main(local_rank, args)
