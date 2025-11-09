"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import os
import torch
import random
import warnings
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
#import wandb
from thop import profile
from models.ours_vkd import build_detector
from utils.args import get_args

from engine_vkd import CustomisedDLE
from datasets import DataFactory, custom_collate
from utils.hico_text_label import hico_unseen_index

warnings.filterwarnings("ignore")

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj



def _get_model_analysis_input(data_loader):
    for images, targets in data_loader:
        return images, targets

def main(rank, args):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
   # torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(rank)

    
    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
   # import pdb; pdb.set_trace()
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16'
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'


    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root,
                           clip_model_name=args.clip_model_name, zero_shot=args.zs, zs_type=args.zs_type,
                           num_classes=args.num_classes, args=args)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root,
                          clip_model_name=args.clip_model_name, args=args)


    train_sampler = DistributedSampler(trainset, num_replicas=args.world_size, rank=rank)
    test_sampler = DistributedSampler(testset, shuffle=False, drop_last=False)


    train_loader = DataLoader(
    dataset=trainset,
    collate_fn=custom_collate, batch_size=args.batch_size,
    num_workers=args.num_workers, pin_memory=False, drop_last=True,
    sampler=train_sampler, shuffle=(train_sampler is None)
    )

    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
        sampler=test_sampler, shuffle=(test_sampler is None)
    )


    VKD_backbone = build_detector(args,rank = rank)


    if os.path.exists(args.resume):
        print(f"===>>> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        VKD_backbone.load_state_dict(checkpoint['model_state_dict'],strict=True)
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")

    engine = CustomisedDLE(
        VKD_backbone, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
        test_loader=test_loader,
        args=args
    )

    param_dicts = [
        { ## others
            "params": [p for n, p in VKD_backbone.named_parameters()
                    if p.requires_grad and 'model' not in n],
            "lr": args.lr_head,
        },
    ]
    n_parameters = sum(p.numel() for p in VKD_backbone.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_head,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)
    
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch=checkpoint['epoch']
        iteration = checkpoint['iteration']
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler, epoch=epoch,iteration=iteration, scaler=scaler)
    else:
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    import json
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()

    engine(args.epochs)

if __name__ == '__main__':
    args = get_args()
    print(args)

    if args.debug:
        os.environ['WANDB_MODE'] ='disabled'
        os.environ["MASTER_PORT"] = args.port
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    print('WORLD_SIZE ' + str(os.environ.get("WORLD_SIZE",1)))


    os.environ["MASTER_ADDR"] = "localhost"
    local_rank = int(os.environ.get("LOCAL_RANK",0))
    args.local_rank = local_rank
    # if local_rank == 0:
    #     wandb.init(project='LAIN', name=args.output_dir)

    args.world_size = int(os.environ.get("WORLD_SIZE",1))
    #import pdb; pdb.set_trace()
    main(local_rank,args)
     # or torch.cuda.set_device(args.gpu_id) if you define that
     