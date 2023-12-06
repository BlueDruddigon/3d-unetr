import argparse
import logging
import os
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter

from datasets import build_dataset
from losses.dice import DiceCELoss, DiceLoss
from models.unetr import UNETR
from optimizers.early_stopping import EarlyStopping
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.dist import setup_for_distributed

# disable warn logging
logging.disable(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Dataset's Hyperparams
    parser.add_argument('--data-root', type=str, default='', help='Path to the root directory of the Dataset')
    parser.add_argument('--in-channels', type=int, default=1, help='Number of channels in Dataset\'s Volumes')
    parser.add_argument('--num-classes', type=int, default=14, help='Number of classes')
    parser.add_argument('--roi-x', type=int, default=96, help='ROI size in x direction')
    parser.add_argument('--roi-y', type=int, default=96, help='ROI size in y direction')
    parser.add_argument('--roi-z', type=int, default=96, help='ROI size in z direction')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for DataLoader')
    parser.add_argument('--train-cache-num', type=int, default=24, help='Number of cached samples in CacheDataset')
    parser.add_argument('--valid-cache-num', type=int, default=8, help='Number of cached samples in CacheDataset')
    
    # Transform's Hyperparams
    parser.add_argument('--a-min', type=float, default=-175., help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a-max', type=float, default=250., help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b-min', type=float, default=0., help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b-max', type=float, default=1., help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space-x', type=float, default=1.5, help='Spacing in x direction')
    parser.add_argument('--space-y', type=float, default=1.5, help='Spacing in y direction')
    parser.add_argument('--space-z', type=float, default=2., help='Spacing in z direction')
    parser.add_argument('--rand-flip-prob', type=float, default=.2, help='RandFlipd aug probability')
    parser.add_argument('--rand-rotate90-prob', type=float, default=.2, help='RandRotate90d aug probability')
    parser.add_argument(
      '--rand-scale-intensity-prob', type=float, default=.1, help='RandScaleIntensityd aug probability'
    )
    parser.add_argument(
      '--rand-shift-intensity-prob', type=float, default=.1, help='RandShiftIntensityd aug probability'
    )
    
    # Feature Extractor's Hyperparams
    parser.add_argument(
      '--patch-size', type=Union[int, Sequence[int]], default=(16, 16, 16), help='Patch size for Vision Transformer'
    )
    parser.add_argument(
      '--embed-dim', type=int, default=768, help='Embedding Dimension for Vision Transformer and UNETR'
    )
    parser.add_argument('--num-layers', type=int, default=12, help='Number of Vision Transformer\'s Encoder layers')
    parser.add_argument('--num-heads', type=int, default=12, help='Number of Attention Head for Vision Transformer')
    parser.add_argument('--mlp-ratio', type=float, default=4., help='Hidden Feature Ratio for MLP')
    parser.add_argument('--qkv-bias', type=bool, default=True, help='Whether using bias for Attention Head')
    parser.add_argument(
      '--qk-scale', type=Optional[float], default=None, help='Overrides default qk scale for Attention Head'
    )
    parser.add_argument(
      '--classification', type=bool, default=False, help='Whether using Vision Transformer\'s Classification Head'
    )
    parser.add_argument('--drop-path-rate', type=float, default=.1, help='Stochastic Depth Decay Rule')
    parser.add_argument('--attn-drop', type=float, default=0., help='Attention Head Dropout Rate')
    parser.add_argument('--proj-drop', type=float, default=0., help='Attention Output Projection Dropout Rate')
    parser.add_argument('--act-layer', type=Callable, default=nn.GELU, help='Activation Layer to choose')
    parser.add_argument(
      '--post-activation',
      type=Optional[str],
      default=None,
      choices=['tanh', None],
      help='Post Activation Layer to be applied in classification head in Vision Transformer'
    )
    parser.add_argument('--save-attn', type=bool, default=False, help='Whether to store the attention map')
    
    # Model's Hyperparams
    parser.add_argument('--model-name', type=str, default='UNETR', choices=['UNETR'], help='Name of the model to use')
    parser.add_argument('--depths', type=int, default=4, help='Number of Encoder and Decoder\'s layers')
    parser.add_argument(
      '--feature-size', type=int, default=64, help='Feature Dimension for UNETR\'s Encoder and Decoder'
    )
    parser.add_argument(
      '--norm-layer', type=Callable, default=nn.BatchNorm3d, help='Normalization layer using in UNETR'
    )
    parser.add_argument('--kernel-size', type=Union[int, Sequence[int]], default=3, help='Kernel sizes using in Conv3D')
    parser.add_argument('--stride', type=Union[int, Sequence[int]], default=1, help='Strides using in Conv3D')
    parser.add_argument(
      '--upsample-kernel-size',
      type=Union[int, Sequence[int]],
      default=2,
      help='Kernel sizes and strides using for Deconv'
    )
    
    # Optimization's Hyperparams
    parser.add_argument(
      '--opt-name',
      type=str,
      default='adamw',
      choices=['sgd', 'adam', 'adamw'],
      help='Optimization Algorithm\'s name to use'
    )
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate for Optim')
    parser.add_argument('--momentum', type=float, default=0.99, help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Regularization weight decay')
    
    # LrScheduler's Hyperparams
    parser.add_argument(
      '--lr-scheduler',
      type=str,
      default='warmup_cosine',
      choices=['warmup_cosine', 'cosine_anneal'],
      help='Name of LRScheduler to use'
    )
    parser.add_argument('--warmup-epochs', type=int, default=50, help='Number of Warmup Epochs')
    
    # Loss's Hyperparams
    parser.add_argument(
      '--loss-fn', type=str, choices=['dice', 'dice_ce'], default='dice_ce', help='Loss function to use'
    )
    parser.add_argument('--lambda-dice', type=float, default=1., help='Weighted decay for Dice Loss')
    parser.add_argument('--lambda-ce', type=float, default=1., help='Weighted decay for CrossEntropy Loss')
    parser.add_argument(
      '--smooth', type=float, default=1e-5, help='Specifies the amount of smoothing when computing the loss'
    )
    parser.add_argument(
      '--sigmoid', type=bool, default=False, help='Whether to apply sigmoid act before computing the loss'
    )
    parser.add_argument(
      '--softmax', type=bool, default=True, help='Whether to apply softmax act before computing the loss'
    )
    parser.add_argument(
      '--include-background', type=bool, default=True, help='Whether to consider background as a class'
    )
    parser.add_argument(
      '--squared-pred', type=bool, default=True, help='Whether take squared prediction as denominator'
    )
    parser.add_argument(
      '--reduction', type=str, default='mean', choices=['mean', 'sum', None], help='Reduction of the loss'
    )
    
    # Training's Hyperparams
    parser.add_argument('--max-epochs', type=int, default=5000, help='Max number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of batch size')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpointing')
    parser.add_argument('--exp-dir', type=str, default='./runs', help='Experimental Directory')
    parser.add_argument('--amp', action='store_false', help='Whether using AMP or not')
    parser.add_argument('--print-freq', type=int, default=100, help='Print Frequency of tqdm')
    parser.add_argument('--eval-freq', type=int, default=5, help='Evaluate Frequency')
    parser.add_argument('--save-freq', type=int, default=5, help='Save checkpoint Frequency')
    parser.add_argument('--patience', type=int, default=5, help='Early Stopping Patience')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_false', help='Whether using distributed training')
    parser.add_argument('--dist-backend', type=str, default='nccl', help='distributed backend')
    
    return parser.parse_args()


def load_checkpoint(
  args: argparse.Namespace, model: Union[nn.Module, DDP], optimizer: optim.Optimizer,
  lr_scheduler: Optional[LRScheduler]
) -> Tuple[argparse.Namespace, Union[nn.Module, DDP], optim.Optimizer, Optional[LRScheduler]]:
    args.start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=torch.device('cpu'))
        if args.distributed:  # DDP
            model.module.load_state_dict(ckpt['state_dict'])
        else:  # nn.Module
            model.load_state_dict(ckpt['state_dict'])
        if optimizer is not None and 'optimizer' in ckpt.keys():
            optimizer.load_state_dict(ckpt['optimizer'])
        if lr_scheduler is not None and 'lr_scheduler' in ckpt.keys():
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        args.start_epoch = ckpt['epoch'] + 1
    else:
        print('Training from scratch.')
    
    return args, model, optimizer, lr_scheduler


def initialize_algorithm(
  args: argparse.Namespace
) -> Tuple[argparse.Namespace, nn.Module, nn.Module, optim.Optimizer, Optional[LRScheduler], EarlyStopping]:
    # Define model
    if args.model_name == 'UNETR':
        model = UNETR(
          args.in_channels,
          args.num_classes,
          img_size=(args.roi_x, args.roi_y, args.roi_z),
          patch_size=args.patch_size,
          embed_dim=args.embed_dim,
          feature_size=args.feature_size,
          depths=args.depths,
          mlp_ratio=args.mlp_ratio,
          num_layers=args.num_layers,
          num_heads=args.num_heads,
          qkv_bias=args.qkv_bias,
          qk_scale=args.qk_scale,
          drop_path_rate=args.drop_path_rate,
          attn_drop=args.attn_drop,
          proj_drop=args.proj_drop,
          act_layer=args.act_layer,
          norm_layer=args.norm_layer,
          kernel_size=args.kernel_size,
          stride=args.stride,
          upsample_kernel_size=args.upsample_kernel_size,
          post_activation=None if not args.classification else args.post_activation,
          save_attn=args.save_attn
        )
    else:
        raise ValueError
    
    # Loss function
    if args.loss_fn == 'dice':
        criterion = DiceLoss(
          include_background=args.include_background,
          sigmoid=args.sigmoid,
          softmax=args.softmax,
          reduction=args.reduction,
          squared_pred=args.squared_pred,
          smooth=args.smooth
        )
    elif args.loss_fn == 'dice_ce':
        criterion = DiceCELoss(
          include_background=args.include_background,
          sigmoid=args.sigmoid,
          softmax=args.softmax,
          reduction=args.reduction,
          squared_pred=args.squared_pred,
          smooth=args.smooth,
          lambda_ce=args.lambda_ce,
          lambda_dice=args.lambda_dice
        )
    else:
        raise ValueError
    
    # Optimization Algorithm
    if args.opt_name == 'sgd':
        optimizer = optim.SGD(
          model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay
        )
    elif args.opt_name == 'adam':
        optimizer = optim.Adam(
          model.parameters(), lr=args.lr, betas=(0.9, args.momentum), weight_decay=args.weight_decay
        )
    elif args.opt_name == 'adamw':
        optimizer = optim.AdamW(
          model.parameters(), lr=args.lr, betas=(0.9, args.momentum), weight_decay=args.weight_decay
        )
    else:
        raise ValueError
    
    # Learning rate scheduler
    if args.lr_scheduler == 'warmup_cosine':
        lr_scheduler = LinearWarmupCosineAnnealingLR(
          optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lr_scheduler == 'cosine_anneal':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    else:
        lr_scheduler = None
    
    early_stop_callback = EarlyStopping(mode='min', patience=args.patience)
    
    return args, model, criterion, optimizer, lr_scheduler, early_stop_callback


def main(args: argparse.Namespace):
    # init distributed training
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend)
        args.rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        setup_for_distributed(args.rank == 0)
    else:
        args.rank = 0
        args.world_size = 1
    
    # prepare device with current rank
    torch.cuda.set_device(args.rank)
    args.device = torch.device(f'cuda:{args.rank}')
    
    # init model, loss_fn, optimizer, lr_scheduler
    args, model, criterion, optimizer, lr_scheduler, early_stop_callback = initialize_algorithm(args)
    
    # move to CUDA
    model = model.to(args.device)
    criterion = criterion.to(args.device)
    
    # wrap model with DDP if distributed training is available
    if args.distributed:
        model = DDP(model, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)
    
    # load from checkpointing if available
    args, model, optimizer, lr_scheduler = load_checkpoint(args, model, optimizer, lr_scheduler)
    
    # weights and logs
    args.exp_dir = os.path.join(args.exp_dir, args.model_name)
    args.save_dir = os.path.join(args.exp_dir, 'weights')
    args.log_dir = os.path.join(args.exp_dir, 'logs')
    
    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Tensorboard logger
    writer = SummaryWriter(args.log_dir)
    
    # get loader from dataset and arguments
    train_loader, valid_loader = build_dataset(args)
    
    # training process
    acc = run_training(
      model,
      criterion,
      optimizer,
      train_loader=train_loader,
      valid_loader=valid_loader,
      args=args,
      scheduler=lr_scheduler,
      writer=writer,
      callbacks=early_stop_callback
    )
    return acc


if __name__ == '__main__':
    args = parse_args()
    main(args)
