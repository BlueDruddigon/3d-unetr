import argparse
import time
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from monai.data import DataLoader
from monai.data.utils import decollate_batch
from monai.metrics.meandice import DiceMetric
from monai.transforms import AsDiscrete
from torch import nn as nn
from torch import optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from optimizers.early_stopping import EarlyStopping
from utils.dist import dist_all_gather
from utils.misc import AverageMeter, save_checkpoint


def train_one_epoch(
  model: nn.Module,
  criterion: nn.Module,
  scaler: GradScaler,
  loader: DataLoader,
  optimizer: optim.Optimizer,
  epoch: int,
  args: argparse.Namespace,
) -> float:
    # set train mode for model and loss_fn
    model.train()
    criterion.train()
    
    # status bar
    args.print_freq = len(loader) // 10
    pbar = tqdm(enumerate(loader), total=len(loader), miniters=args.print_freq)
    
    # metrics logger
    run_loss = AverageMeter()
    batch_timer = AverageMeter()
    
    end = time.time()
    for idx, batch_data in pbar:
        # decode the loader's data
        if isinstance(batch_data, list):
            images, labels = batch_data
        else:
            images, labels = batch_data['image'], batch_data['label']
        
        # set device for inputs and targets
        images, labels = images.to(args.device), labels.to(args.device)
        
        with torch.autocast(device_type=args.device.type, enabled=args.amp):
            logits: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(logits, labels)
        
        # Back-propagation
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # gather loss values and update metric logger
        if args.distributed:
            loss_list = dist_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            n_samples = args.batch_size * args.world_size
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=n_samples)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        batch_timer.update(time.time() - end)
        end = time.time()
        
        if args.rank == 0:
            # update pbar's status on master process
            s = f'Epoch [{epoch}/{args.max_epochs}][{idx + 1}/{len(loader)}] ' \
                f'Time/b: {batch_timer.val:.2f}s ({batch_timer.avg:.2f}s) ' \
                f'Loss/b: {run_loss.val:.4f} ({run_loss.avg:.4f})'
            pbar.set_description(s)
    
    return run_loss.avg


@torch.no_grad()
def validate_epoch(
  model: nn.Module,
  criterion: nn.Module,
  loader: DataLoader,
  epoch: int,
  acc_func: nn.Module,
  args: argparse.Namespace,
  post_label: Optional[Callable] = None,
  post_pred: Optional[Callable] = None,
) -> float:
    # set evaluate mode for model and loss_fn
    model.eval()
    criterion.eval()
    
    assert post_pred is not None
    assert post_label is not None
    
    # status bar
    args.print_freq = len(loader) // 10
    pbar = tqdm(enumerate(loader), total=len(loader), miniters=args.print_freq)
    
    valid_acc = AverageMeter()
    batch_timer = AverageMeter()
    
    end = time.time()
    for idx, batch_data in pbar:
        # decode the loader's data
        if isinstance(batch_data, list):
            images, labels = batch_data
        else:
            images, labels = batch_data['image'], batch_data['label']
        
        images, labels = images.to(args.device), labels.to(args.device)
        
        with torch.autocast(device_type=args.device.type, enabled=args.amp):
            logits: torch.Tensor = model(images)
        
        if not logits.is_cuda:  # make both `targets` and `logits` in same device
            labels = labels.cpu()
        
        post_labels = [post_label(t) for t in decollate_batch(labels)]
        post_outputs = [post_pred(t) for t in decollate_batch(logits)]
        
        acc = acc_func(y_pred=post_outputs, y=post_labels)
        acc = acc.to(args.device)
        
        # gather valid metrics and update metric loggers
        if args.distributed:
            acc_list = dist_all_gather([acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            n_samples = args.batch_size * args.world_size
            valid_acc.update(np.mean([np.nanmean(l) for l in acc_list]), n=n_samples)
        else:
            acc_list = acc.detach().cpu().numpy()
            valid_acc.update(np.mean([np.nanmean(l) for l in acc_list]), n=args.batch_size)
        batch_timer.update(time.time() - end)
        end = time.time()
        
        if args.rank == 0:
            # update pbar's status on master process
            s = f'Validation [{epoch}/{args.max_epochs}][{idx + 1}/{len(loader)}] ' \
                f'Time/b: {batch_timer.val:.2f}s ({batch_timer.avg:.2f}s) ' \
                f'Accuracy/b: {valid_acc.val:.4f} ({valid_acc.avg:.4f})'
            pbar.set_description(s)
    
    return valid_acc.avg


def run_training(
  model: Union[nn.Module, DDP],
  criterion: nn.Module,
  optimizer: optim.Optimizer,
  train_loader: DataLoader,
  valid_loader: DataLoader,
  args: argparse.Namespace,
  scheduler: Optional[LRScheduler] = None,
  callbacks: Optional[EarlyStopping] = None,
  writer: Optional[SummaryWriter] = None,
):
    scaler = None
    if args.amp:  # Gradient Scaling if using Automatic Mixed Precision
        scaler = GradScaler()
    
    # discrete post-processing
    post_label = AsDiscrete(to_onehot=args.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)
    
    # Accuracy Metrics
    acc_func = DiceMetric(include_background=True, reduction='mean', get_not_nans=True)
    
    # training
    best_valid_acc = 0.
    for epoch in range(args.start_epoch, args.max_epochs):
        if args.distributed:  # update dataloader sampler's epoch and synchronize between all processes
            train_loader.sampler.set_epoch(epoch)
            dist.barrier()
        # train in current epoch
        train_loss = train_one_epoch(
          model, criterion, scaler=scaler, loader=train_loader, optimizer=optimizer, epoch=epoch, args=args
        )
        if args.rank == 0 and writer is not None:  # tensorboard log for `train_loss` if available and in master process
            writer.add_scalar('train_loss', train_loss, epoch)
        
        # Validation
        if (epoch+1) % args.eval_freq == 0:
            if args.distributed:  # wait for synchronization
                dist.barrier()
            
            # compute the validation metric
            valid_avg_acc = validate_epoch(
              model,
              criterion,
              valid_loader,
              epoch=epoch,
              args=args,
              acc_func=acc_func,
              post_pred=post_pred,
              post_label=post_label
            )
            
            print(f'Final Validation Acc: {valid_avg_acc:.6f}')
            if writer is not None:  # tensorboard logs if available
                writer.add_scalar('valid_acc', valid_avg_acc, epoch)
            
            # check current `best_valid_acc` for improvement
            update_best_valid = False  # the flag to know if the weight's saved when update `best_valid_acc`
            if valid_avg_acc > best_valid_acc:
                print(f'New best_valid_acc: ({best_valid_acc:.6f} -> {valid_avg_acc:.6f})')
                best_valid_acc = valid_avg_acc
                update_best_valid = True
                
                # save checkpoint when best_valid_acc is updated
                save_checkpoint(
                  model,
                  epoch,
                  args,
                  filename='model_best.pth',
                  best_acc=best_valid_acc,
                  optimizer=optimizer,
                  scheduler=scheduler
                )
            
            if callbacks is not None and callbacks.step(valid_avg_acc):  # check if the training must early stop
                print(
                  f'Early Stopping at epoch {epoch}, '
                  f'current valid_acc: {valid_avg_acc:.4f}, best_valid_acc: {best_valid_acc:.4f}'
                )
                save_checkpoint(
                  model,
                  epoch,
                  args,
                  filename='model_last.pth',
                  best_acc=best_valid_acc,
                  optimizer=optimizer,
                  scheduler=scheduler
                )
                break
            
            if (epoch+1) % args.save_freq == 0 and not update_best_valid:  # save checkpoint frequently
                save_checkpoint(model, epoch, args, best_acc=best_valid_acc, optimizer=optimizer, scheduler=scheduler)
        
        if scheduler is not None:  # Update LRScheduler's state if available
            scheduler.step()
    
    print(f'Training finished! Best accuracy: {best_valid_acc}')  # end training
    
    return best_valid_acc
