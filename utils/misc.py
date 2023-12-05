import argparse
import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from monai.data.utils import select_cross_validation_folds
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LRScheduler

__all__ = ['seed_everthing', 'save_checkpoint', 'AverageMeter']


def save_checkpoint(
  model: Union[nn.Module, DDP],
  epoch: int,
  args: argparse.Namespace,
  best_acc: float = 0.,
  filename: str = '',
  optimizer: Optional[optim.Optimizer] = None,
  scheduler: Optional[LRScheduler] = None,
):
    """Save checkpoint utility

    :param model: Model to use, which is torch.nn.Module or DDP
    :param epoch: The current epoch
    :param args: The arguments that set.
    :param best_acc: Current best_valid_acc. Default: 0.
    :param optimizer: Optimizer to use. Default: None
    :param scheduler: LR Scheduler to use. Default: None
    """
    state_dict = model.module.state_dict() if args.distributed else model.state_dict()
    save_dict = {
      'state_dict': state_dict,
      'epoch': epoch,
      'best_valid_acc': best_acc,
    }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    if not filename:
        filename = os.path.join(args.save_dir, f'weights-{epoch}-{best_acc:.2f}.pt')
    torch.save(save_dict, filename)
    print('Saved checkpoint', filename)


def seed_everthing(seed: int) -> None:
    """
    :param seed: seed to be set in random state
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AverageMeter:
    """ Metrics Monitor """
    def __init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, value, n=1) -> None:
        """
        :param value (number): The value to be updated and calculated
        :param n (int): The number of samples to be updated
        """
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)
