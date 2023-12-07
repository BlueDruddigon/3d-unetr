import argparse
import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LRScheduler

__all__ = ['seed_everthing', 'save_checkpoint', 'AverageMeter', 'PostProcessing']


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
        filename = os.path.join(args.save_dir, f'weights-{epoch}-{best_acc:.2f}.pth')
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
        """ Initial state """
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


class PostProcessing:
    def __init__(
      self,
      argmax: bool = False,
      to_onehot: Optional[int] = None,
      threshold: Optional[float] = None,
      round: Optional[bool] = None
    ) -> None:
        """Post-Processing Wrapper

        :param argmax (bool): Whether applying `argmax` or not
        :param to_onehot (int | None): If not None, apply One-hot Encoding to the inputs with num_classes=to_onehot.
        :param threshold (float | None): If not None, apply thresholding the inputs with threshold value
        :param round (bool | None): If not None, apply `torch.round`.
        :raises ValueError: to_onehot=True/False is deprecated
        """
        super().__init__()
        
        self.argmax = argmax
        
        if isinstance(to_onehot, bool):
            raise ValueError('`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.')
        self.to_onehot = to_onehot
        self.threshold = threshold
        self.round = round
    
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: the inputs tensor to be transformed
        :return: transformed version of inputs tensor
        """
        if self.argmax:
            inputs = inputs.argmax(dim=0, keepdim=True)
        
        if self.to_onehot is not None:
            inputs = F.one_hot(inputs.long(), num_classes=self.to_onehot).permute(0, 4, 1, 2, 3).squeeze(0)
        
        if self.threshold is not None:
            inputs = inputs >= self.threshold
        
        if self.round is not None:
            inputs = torch.round(inputs)
        
        return inputs
