import random

import numpy as np
import torch

__all__ = ['seed_everthing', 'AverageMeter']


def seed_everthing(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, value, n=1) -> None:
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)
