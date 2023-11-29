from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(
      self,
      include_background: bool = True,
      sigmoid: bool = False,
      softmax: bool = True,
      reduction: str = 'mean',
      squared_pred: bool = False,
      smooth: float = 1e-7,
    ) -> None:
        super(DiceLoss, self).__init__()
        
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.reduction = reduction
        self.smooth = smooth
        self.squared_pred = squared_pred
        self.include_background = include_background
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            inputs = F.sigmoid(inputs)
        
        num_classes = inputs.shape[1]
        if self.softmax:
            assert num_classes != 1, 'single channel prediction, `softmax=True` is ignored'
            inputs = F.softmax(inputs, dim=1)
        
        if num_classes != 1:
            if targets.shape[1] == 1 or targets.ndim == 4:
                targets = targets.squeeze(1)
            targets = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2)
        
        if not self.include_background:
            assert num_classes != 1, 'single channel prediction, `include_background=False` ignored.'
            inputs = inputs[:, 1:]
            targets = targets[:, 1:]
        
        assert inputs.shape == targets.shape, \
            f'Different shape between inputs ({inputs.shape}) and targets ({targets.shape})'
        
        reduce_axis: List[int] = torch.arange(2, len(inputs.shape)).tolist()
        
        intersection = torch.sum(targets * inputs, dim=reduce_axis)
        if self.squared_pred:
            denominator = torch.sum(inputs ** 2, dim=reduce_axis) + torch.sum(targets ** 2, dim=reduce_axis)
        else:
            denominator = torch.sum(inputs, dim=reduce_axis) + torch.sum(targets, dim=reduce_axis)
        loss: torch.Tensor = 1. - (2.*intersection + self.smooth) / (denominator + self.smooth)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError(f'Unsupported reduction method {self.reduction}. Available: ["sum", "mean"].')
        
        return loss
