from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DiceLoss', 'DiceCELoss']


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
        """The Combination of DiceLoss and CrossEntropyLoss

        :param include_background (bool): Whether consider background as the first class. Default: True.
        :param sigmoid (bool): Whether apply sigmoid function. Default: False.
        :param softmax (bool): Whether applying softmax function. Default: True.
        :param reduction (str): The reduction method. Default: 'mean'
        :param squared_pred (bool): Whether using squared prediction at the denominator. Default: True.
        :param smooth (float): Smooth value to avoid divided by zero. Default: 1e-5.
        """
        super(DiceLoss, self).__init__()
        
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.reduction = reduction
        self.smooth = smooth
        self.squared_pred = squared_pred
        self.include_background = include_background
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss from dice coefficient.

        :param inputs (torch.Tensor): The prediction tensor
        :param targets (torch.Tensor): The ground-truth tensor
        :return: The loss value.
        """
        if self.sigmoid:
            inputs = F.sigmoid(inputs)
        
        num_classes = inputs.shape[1]
        if self.softmax:
            assert num_classes != 1, 'single channel prediction, `softmax=True` is ignored'
            inputs = F.softmax(inputs, dim=1)
        
        if num_classes != 1:
            if targets.shape[1] == 1 or targets.ndim == 5:
                targets = targets.squeeze(1)
            targets = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3)
        
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


class DiceCELoss(nn.Module):
    def __init__(
      self,
      include_background: bool = True,
      sigmoid: bool = False,
      softmax: bool = True,
      squared_pred: bool = True,
      reduction: str = 'mean',
      smooth: float = 1e-5,
      lambda_dice: float = 1.,
      lambda_ce: float = 1.
    ) -> None:
        """The Combination of DiceLoss and CrossEntropyLoss

        :param include_background (bool): Whether consider background as the first class. Default: True.
        :param sigmoid (bool): Whether apply sigmoid function. Default: False.
        :param softmax (bool): Whether applying softmax function. Default: True.
        :param squared_pred (bool): Whether using squared prediction at the denominator. Default: True.
        :param reduction (str): The reduction method. Default: 'mean'
        :param smooth (float): Smooth value to avoid divided by zero. Default: 1e-5.
        :param lambda_dice (float): Weighted value for DiceLoss. Default: 1.
        :param lambda_ce (float): Weighted value for CrossEntropyLoss. Default: 1.
        """
        super().__init__()
        
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        if self.lambda_dice < 0.:
            raise ValueError
        if self.lambda_ce < 0.:
            raise ValueError
        
        self.dice = DiceLoss(
          include_background=include_background,
          sigmoid=sigmoid,
          softmax=softmax,
          reduction=reduction,
          squared_pred=squared_pred,
          smooth=smooth
        )
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.binary_cross_entropy = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combination weighted losses.

        :param inputs (torch.Tensor): The prediction tensor
        :param targets (torch.Tensor): The ground-truth tensor
        :return: The total loss value.
        """
        if len(inputs.shape) != len(targets.shape):
            raise ValueError(
              "the number of dimensions for input and target should be the same, "
              f"got shape {inputs.shape} and {targets.shape}."
            )
        
        dice_loss = self.dice(inputs, targets)
        ce_loss = self.ce(inputs, targets) if inputs.shape[1] != 1 else self.bce(inputs, targets)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
        return total_loss
    
    def ce(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute CrossEntropy Loss"""
        if inputs.shape[1] != 1 and targets.shape[1] == 1:
            targets = targets.squeeze(1).long()
        elif not torch.is_floating_point(targets):
            targets = targets.to(dtype=inputs.dtype)
        
        return self.cross_entropy(inputs, targets)
    
    def bce(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Binary CrossEntropy Loss"""
        if not torch.is_floating_point(targets):
            targets = targets.to(dtype=inputs.dtype)
        
        return self.binary_cross_entropy(inputs, targets)
