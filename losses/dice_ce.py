import torch
import torch.nn as nn

from .dice import SoftDiceLoss


class DiceCELoss(nn.Module):
    def __init__(self, gamma: float = 0.4, smooth: float = 1e-6) -> None:
        super().__init__()
        
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=smooth)
        self.dice = SoftDiceLoss(smooth=smooth)
        
        self.gamma = gamma
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ce = self.cross_entropy(y_pred, y_true)
        dice = self.dice(y_pred, y_true)
        return dice * self.gamma + ce * (1. - self.gamma)
