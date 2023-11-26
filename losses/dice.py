import torch
import torch.nn as nn


def soft_dice_coefficient(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-6):
    inter = torch.sum(y_true * y_pred)
    union = torch.sum(y_true ** 2) + torch.sum(y_pred ** 2)
    return (2.*inter + smooth) / (union+smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        
        self.smooth = smooth
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return 1 - soft_dice_coefficient(y_pred, y_true, self.smooth)
