import torch
import torch.nn as nn

from .dice import DiceLoss


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
          reduction=self.reduction,
          squared_pred=squared_pred,
          smooth=smooth
        )
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.binary_cross_entropy = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """

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
            targets = targets.to(dtype=inputs.dtyp)
        
        return self.cross_entropy(inputs, targets)
    
    def bce(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Binary CrossEntropy Loss"""
        if not torch.is_floating_point(targets):
            targets = targets.to(dtype=inputs.dtype)
        
        return self.binary_cross_entropy(inputs, targets)
