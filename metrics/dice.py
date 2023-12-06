from typing import List, Optional, Union

import torch


class DiceMetric:
    reductions = ['mean', 'sum', None]
    
    def __init__(
      self,
      include_background: bool = True,
      sigmoid: bool = False,
      softmax: Optional[bool] = None,
      reduction: str = 'mean',
      num_classes: Optional[int] = None,
    ) -> None:
        """Class to compute Dice Metric

        :param include_background (bool): Whether adding background as first class. Default: True.
        :param sigmoid (bool): Whether applying sigmoid function. Default: False.
        :param softmax (bool | None): Whether applying softmax function. Default: None.
        :param num_classes (int | None): The number of classes. Default: None.
        :param reduction (str | None): The reduction method. Default: 'mean'.
        """
        super().__init__()
        
        self.include_background = include_background
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.num_classes = num_classes
        assert reduction in self.reductions, NotImplementedError
        self.reduction = reduction
    
    def _compute_dice_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute Dice score from prediction and ground-truth

        :param y_pred (torch.Tensor): predict tensor
        :param y_true (torch.Tensor): ground-truth tensor
        :return: A tensor that stores dice score value
        """
        inter = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        return (2.*inter) / union
    
    def _compute_list(self, y_pred: List[torch.Tensor], y_true: List[torch.Tensor]):
        """Computes metric if inputs are list of torch.Tensor
        
        :param y_pred (List[torch.Tensor]): List of prediction tensors
        :param y_true (List[torch.Tensor]): List of ground-truth tensors
        :return: Value of Dice Metric with reduction
        """
        ret = torch.stack([
          self._compute_tensor(
            p.detach().unsqueeze(0),
            g.detach().unsqueeze(0),
          ) for p, g in zip(y_pred, y_true)
        ])
        
        if self.reduction == 'mean':
            return ret.mean().detach().cpu()
        elif self.reduction == 'sum':
            return ret.sum().detach().cpu()
        elif self.reduction is None:
            return ret.detach().cpu()
        else:
            raise ValueError
    
    def _compute_tensor(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Computes metric if inputs are torch.Tensor

        :param y_pred (torch.Tensor): Prediction tensor
        :param y_true (torch.Tensor): Ground-truth tensor
        :return: Value of Dice Metric with reduction
        """
        num_classes = y_pred.shape[1] if self.num_classes is None else self.num_classes
        
        if self.softmax:
            assert num_classes > 1
            y_pred = y_pred.softmax(dim=1, keepdim=True)
        elif self.sigmoid:
            y_pred = y_pred > 0.5
        
        first_channel = 0 if self.include_background else 1
        data = []
        for p, t in zip(y_pred, y_true):
            c_list = []
            for c in range(first_channel, num_classes) if num_classes > 1 else [1]:
                c_list.append(self._compute_dice_score(p[c], t[c]))
            data.append(torch.stack(c_list))
        
        data = torch.stack(data, dim=0).contiguous()
        if self.reduction == 'mean':
            return data.mean().detach().cpu()
        elif self.reduction == 'sum':
            return data.sum().detach().cpu()
        elif self.reduction is None:
            return data.detach().cpu()
        else:
            raise ValueError
    
    def __call__(
      self, preds: Union[torch.Tensor, List[torch.Tensor]], targets: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """Computes Dice Metric depends on inputs type

        :param preds (torch.Tensor | List[torch.Tensor]):
        :param targets (torch.Tensor | List[torch.Tensor]):
        :return: Value of Dice Metric with reduction
        """
        if isinstance(preds, (list, tuple)) or isinstance(targets, (list, tuple)):
            return self._compute_list(preds, targets)
        if isinstance(preds, torch.Tensor) or isinstance(targets, torch.Tensor):
            return self._compute_tensor(preds.detach(), targets.detach())
        raise ValueError