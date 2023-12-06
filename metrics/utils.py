from typing import Any, Tuple, Union

import torch


def run_metric_reduction(t: torch.Tensor, reduction: str = 'mean') -> Tuple[Union[torch.Tensor, Any], torch.Tensor]:
    nans = torch.isnan(t)
    not_nans = ~nans
    
    t_zero = torch.zeros(1, device=t.device, dtype=torch.float)
    t[nans] = 0
    
    if reduction is None:
        return t, not_nans.float()
    elif reduction == 'mean':
        # take mean by channel (accounting for nans)
        not_nans = not_nans.sum(dim=1).float()
        t = torch.where(not_nans > 0, t.sum(dim=1).float() / not_nans, t_zero)
        
        # mean by batch
        not_nans = (not_nans > 0).sum(dim=0).float()
        t = torch.where(not_nans > 0, t.sum(dim=0).float() / not_nans, t_zero)
    elif reduction == 'sum':
        not_nans = not_nans.sum(dim=[0, 1]).float()
        t = torch.sum(t, dim=[0, 1])  # sum over the batch and channel dims
    else:
        raise ValueError
    
    return t, not_nans
