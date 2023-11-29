from typing import Callable

import torch


class EarlyStopping:
    """Early Stopping Callback simple implementation"""
    
    mode_dict = {'min': torch.lt, 'max': torch.gt}
    
    def __init__(self, patience: int = 5, mode: str = 'min', min_delta: float = 1e-5) -> None:
        super().__init__()
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        
        self.mode = mode
        if self.mode not in self.mode_dict:
            raise NotImplementedError
        
        self.best_score = torch.tensor(torch.inf) if self.mode == 'min' else torch.tensor(-torch.inf)
    
    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]
    
    def step(self, metric: torch.Tensor) -> bool:
        if not isinstance(metric, torch.Tensor):
            metric = torch.tensor(metric)
        if self.monitor_op(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
        elif self.monitor_op(self.min_delta, metric - self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
