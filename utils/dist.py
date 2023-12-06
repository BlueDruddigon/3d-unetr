from typing import List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist

__all__ = ['dist_all_gather', 'setup_for_distributed']


def setup_for_distributed(is_master: bool):
    """Setup print function at master process
    
    :param is_master (bool): Whether is master process or not.
    """
    import builtins as __builtins__
    builtin_print = __builtins__.print
    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtins__.print = print


def dist_all_gather(
  tensor_list: List[torch.Tensor],
  valid_batch_size: Optional[int] = None,
  out_numpy: bool = False,
  world_size: Optional[int] = None,
  no_barrier: bool = False,
  is_valid: Optional[torch.Tensor] = None
) -> List[List[Union[torch.Tensor, np.ndarray]]]:
    """Performs an all-gather operation in a distributed setting.

    :param tensor_list (list): A list of tensor to be gathered from all processes.
    :param valid_batch_size (int, Optional): The number of valid batches across all processes.
    :param out_numpy (bool, Optional): Whether to return the gathered tensors as a numpy array or a list.
    :param world_size (int, Optional): The total number of processes. Default: None.
    :param no_barrier (bool, Optional): Whether to create a barrier before gathering tensors. Default: False.
    :param is_valid (bool, Optional): Whether the current process has a valid input tensor or not. Default: None.

    :return: A List[List[Union[torch.Tensor, np.ndarray]]], where each list corresponds to a tensor in tensor_list
             and contains the gathered tensors from all processes.
    """
    if world_size is None:
        world_size = dist.get_world_size()
    
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    
    if not no_barrier:
        dist.barrier()
    
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            dist.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            
            tensor_list_out.append(gather_list)
    
    return tensor_list_out
