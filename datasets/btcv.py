import glob
from typing import Callable, Dict, Optional

from monai.data.dataset import CacheDataset
from torch.utils.data import random_split


class AbdomenDataset(CacheDataset):
    def __init__(
      self,
      root: str,
      transform: Optional[Dict[str, Callable]] = None,
      is_train: bool = True,
      num_workers: int = 4
    ) -> None:
        images = sorted(glob.glob(f'{root}/train/image/image*.nii.gz'))
        labels = sorted(glob.glob(f'{root}/train/label/label*.nii.gz'))
        self.ids = [{'image': image, 'label': label} for image, label in zip(images, labels)]
        self.transform = transform
        
        train_list, valid_list = random_split(self.ids, [0.8, 0.2])
        if is_train:
            transform = self.transform['train'] if self.transform is not None else None
            super().__init__(train_list, transform=transform, cache_num=24, cache_rate=1., num_workers=num_workers)
        else:
            transform = self.transform['valid'] if self.transform is not None else None
            super().__init__(valid_list, transform=transform, cache_num=8, cache_rate=1., num_workers=num_workers)
