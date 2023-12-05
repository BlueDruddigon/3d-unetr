import glob
from typing import Callable, Dict, Optional

from monai.data.dataset import Dataset
from torch.utils.data import random_split

from utils.misc import seed_everthing


class AbdomenDataset(Dataset):
    def __init__(
      self,
      root: str,
      transform: Optional[Dict[str, Callable]] = None,
      is_train: bool = True,
    ) -> None:
        images = sorted(glob.glob(f'{root}/train/image/image*.nii.gz'))
        labels = sorted(glob.glob(f'{root}/train/label/label*.nii.gz'))
        self.ids = [{'image': image, 'label': label} for image, label in zip(images, labels)]
        self.transform = transform
        
        seed_everthing(1234)
        train_list, valid_list = random_split(self.ids, [0.8, 0.2])
        if is_train:
            transform = self.transform['train'] if self.transform is not None else None
            super().__init__(train_list, transform=transform)
        else:
            transform = self.transform['valid'] if self.transform is not None else None
            super().__init__(valid_list, transform=transform)
