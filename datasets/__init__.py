import argparse

from monai.data.dataloader import DataLoader

from .btcv import AbdomenDataset
from .samplers import CustomSampler
from .transforms import get_default_transforms


def build_dataset(args: argparse.Namespace):
    transforms = get_default_transforms(args)
    train_set = AbdomenDataset(args.data_root, transform=transforms, is_train=args.is_train, num_workers=args.workers),
    valid_set = AbdomenDataset(
      args.data_root, transform=transforms, is_train=not args.is_train, num_workers=args.workers
    )
    
    # data sampler and dataloader
    train_sampler = CustomSampler(train_set) if args.distributed else None
    train_loader = DataLoader(
      train_set,
      batch_size=args.batch_size,
      shuffle=(train_sampler is None),
      num_workers=args.workers,
      sampler=train_sampler,
      pin_memory=True,
      persistent_workers=True,
    )
    
    valid_sampler = CustomSampler(valid_set, shuffle=False) if args.distributed else None
    valid_loader = DataLoader(
      valid_set,
      batch_size=1,
      shuffle=False,
      num_workers=args.workers,
      sampler=valid_sampler,
      pin_memory=True,
      persistent_workers=True
    )
    
    return train_loader, valid_loader
