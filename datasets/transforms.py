import argparse

import monai.transforms as T


def get_default_transforms(args: argparse.Namespace):
    train_transforms = T.Compose([
      T.LoadImaged(keys=['image', 'label'], image_only=True, ensure_channel_first=True),
      T.ScaleIntensityRanged(
        keys='image', a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
      ),
      T.CropForegroundd(keys=['image', 'label'], allow_smaller=True, source_key='image'),
      T.Orientationd(keys=['image', 'label'], axcodes='RAS'),
      T.Spacingd(
        keys=['image', 'label'], pixdim=(args.space_x, args.space_y, args.space_z), mode=('bilinear', 'nearest')
      ),
      T.EnsureTyped(keys=['image', 'label'], track_meta=False),
      T.RandCropByPosNegLabeld(
        keys=['image', 'label'],
        spatial_size=(args.roi_x, args.roi_y, args.roi_z),
        image_key='image',
        label_key='label',
        pos=1,
        neg=1,
        num_samples=4,
        allow_smaller=True,
        image_threshold=0
      ),
      T.ResizeWithPadOrCropd(
        keys=['image', 'label'], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'
      ),
      T.RandFlipd(keys=['image', 'label'], spatial_axis=0, prob=args.rand_flip_prob),
      T.RandFlipd(keys=['image', 'label'], spatial_axis=1, prob=args.rand_flip_prob),
      T.RandFlipd(keys=['image', 'label'], spatial_axis=2, prob=args.rand_flip_prob),
      T.RandRotate90d(keys=['image', 'label'], prob=args.rand_rotate90_prob, max_k=3),
      T.RandScaleIntensityd(keys='image', factors=.1, prob=args.rand_scale_intensity_prob),
      T.RandShiftIntensityd(keys='image', offsets=.1, prob=args.rand_shift_intensity_prob),
      T.ToTensord(keys=['image', 'label']),
    ])
    
    valid_transforms = T.Compose([
      T.LoadImaged(keys=['image', 'label'], image_only=True, ensure_channel_first=True),
      T.ScaleIntensityRanged(
        keys='image', a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
      ),
      T.CropForegroundd(keys=['image', 'label'], allow_smaller=True, source_key='image'),
      T.Orientationd(keys=['image', 'label'], axcodes='RAS'),
      T.Spacingd(
        keys=['image', 'label'], pixdim=(args.space_x, args.space_y, args.space_z), mode=('bilinear', 'nearest')
      ),
      T.EnsureTyped(keys=['image', 'label'], track_meta=False),
      T.ToTensord(keys=['image', 'label']),
    ])
    
    return {'train': train_transforms, 'valid': valid_transforms}
