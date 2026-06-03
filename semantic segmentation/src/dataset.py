import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
import random

class SegmentationDataset(Dataset):
    def __init__(self, train=True, transform=None, data_root=config.DATA_ROOT):
        self.train = train
        self.transform = transform
        if train:
            images_dir = config.TRAIN_IMAGES_DIR
            masks_dir = config.TRAIN_MASKS_DIR
            self.data_path = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
            self.target_path = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
        else:
            images_dir = config.TEST_IMAGES_DIR
            self.data_path = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
            self.target_path = None
        self.length = len(self.data_path)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.data_path[idx]).convert('RGB'))
        if self.train:
            mask = np.array(Image.open(self.target_path[idx]))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            augmented = self.transform(image=img, mask=mask)
            return augmented['image'], augmented['mask'].long()
        else:
            augmented = self.transform(image=img)
            return augmented['image']

    def __len__(self):
        return self.length

def get_transforms():
    train_transform = A.Compose([
        A.HorizontalFlip(p=config.FLIP_PROB),
        A.RandomRotate90(p=config.ROTATE90_PROB),
        A.RandomBrightnessContrast(
            brightness_limit=config.BRIGHTNESS_CONTRAST_LIMIT,
            contrast_limit=config.BRIGHTNESS_CONTRAST_LIMIT,
            p=config.BRIGHTNESS_CONTRAST_PROB
        ),
        A.GaussNoise(std_range=config.GAUSS_NOISE_STD_RANGE, p=config.GAUSS_NOISE_PROB),
        A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
        ToTensorV2(),
    ])
    return train_transform, val_transform

def create_loaders():
    train_transform, val_transform = get_transforms()
    full_train = SegmentationDataset(train=True, transform=train_transform)
    full_val = SegmentationDataset(train=True, transform=val_transform)
    total_len = len(full_train)
    train_len = int(config.TRAIN_SPLIT * total_len)
    val_len = total_len - train_len
    indices = list(range(total_len))
    random.seed(config.RANDOM_SEED)
    random.shuffle(indices)
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]
    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_val, val_indices)
    test_dataset = SegmentationDataset(train=False, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
