import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as tfs

from config import IMAGE_SIZE


class CatsDogsDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.train = train

        self.transforms = tfs.Compose([
            tfs.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            tfs.RandomHorizontalFlip(p=0.5),
            tfs.ToImage(),
            tfs.ToDtype(torch.float32, scale=True)
        ])

        self.samples = []

        if train:
            cats_dir = os.path.join(root_dir, "cats")
            dogs_dir = os.path.join(root_dir, "dogs")

            for file in os.listdir(cats_dir):
                self.samples.append(
                    (os.path.join(cats_dir, file), 0)
                )

            for file in os.listdir(dogs_dir):
                self.samples.append(
                    (os.path.join(dogs_dir, file), 1)
                )

        else:
            for file in os.listdir(root_dir):
                self.samples.append(
                    os.path.join(root_dir, file)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.train:
            path, label = self.samples[idx]

            image = Image.open(path).convert("RGB")
            image = self.transforms(image)

            return image, label

        path = self.samples[idx]

        image = Image.open(path).convert("RGB")
        image = self.transforms(image)

        return image
