import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.v2 as tfs

class CatsDogsDataset(Dataset):
    def __init__(self, train=True, transforms=None, data_root="data/"):
        self.transforms = transforms
        self.train = train

        if train:
            cats_dir = os.path.join(data_root, "train/cats")
            dogs_dir = os.path.join(data_root, "train/dogs")
            cats = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)]
            dogs = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)]
            self.paths = cats + dogs
            self.targets = [[1,0]] * len(cats) + [[0,1]] * len(dogs)
        else:
            test_dir = os.path.join(data_root, "test")
            self.paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
            self.targets = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        if self.train:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return img, target
        else:
            return img

def get_dataloaders(batch_size=32, train_split=0.9, img_size=(224,224), data_root="data/"):
    transform = tfs.Compose([
        tfs.Resize(img_size),
        tfs.ToImage(),
        tfs.ToDtype(torch.float32, scale=True)
    ])
    
    full_train = CatsDogsDataset(train=True, transforms=transform, data_root=data_root)
    train_len = int(train_split * len(full_train))
    val_len = len(full_train) - train_len
    train_dataset, val_dataset = random_split(full_train, [train_len, val_len])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = CatsDogsDataset(train=False, transforms=transform, data_root=data_root)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
