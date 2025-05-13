import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(train_dir, val_dir, batch_size=8, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(train_dir, transform)
    val_ds   = datasets.ImageFolder(val_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
