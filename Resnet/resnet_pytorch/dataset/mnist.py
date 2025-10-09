from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import random
import numpy as np

def get_dataloader(root: str, batch_size: int = 256, num_workers: int = 4, seed: int = 1008):
    # 再現性のためのseed固定
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Transform設定：Normalizeなし
    transform = transforms.Compose([
        transforms.ToTensor(),          # 0〜1 に正規化（自動）
        transforms.RandomRotation(90),  # Tensor対応のランダム回転
    ])

    # データセットを定義
    train_dataset = MNIST(root=root, train=True,  download=True, transform=transform)
    test_dataset  = MNIST(root=root, train=False, download=True, transform=transform)

    # DataLoader設定（GPU転送最適化）
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    return train_dataloader, test_dataloader
