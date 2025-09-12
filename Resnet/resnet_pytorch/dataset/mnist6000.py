import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

def get_dataloader_6000(root: str, batch_size: int=64, shuffle: bool=True) -> (DataLoader, DataLoader):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=90),
    ])

    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)

    train_size = 6000  # 必要なサンプル数
    if len(train_dataset) > train_size:
        indices = sorted(torch.arange(train_size).tolist())  # 最初の6000サンプルを順番に取得
        train_dataset = Subset(train_dataset, indices)  

    # バッチを **順番に全通り** 取得するように shuffle=False にする
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader