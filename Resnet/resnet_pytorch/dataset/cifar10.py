import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloader_cifar10(root="data_cifar10", batch_size=1024, shuffle=True):
    """CIFAR-10 のデータローダーを取得"""
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # データ拡張
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader