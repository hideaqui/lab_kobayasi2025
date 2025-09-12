import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloader_cifar100(root="data_cifar100", batch_size=1024, shuffle=True):
    """CIFAR-100 のデータローダーを取得"""

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # データ拡張
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # CIFAR-100 の正規化
    ])

    trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)

    return train_loader, test_loader