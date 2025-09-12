import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

def get_dataloader(root="data", batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=90),
    ])

    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, train_dataset, test_dataset

def visualize_samples(train_dataset, test_dataset, num_samples=50):
    """トレーニングデータとテストデータから num_samples 個ずつ取得し、並べて表示"""
    
    # ランダムに num_samples 個ずつ選択
    train_indices = torch.randperm(len(train_dataset))[:num_samples]
    test_indices = torch.randperm(len(test_dataset))[:num_samples]

    train_samples = Subset(train_dataset, train_indices.tolist())
    test_samples = Subset(test_dataset, test_indices.tolist())

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples, 2))

    # トレーニングセットを表示
    for i, (image, label) in enumerate(train_samples):
        axes[0, i].imshow(image.squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"T: {label}")

    # テストセットを表示
    for i, (image, label) in enumerate(test_samples):
        axes[1, i].imshow(image.squeeze(), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(f"E: {label}")

    plt.suptitle(f"Top: Training Samples | Bottom: Test Samples", fontsize=12)
    plt.show()

if __name__ == "__main__":
    train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloader(root="data_6016")
    
    visualize_samples(train_dataset, test_dataset, num_samples=50)