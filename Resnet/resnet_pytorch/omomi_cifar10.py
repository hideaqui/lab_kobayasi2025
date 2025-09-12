import os
import csv
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from dataset.cifar10 import get_dataloader_cifar10  # CIFAR-10 のデータローダー
from model.resnet3 import get_resnet  # ResNet モデルの取得

def evaluate_accuracy(model, dataloader, device):
    """データローダーに対する正答率を計算"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train(total_epoch: int = 200, batch_size: int = 1024):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # **CIFAR-10 データローダーの取得**
    train_dataloader, test_dataloader = get_dataloader_cifar10(root="data_cifar10", batch_size=batch_size, shuffle=True)

    # **データセットサイズを明示**
    print(f"Train dataset size: {len(train_dataloader.dataset)} (Expected: 50,000)")
    print(f"Test dataset size: {len(test_dataloader.dataset)} (Expected: 10,000)")

    # **Batch Normalization を有効化**
    model = get_resnet(pretrained=False, num_classes=10, disable_bn=False).to(device)  # ← 修正

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # **ログ用リスト**
    accuracy_log, test_accuracy_log, generalization_gap = [], [], []
    train_loss_log, learning_rate_log = [], []

    last_activation, last_fc_weight, last_fc_bias = None, None, None

    def activation_hook(module, input, output):
        """layer4 の活性化を記録"""
        nonlocal last_activation
        last_activation = output.cpu().detach().numpy().reshape(-1, 512, 1, 1)

    def fc_weight_hook(module, input, output):
        """最終層（fc）の重みを記録"""
        nonlocal last_fc_weight, last_fc_bias
        last_fc_weight = module.weight.cpu().detach().numpy()
        last_fc_bias = module.bias.cpu().detach().numpy()

    model.layer4.register_forward_hook(activation_hook)
    model.fc.register_forward_hook(fc_weight_hook)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/Users/hide/卒業研究/resnet_pytorch/output_cifar10/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        for epoch in range(1, total_epoch + 1):
            model.train()
            total_loss = 0.0
            batch_count = 0

            for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch}/{total_epoch}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(images)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            scheduler.step()

            # **エポックごとの評価**
            train_acc = evaluate_accuracy(model, train_dataloader, device)
            test_acc = evaluate_accuracy(model, test_dataloader, device)
            gap = train_acc - test_acc
            current_lr = optimizer.param_groups[0]['lr']

            # **エポック最後の layer4 活性化と fc の重みを保存**
            if last_activation is not None:
                np.save(os.path.join(output_dir, f"epoch_{epoch}_layer4.npy"), last_activation)
            if last_fc_weight is not None and last_fc_bias is not None:
                np.save(os.path.join(output_dir, f"epoch_{epoch}_fc_weight.npy"), last_fc_weight)
                np.save(os.path.join(output_dir, f"epoch_{epoch}_fc_bias.npy"), last_fc_bias)

            # **ログの保存**
            accuracy_log.append(train_acc)
            test_accuracy_log.append(test_acc)
            generalization_gap.append(gap)
            train_loss_log.append(total_loss / batch_count)
            learning_rate_log.append(current_lr)

            print(f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}, Gap = {gap:.4f}, LR = {current_lr:.6f}")

            save_results(output_dir, accuracy_log, test_accuracy_log, generalization_gap, train_loss_log, learning_rate_log)

    except KeyboardInterrupt:
        print("\n⏹️ Training manually interrupted. Saving current results...")
        save_results(output_dir, accuracy_log, test_accuracy_log, generalization_gap, train_loss_log, learning_rate_log)
        print(f"✅ Training logs saved in {output_dir}")

    # **学習結果をプロット**
    plot_training_results(output_dir, accuracy_log, test_accuracy_log, train_loss_log)

def save_results(output_dir, accuracy_log, test_accuracy_log, generalization_gap, train_loss_log, learning_rate_log):
    """エポックごとの記録を CSV に保存"""
    output_path = os.path.join(output_dir, "epoch_accuracies.csv")
    with open(output_path, "w", newline='') as csvfile:
        fieldnames = ['epoch', 'train_accuracy', 'test_accuracy', 'generalization_gap', 'train_loss', 'learning_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(len(accuracy_log)):
            writer.writerow({
                'epoch': epoch + 1,
                'train_accuracy': accuracy_log[epoch],
                'test_accuracy': test_accuracy_log[epoch],
                'generalization_gap': generalization_gap[epoch],
                'train_loss': train_loss_log[epoch],
                'learning_rate': learning_rate_log[epoch]
            })

def plot_training_results(output_dir, accuracy_log, test_accuracy_log, train_loss_log):
    """学習結果をプロット"""
    epochs = np.arange(1, len(accuracy_log) + 1)

    plt.figure(figsize=(12, 5))

    # **精度のプロット**
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy_log, label="Train Accuracy", marker="o")
    plt.plot(epochs, test_accuracy_log, label="Test Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training & Test Accuracy")

    # **損失のプロット**
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss_log, label="Train Loss", marker="o", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_plot.png"))
    plt.show()

if __name__ == "__main__":
    train(total_epoch=200, batch_size=1024)