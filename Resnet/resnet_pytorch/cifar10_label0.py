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

# **記録する特定のラベル**
TARGET_LABEL = 0  # ラベル0のデータのみ記録

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

    # **ResNet モデルの取得**
    model = get_resnet(pretrained=False, num_classes=10, disable_bn=False).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # **ログ用リスト**
    accuracy_log, test_accuracy_log, generalization_gap = [], [], []
    train_loss_log, learning_rate_log = [], []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/Users/hide/卒業研究/resnet_pytorch/output_cifar10/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # **各層の活性化を保存する辞書**
    activations_per_layer = {f'layer{i}': None for i in range(5)}

    def hook(module, input, output, layer_name):
        """特定のラベル (TARGET_LABEL) のみを記録 (最後のバッチ)"""
        global last_labels  # 🔹 最後のバッチのラベル
        mask = (last_labels == TARGET_LABEL).view(-1, 1, 1, 1)  # 🔹 正しい形にリサイズ
        if mask.any():
            selected_activations = output[mask.expand_as(output)].view(-1, *output.shape[1:])  # 🔹 選択した活性化データ
            activations_per_layer[layer_name] = selected_activations.cpu().detach().numpy()

    # **フックを登録**
    for i, layer in enumerate([model.conv1, model.layer1, model.layer2, model.layer3, model.layer4]):
        layer_name = f'layer{i}'
        layer.register_forward_hook(lambda module, input, output, ln=layer_name: hook(module, input, output, ln))

    try:
        for epoch in range(1, total_epoch + 1):
            model.train()
            total_loss = 0.0
            batch_count = 0

            for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch}/{total_epoch}"):
                images, labels = images.to(device), labels.to(device)

                global last_labels
                last_labels = labels  # 🔹 ミニバッチのラベルを保存

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

            # **エポックごとに最後のバッチの層の出力を保存**
            for layer_name, activations in activations_per_layer.items():
                if activations is not None:
                    np.save(os.path.join(output_dir, f"epoch_{epoch}_{layer_name}_label{TARGET_LABEL}.npy"), activations)
                    print(f"✅ Saved {layer_name} (label {TARGET_LABEL}) at epoch {epoch}: shape={activations.shape}")

            # **ログの保存**
            accuracy_log.append(train_acc)
            test_accuracy_log.append(test_acc)
            generalization_gap.append(gap)
            train_loss_log.append(total_loss / batch_count)
            learning_rate_log.append(current_lr)

            print(f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}, Gap = {gap:.4f}, LR = {current_lr:.6f}")

            save_results(output_dir, accuracy_log, test_accuracy_log, generalization_gap, train_loss_log, learning_rate_log)

            # **学習精度が 1.0 に達したら停止**
            if train_acc == 1.0:
                print("🎯 Train accuracy reached 1.0. Stopping training.")
                break

    except KeyboardInterrupt:
        print("\n⏹️ Training manually interrupted. Saving current results...")
        save_results(output_dir, accuracy_log, test_accuracy_log, generalization_gap, train_loss_log, learning_rate_log)
        print(f"✅ Training logs saved in {output_dir}")

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

if __name__ == "__main__":
    train(total_epoch=200, batch_size=1024)