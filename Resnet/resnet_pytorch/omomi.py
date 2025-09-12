import os
import csv
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from datetime import datetime
from dataset.mnist6000 import get_dataloader_6000 as get_dataloader
from model.resnet import get_resnet


def evaluate_accuracy(model, dataloader, device):
    """データローダーに対する正答率を計算（全サンプルを評価）"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(axis=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train(total_epoch: int = 200, batch_size: int = 1024):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # データローダーの取得
    train_dataloader, test_dataloader = get_dataloader(root="data_6000", batch_size=batch_size, shuffle=True)

    # `drop_last=True` を適用
    train_dataloader = torch.utils.data.DataLoader(
        train_dataloader.dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataloader.dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    model = get_resnet(pretrained=False).to(device)

    # Batch Normalization を評価モードに固定
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 50エポックごとに減衰

    # 記録用リスト
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
    output_dir = f"/Users/hide/卒業研究/resnet_pytorch/output_6000/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        for epoch in range(1, total_epoch + 1):
            model.train()
            total_loss = 0.0
            batch_count = 0

            for i, (images, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{total_epoch}")):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(images)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            scheduler.step()

            # エポックごとの評価
            train_acc = evaluate_accuracy(model, train_dataloader, device)
            test_acc = evaluate_accuracy(model, test_dataloader, device)
            gap = train_acc - test_acc
            current_lr = optimizer.param_groups[0]['lr']

            # エポック最後の layer4 活性化と fc の重みを保存
            if last_activation is not None:
                np.save(os.path.join(output_dir, f"epoch_{epoch}_layer4.npy"), last_activation)
            if last_fc_weight is not None and last_fc_bias is not None:
                np.save(os.path.join(output_dir, f"epoch_{epoch}_fc_weight.npy"), last_fc_weight)
                np.save(os.path.join(output_dir, f"epoch_{epoch}_fc_bias.npy"), last_fc_bias)

            # ログの保存
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