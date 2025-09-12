import sys
import os

# `resnet_pytorch` のパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from datetime import datetime
from dataset.mnist6000 import get_dataloader_6000 as get_dataloader
from model.resnet import get_resnet



def evaluate_accuracy(model, dataloader, device):
    """データローダーに対する正答率を計算"""
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

def train(total_epoch: int = 500):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # **Shuffle をオフにし、バッチサイズを 1024 に設定（過学習を促進）**
    train_dataloader, test_dataloader = get_dataloader(root="data_6000", batch_size=1024, shuffle=False)
    model = get_resnet(pretrained=False).to(device)

    criterion = nn.CrossEntropyLoss()
    
    # **Weight Decay（正則化なし）で SGD を使う**
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
    
    # **学習率スケジューラもなし**
    accuracy_log = []
    test_accuracy_log = []
    generalization_gap = []
    last_activation = None

    def hook(module, input, output):
        nonlocal last_activation
        last_activation = output.cpu().detach().numpy().reshape(-1, 512, 1, 1)

    model.layer4.register_forward_hook(hook)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/Users/hide/卒業研究/resnet_pytorch/output_6000/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    train_acc_0 = evaluate_accuracy(model, train_dataloader, device)
    test_acc_0 = evaluate_accuracy(model, test_dataloader, device)

    accuracy_log.append(train_acc_0)
    test_accuracy_log.append(test_acc_0)
    generalization_gap.append(train_acc_0 - test_acc_0)

    if last_activation is not None:
        np.save(os.path.join(output_dir, "epoch_0_layer4.npy"), last_activation)

    print(f"Epoch 0: Train Acc = {train_acc_0:.4f}, Test Acc = {test_acc_0:.4f}, Gap = {generalization_gap[0]:.4f}")

    try:
        for epoch in range(1, total_epoch + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for i, (images, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(images)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                preds = out.argmax(axis=1)
                train_loss += loss.item()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            train_acc = evaluate_accuracy(model, train_dataloader, device)
            test_acc = evaluate_accuracy(model, test_dataloader, device)
            gap = train_acc - test_acc

            accuracy_log.append(train_acc)
            test_accuracy_log.append(test_acc)
            generalization_gap.append(gap)

            if last_activation is not None:
                np.save(os.path.join(output_dir, f"epoch_{epoch}_layer4.npy"), last_activation)

            print(f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}, Gap = {gap:.4f}")

            save_results(output_dir, accuracy_log, test_accuracy_log, generalization_gap)

    except KeyboardInterrupt:
        print("\n⏹️ Training manually interrupted. Saving current results...")
        save_results(output_dir, accuracy_log, test_accuracy_log, generalization_gap)
        print(f"✅ Training logs saved in {output_dir}")

def save_results(output_dir, accuracy_log, test_accuracy_log, generalization_gap):
    """エポックごとの記録を CSV に保存"""
    output_path = os.path.join(output_dir, "epoch_accuracies.csv")
    with open(output_path, "w", newline='') as csvfile:
        fieldnames = ['epoch', 'train_accuracy', 'test_accuracy', 'generalization_gap']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(len(accuracy_log)):
            writer.writerow({
                'epoch': epoch + 1,
                'train_accuracy': accuracy_log[epoch],
                'test_accuracy': test_accuracy_log[epoch],
                'generalization_gap': generalization_gap[epoch]
            })

if __name__ == "__main__":
    train(total_epoch=10000)