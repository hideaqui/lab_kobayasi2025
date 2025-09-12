import os
import csv
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from datetime import datetime
from dataset.mnist import get_dataloader
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

def train(total_epoch: int = 100):
    # デバイスの選択
    device = torch.device("mps") if torch.backends.mps.is_available() else \
             torch.device("cuda") if torch.cuda.is_available() else \
             torch.device("cpu")
    print(f"Using device: {device}")

    # **データローダーのバッチサイズを 1024 に設定**
    train_dataloader, test_dataloader = get_dataloader(root="data", batch_size=1024)

    # ResNetの取得（事前学習なし）
    model = get_resnet(pretrained=False).to(device)

    # 損失関数と最適化
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 記録用リスト
    accuracy_log = []
    test_accuracy_log = []
    generalization_gap = []

    # Layer4 の活性化保存
    second_last_activation = None  
    last_activation = None         

    def hook(module, input, output):
        nonlocal last_activation, second_last_activation
        second_last_activation = last_activation  
        last_activation = output.cpu().detach().numpy()  

    model.layer4.register_forward_hook(hook)

    # 保存用フォルダ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/Users/hide/卒業研究/resnet_pytorch/output/1024batch_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # **エポック0 の Layer4 の活性化と正答率を保存**
    model.eval()
    with torch.no_grad():
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  
            preds = outputs.argmax(axis=1)
            train_acc_0 = (preds == labels).sum().item() / labels.size(0)  
            break  

    test_acc_0 = evaluate_accuracy(model, test_dataloader, device)

    # **エポック0 のデータを保存**
    np.save(os.path.join(output_dir, "epoch_0_layer4.npy"), last_activation)

    # **ログ順序を統一**
    accuracy_log.append(train_acc_0)
    test_accuracy_log.append(test_acc_0)
    generalization_gap.append(train_acc_0 - test_acc_0)  

    print(f"Epoch 0: Train Acc = {train_acc_0:.4f}, Test Acc = {test_acc_0:.4f}, Gap = {generalization_gap[0]:.4f}")

    # **エポックごとの学習**
    for epoch in range(1, total_epoch + 1):  
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch}/{total_epoch}"):
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

        scheduler.step()

        # **エポックごとの正答率**
        train_acc = train_correct / train_total
        test_acc = evaluate_accuracy(model, test_dataloader, device)

        # **記録**
        accuracy_log.append(train_acc)
        test_accuracy_log.append(test_acc)
        generalization_gap.append(train_acc - test_acc)  

        # **Layer4 の活性化保存**
        np.save(os.path.join(output_dir, f"epoch_{epoch}_layer4.npy"), second_last_activation)

        print(f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}, Gap = {generalization_gap[epoch]}")

    return accuracy_log, test_accuracy_log, generalization_gap, output_dir  

def save_epoch_accuracies():
    accuracy_log, test_accuracy_log, generalization_gap, output_dir = train(total_epoch=100)

    # **CSVファイルに保存**
    output_path = os.path.join(output_dir, "epoch_accuracies.csv")
    with open(output_path, "w", newline='') as csvfile:
        fieldnames = ['epoch', 'train_accuracy', 'test_accuracy', 'generalization_gap']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(len(accuracy_log)):
            writer.writerow({
                'epoch': epoch,  
                'train_accuracy': accuracy_log[epoch],
                'test_accuracy': test_accuracy_log[epoch],
                'generalization_gap': generalization_gap[epoch]
            })

    print(f"Epoch accuracies and activations saved in {output_dir}")

if __name__ == "__main__":
    save_epoch_accuracies()