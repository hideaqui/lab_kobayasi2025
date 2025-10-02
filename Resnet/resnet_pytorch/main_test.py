import torch.nn.init as init

def initialize_weights(model, mode="kaiming"):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if mode == "normal":
                init.normal_(m.weight, mean=0.0, std=0.02)
            elif mode == "uniform":
                init.uniform_(m.weight, a=-0.1, b=0.1)
            elif mode == "xavier_uniform":
                init.xavier_uniform_(m.weight)
            elif mode == "xavier_normal":
                init.xavier_normal_(m.weight)
            elif mode == "kaiming_uniform":
                init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif mode == "kaiming_normal":
                init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                init.zeros_(m.bias)
import os
import os
print(">>> running script from:", os.path.abspath(__file__))
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

def train(total_epoch: int = 20):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    train_dataloader, test_dataloader = get_dataloader(root="data", batch_size=64)

    # 事前学習済みモデルを使用しない
    model = get_resnet(pretrained=False).to(device)
    initialize_weights(model, mode="kaiming_normal")  # ここで方式を切り替え

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    accuracy_log = []
    test_accuracy_log = []
    generalization_gap = []

    last_activation = None  # 最後のミニバッチの活性化を保存

    def hook(module, input, output):
        nonlocal last_activation
        last_activation = output.cpu().detach().numpy()

    model.layer4.register_forward_hook(hook)

    # 結果保存用フォルダの作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/Users/hide/lab_kobayasi2025/Resnet/resnet_pytorch/output/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 学習前（epoch 0）の Layer4 の活性化と正答率を保存
    model.eval()
    with torch.no_grad():
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 最初のバッチで Layer4 の出力を得る
            preds = outputs.argmax(axis=1)
            train_acc_0 = (preds == labels).sum().item() / labels.size(0)  # 最初のバッチの train_acc
            break  # 最初のバッチのみ使用

    test_acc_0 = evaluate_accuracy(model, test_dataloader, device)

    # epoch 0 のデータを保存
    np.save(os.path.join(output_dir, "epoch_0_layer4.npy"), last_activation)
    accuracy_log.append(train_acc_0)  # 学習前の train_acc
    test_accuracy_log.append(test_acc_0)
    generalization_gap.append(train_acc_0 - test_acc_0)  # 汎化誤差

    print(f"Epoch 0: Train Acc = {train_acc_0:.4f}, Test Acc = {test_acc_0:.4f}, Gap = {train_acc_0 - test_acc_0:.4f}")

    for epoch in range(total_epoch):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
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

        # エポックごとの正答率を取得
        train_acc = train_correct / train_total
        test_acc = evaluate_accuracy(model, test_dataloader, device)

        # 記録
        accuracy_log.append(train_acc)
        test_accuracy_log.append(test_acc)
        generalization_gap.append(train_acc - test_acc)  # 汎化誤差

        # 最後のミニバッチの Layer4 の活性化を保存
        np.save(os.path.join(output_dir, f"epoch_{epoch+1}_layer4.npy"), last_activation)

        print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}, Gap = {generalization_gap[epoch]:.4f}")

    return accuracy_log, test_accuracy_log, generalization_gap

def save_epoch_accuracies():
    accuracy_log, test_accuracy_log, generalization_gap = train(total_epoch=20)

    # 結果を保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/Users/hide/lab_kobayasi2025/Resnet/resnet_pytorch/output/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # CSVファイルに保存
    output_path = os.path.join(output_dir, "epoch_accuracies.csv")
    with open(output_path, "w", newline='') as csvfile:
        fieldnames = ['epoch', 'train_accuracy', 'test_accuracy', 'generalization_gap']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(len(accuracy_log)):
            writer.writerow({
                'epoch': epoch,  # 0 からスタート
                'train_accuracy': accuracy_log[epoch],
                'test_accuracy': test_accuracy_log[epoch],
                'generalization_gap': generalization_gap[epoch]
            })

    print(f"Epoch accuracies saved in {output_path}")

if __name__ == "__main__":
    save_epoch_accuracies()