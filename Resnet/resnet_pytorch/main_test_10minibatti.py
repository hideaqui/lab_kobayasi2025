import os
import csv
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from datetime import datetime
from dataset.mnist6000 import get_dataloader_6000 as get_dataloader
from model.resnet import get_resnet


def evaluate_accuracy(model, dataloader, device, sample_batches=10):
    """データローダーに対する正答率を計算（指定したバッチ数のみ評価）"""
    model.eval()
    correct, total = 0, 0
    batch_count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(axis=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            batch_count += 1
            if batch_count >= sample_batches:
                break  # 指定したバッチ数で評価を止める

    return correct / total


def train(total_epoch: int = 40, sample_batches: int = 10):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # データローダーの取得
    train_dataloader, test_dataloader = get_dataloader(root="data_6000", batch_size=64, shuffle=True)
    model = get_resnet(pretrained=False).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    accuracy_log = []
    test_accuracy_log = []
    generalization_gap = []
    batchwise_log = []

    last_activation = None  # **最後のミニバッチの活性化を保存する変数**

    def hook(module, input, output):
        nonlocal last_activation
        last_activation = output.cpu().detach().numpy().reshape(-1, 512, 1, 1)  # **(64, 512, 1, 1) に統一**

    model.layer4.register_forward_hook(hook)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/Users/hide/卒業研究/resnet_pytorch/output_6000/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # **エポック0: 訓練データとテストデータの accuracy を記録**
    model.eval()
    train_acc_0 = evaluate_accuracy(model, train_dataloader, device, sample_batches)
    test_acc_0 = evaluate_accuracy(model, test_dataloader, device, sample_batches)
    gap_0 = train_acc_0 - test_acc_0

    accuracy_log.append(train_acc_0)
    test_accuracy_log.append(test_acc_0)
    generalization_gap.append(gap_0)

    print(f"Epoch 0: Train Acc = {train_acc_0:.4f}, Test Acc = {test_acc_0:.4f}, Gap = {gap_0:.4f}")

    try:
        for epoch in range(1, total_epoch + 1):
            model.train()
            batch_count = 0

            for i, (images, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(images)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                batch_count += 1

                # **10バッチごとに中間層出力を記録**
                if batch_count % sample_batches == 0:
                    batch_train_acc = evaluate_accuracy(model, train_dataloader, device, sample_batches)
                    batch_test_acc = evaluate_accuracy(model, test_dataloader, device, sample_batches)
                    batch_gap = batch_train_acc - batch_test_acc  # ✅ 追加

                    batchwise_log.append([epoch, batch_count, batch_train_acc, batch_test_acc, batch_gap])  # ✅ gap 追加

                    if last_activation is not None:
                        np.save(os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_count}_layer4.npy"), last_activation)

            scheduler.step()

            train_acc = evaluate_accuracy(model, train_dataloader, device, sample_batches)
            test_acc = evaluate_accuracy(model, test_dataloader, device, sample_batches)
            gap = train_acc - test_acc

            accuracy_log.append(train_acc)
            test_accuracy_log.append(test_acc)
            generalization_gap.append(gap)

            print(f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}, Gap = {gap:.4f}")

            save_results(output_dir, accuracy_log, test_accuracy_log, generalization_gap)
            save_batchwise_results(output_dir, batchwise_log)

    except KeyboardInterrupt:
        print("\n⏹️ Training manually interrupted. Saving current results...")
        save_results(output_dir, accuracy_log, test_accuracy_log, generalization_gap)
        save_batchwise_results(output_dir, batchwise_log)
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


def save_batchwise_results(output_dir, batchwise_log):
    """バッチごとの評価結果を CSV に保存"""
    output_path = os.path.join(output_dir, "batchwise_accuracies.csv")
    with open(output_path, "w", newline='') as csvfile:
        fieldnames = ['epoch', 'batch', 'train_accuracy', 'test_accuracy', 'generalization_gap']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for entry in batchwise_log:
            writer.writerow({
                'epoch': entry[0],
                'batch': entry[1],
                'train_accuracy': entry[2],
                'test_accuracy': entry[3],
                'generalization_gap': entry[4]  # ✅ gap 追加
            })


if __name__ == "__main__":
    train(total_epoch=40)