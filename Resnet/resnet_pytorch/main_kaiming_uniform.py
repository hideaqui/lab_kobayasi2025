import random
import os
import csv
import numpy as np
import torch
import torch.nn.init as init
from torch import nn, optim
from tqdm import tqdm
from datetime import datetime
from dataset.mnist import get_dataloader
from model.resnet import get_resnet

# GPU最適化設定（RTX対応）
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# seedの設定###########################################
seed = 1009
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
########################################################

# モデルの重みを初期化する関数（Kaiming一様分布初期化）
def initialize_weights(model, mode="kaiming_uniform"):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if mode == "kaiming_uniform":
                # He(Kaiming)の一様分布初期化（ReLU向け）
                init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                init.zeros_(m.bias)

# モデルの精度を評価する関数
@torch.inference_mode()
def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(axis=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total

def save_weights_and_activations(model, activations, output_dir, epoch, layer_names):
    # 活性化保存
    for layer_name, act in activations.items():
        np.save(os.path.join(output_dir, f"epoch_{epoch}_{layer_name}.npy"), act)

    # 重み保存
    for layer_name in layer_names:
        layer = getattr(model, layer_name, None)
        if layer is not None:
            if isinstance(layer, nn.Sequential):
                weights = {}
                for name, module in layer.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        weights[name] = module.weight.cpu().detach().numpy()
                np.savez(os.path.join(output_dir, f"epoch_{epoch}_{layer_name}_weights.npz"), **weights)
            else:
                if hasattr(layer, 'weight'):
                    np.save(os.path.join(output_dir, f"epoch_{epoch}_{layer_name}_weights.npy"),
                            layer.weight.cpu().detach().numpy())

# 学習を行うメイン関数
def train(total_epoch: int = 20, mode="kaiming_uniform", seed=seed):
    # デバイス選択
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用中のデバイス: {device}")

    # データロード
    train_dataloader, test_dataloader = get_dataloader(root="data", batch_size=256, num_workers=4, seed=seed)

    # モデルを作成して初期化
    model = get_resnet(pretrained=False).to(device)
    initialize_weights(model, mode=mode)

    # 損失関数・最適化
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # ログ用リスト
    accuracy_log, test_accuracy_log, generalization_gap = [], [], []

    # 中間層の出力保存用
    activations = {}

    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.cpu().detach().numpy()
        return hook

    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        getattr(model, layer_name).register_forward_hook(get_hook(layer_name))

    # 出力フォルダ
    output_dir = f"./output/{mode}_{str(seed)}"
    os.makedirs(output_dir, exist_ok=True)

    # 初期状態の精度
    train_acc_0 = evaluate_accuracy(model, train_dataloader, device)
    test_acc_0 = evaluate_accuracy(model, test_dataloader, device)

    layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'fc']

    save_weights_and_activations(model, activations, output_dir, 0, layer_names)

    # 精度ログ更新
    accuracy_log.append(train_acc_0)
    test_accuracy_log.append(test_acc_0)
    generalization_gap.append(train_acc_0 - test_acc_0)

    print(f"Epoch 0 - Loss: N/A, Train Acc: {train_acc_0:.4f}, Test Acc: {test_acc_0:.4f}")

    # 学習ループ
    for epoch in range(total_epoch):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epoch}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()

        epoch_loss = running_loss / len(train_dataloader.dataset)

        # 評価
        train_acc = evaluate_accuracy(model, train_dataloader, device)
        test_acc = evaluate_accuracy(model, test_dataloader, device)
        accuracy_log.append(train_acc)
        test_accuracy_log.append(test_acc)
        generalization_gap.append(train_acc - test_acc)

        save_weights_and_activations(model, activations, output_dir, epoch+1, layer_names)

        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    return accuracy_log, test_accuracy_log, generalization_gap, output_dir

# エポックごとの精度をCSVに保存
def save_epoch_accuracies():
    accuracy_log, test_accuracy_log, generalization_gap, output_dir = train(total_epoch=20, mode="kaiming_uniform", seed=seed)
    output_path = os.path.join(output_dir, "epoch_accuracies_kaiming_uniform.csv")
    with open(output_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch','train_accuracy','test_accuracy','generalization_gap'])
        for epoch in range(len(accuracy_log)):
            writer.writerow([epoch, accuracy_log[epoch], test_accuracy_log[epoch], generalization_gap[epoch]])
    print(f"Epoch accuracies saved in {output_path}")

if __name__ == "__main__":
    save_epoch_accuracies()