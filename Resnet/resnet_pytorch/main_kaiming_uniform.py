import random
# seedの設定###########################################
seed = 1008
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
########################################################

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
def evaluate_accuracy(model, dataloader, device):
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
    train_dataloader, test_dataloader = get_dataloader(root="data", batch_size=64)

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

    for layer_name, act in activations.items():
        np.save(os.path.join(output_dir, f"epoch_0_{layer_name}.npy"), act)

    # 各層の重み保存
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'fc']
    for layer_name in layer_names:
        layer = getattr(model, layer_name, None)
        if layer is not None:
            if isinstance(layer, nn.Sequential):
                weights = {}
                for name, module in layer.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        weights[name] = module.weight.cpu().detach().numpy()
                np.savez(os.path.join(output_dir, f"epoch_0_{layer_name}_weights.npz"), **weights)
            else:
                if hasattr(layer, 'weight'):
                    np.save(os.path.join(output_dir, f"epoch_0_{layer_name}_weights.npy"),
                            layer.weight.cpu().detach().numpy())

    # 精度ログ更新
    accuracy_log.append(train_acc_0)
    test_accuracy_log.append(test_acc_0)
    generalization_gap.append(train_acc_0 - test_acc_0)

    # 学習ループ
    for epoch in range(total_epoch):
        model.train()
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 評価
        train_acc = evaluate_accuracy(model, train_dataloader, device)
        test_acc = evaluate_accuracy(model, test_dataloader, device)
        accuracy_log.append(train_acc)
        test_accuracy_log.append(test_acc)
        generalization_gap.append(train_acc - test_acc)

        # 活性化と重み保存
        for layer_name, act in activations.items():
            np.save(os.path.join(output_dir, f"epoch_{epoch+1}_{layer_name}.npy"), act)

        for layer_name in layer_names:
            layer = getattr(model, layer_name, None)
            if layer is not None:
                if isinstance(layer, nn.Sequential):
                    weights = {}
                    for name, module in layer.named_modules():
                        if isinstance(module, (nn.Conv2d, nn.Linear)):
                            weights[name] = module.weight.cpu().detach().numpy()
                    np.savez(os.path.join(output_dir, f"epoch_{epoch+1}_{layer_name}_weights.npz"), **weights)
                else:
                    if hasattr(layer, 'weight'):
                        np.save(os.path.join(output_dir, f"epoch_{epoch+1}_{layer_name}_weights.npy"),
                                layer.weight.cpu().detach().numpy())

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