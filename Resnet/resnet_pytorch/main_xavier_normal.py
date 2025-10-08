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

def initialize_weights(model, mode="xavier_normal"):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if mode == "xavier_normal":
                init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

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

def train(total_epoch: int = 20, mode="xavier_normal"):
    # 1. CUDA(GPU)が利用可能かチェックして最優先で使用
    # 2. CUDAが使えない場合はMPS、それも使えない場合はCPUを使用
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用中のデバイス: {device}")
    train_dataloader, test_dataloader = get_dataloader(root="data", batch_size=64)

    model = get_resnet(pretrained=False).to(device)
    initialize_weights(model, mode=mode)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    accuracy_log, test_accuracy_log, generalization_gap = [], [], []
    last_activation = None
    def hook(module, input, output):
        nonlocal last_activation
        last_activation = output.cpu().detach().numpy()
    model.layer4.register_forward_hook(hook)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./output/{mode}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # epoch 0
    train_acc_0 = evaluate_accuracy(model, train_dataloader, device)
    test_acc_0 = evaluate_accuracy(model, test_dataloader, device)

    np.save(os.path.join(output_dir, "epoch_0_layer4.npy"), last_activation)
    # 各層ごとの重みを保存（エポック0）
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'fc']
    for layer_name in layer_names:
        layer = getattr(model, layer_name, None)
        if layer is not None:
            weights = None
            # layer1〜layer4はSequentialの可能性があるため、重みをまとめる
            if isinstance(layer, nn.Sequential):
                weights = {}
                for name, module in layer.named_modules():
                    # Conv2dやLinearのweightを収集
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        weights[name] = module.weight.cpu().detach().numpy()
                # numpyに保存するためdictをnp.savez形式で保存
                np.savez(os.path.join(output_dir, f"epoch_0_{layer_name}_weights.npz"), **weights)
            else:
                # fc層など単一の層の場合
                if hasattr(layer, 'weight'):
                    weights = layer.weight.cpu().detach().numpy()
                    np.save(os.path.join(output_dir, f"epoch_0_{layer_name}_weights.npy"), weights)

    accuracy_log.append(train_acc_0)
    test_accuracy_log.append(test_acc_0)
    generalization_gap.append(train_acc_0 - test_acc_0)

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

        # eval both train and test accuracies
        train_acc = evaluate_accuracy(model, train_dataloader, device)
        test_acc = evaluate_accuracy(model, test_dataloader, device)
        accuracy_log.append(train_acc)
        test_accuracy_log.append(test_acc)
        generalization_gap.append(train_acc - test_acc)

        np.save(os.path.join(output_dir, f"epoch_{epoch+1}_layer4.npy"), last_activation)

        # 各層ごとの重みを保存
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'fc']
        for layer_name in layer_names:
            layer = getattr(model, layer_name, None)
            if layer is not None:
                weights = None
                # layer1〜layer4はSequentialの可能性があるため、重みをまとめる
                if isinstance(layer, nn.Sequential):
                    weights = {}
                    for name, module in layer.named_modules():
                        # Conv2dやLinearのweightを収集
                        if isinstance(module, (nn.Conv2d, nn.Linear)):
                            weights[name] = module.weight.cpu().detach().numpy()
                    # numpyに保存するためdictをnp.savez形式で保存
                    np.savez(os.path.join(output_dir, f"epoch_{epoch+1}_{layer_name}_weights.npz"), **weights)
                else:
                    # fc層など単一の層の場合
                    if hasattr(layer, 'weight'):
                        weights = layer.weight.cpu().detach().numpy()
                        np.save(os.path.join(output_dir, f"epoch_{epoch+1}_{layer_name}_weights.npy"), weights)

    return accuracy_log, test_accuracy_log, generalization_gap, output_dir

def save_epoch_accuracies():
    accuracy_log, test_accuracy_log, generalization_gap, output_dir = train(total_epoch=20, mode="xavier_normal")
    output_path = os.path.join(output_dir, "epoch_accuracies_xavier_normal.csv")
    with open(output_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch','train_accuracy','test_accuracy','generalization_gap'])
        for epoch in range(len(accuracy_log)):
            writer.writerow([epoch, accuracy_log[epoch], test_accuracy_log[epoch], generalization_gap[epoch]])
    print(f"Epoch accuracies saved in {output_path}")

if __name__ == "__main__":
    save_epoch_accuracies()