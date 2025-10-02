import os, csv, numpy as np, torch
import torch.nn.init as init
from torch import nn, optim
from tqdm import tqdm
from datetime import datetime
from dataset.mnist import get_dataloader #mnistの学習データを取得
from model.resnet import get_resnet
import random

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# seedの設定
seed = 1002
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def initialize_weights(model, mode="xavier_normal"):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if mode == "xavier_normal":
                init.uniform_(m.weight, a=-0.1, b=0.1)
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

def train(total_epoch: int = 20, mode="xavier_normal", seed = seed):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    train_dataloader, test_dataloader = get_dataloader(root="data", batch_size=64)

    model = get_resnet(pretrained=False).to(device)
    initialize_weights(model, mode=mode)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    accuracy_log, test_accuracy_log, generalization_gap = [], [], []
    
#各層の中間出力を記録#################################################################
    activation_1 = None
    def hook1(module, input, output):
        nonlocal activation_1
        activation_1 = output.cpu().detach().numpy()
    model.layer1.register_forward_hook(hook1)
    
    activation_2 = None
    def hook2(module, input, output):
        nonlocal activation_2
        activation_2 = output.cpu().detach().numpy()
    model.layer2.register_forward_hook(hook2)
    
    activation_3 = None
    def hook3(module, input, output):
        nonlocal activation_3
        activation_3 = output.cpu().detach().numpy()
    model.layer3.register_forward_hook(hook3)
    
    activation_4 = None
    def hook4(module, input, output):
        nonlocal activation_4
        activation_4 = output.cpu().detach().numpy()
    model.layer4.register_forward_hook(hook4)
####################################################################################3

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./output/{mode}_{str(seed)}"
    os.makedirs(output_dir, exist_ok=True)

    # エポック0の訓練精度を全データで再評価
    train_acc_0 = evaluate_accuracy(model, train_dataloader, device)
    test_acc_0 = evaluate_accuracy(model, test_dataloader, device)

    np.save(os.path.join(output_dir, "epoch_0_layer1.npy"), activation_1)
    np.save(os.path.join(output_dir, "epoch_0_layer2.npy"), activation_2)
    np.save(os.path.join(output_dir, "epoch_0_layer3.npy"), activation_3)
    np.save(os.path.join(output_dir, "epoch_0_layer4.npy"), activation_4)
    accuracy_log.append(train_acc_0)
    test_accuracy_log.append(test_acc_0)
    generalization_gap.append(train_acc_0 - test_acc_0)
    
    print(f"Epoch 0: Train Acc = {train_acc_0:.4f}, Test Acc = {test_acc_0:.4f}, Gap = {train_acc_0 - test_acc_0:.4f}")

    for epoch in range(total_epoch):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
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

        # エポック終了後に eval モードで全データに対して正答率を再評価
        train_acc = evaluate_accuracy(model, train_dataloader, device)
        test_acc = evaluate_accuracy(model, test_dataloader, device)
        accuracy_log.append(train_acc)
        test_accuracy_log.append(test_acc)
        generalization_gap.append(train_acc - test_acc)

        np.save(os.path.join(output_dir, f"epoch_{epoch+1}_layer1.npy"), activation_1)
        np.save(os.path.join(output_dir, f"epoch_{epoch+1}_layer2.npy"), activation_2)
        np.save(os.path.join(output_dir, f"epoch_{epoch+1}_layer3.npy"), activation_3)
        np.save(os.path.join(output_dir, f"epoch_{epoch+1}_layer4.npy"), activation_4)
        print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}, Gap = {train_acc - test_acc:.4f}")

    return accuracy_log, test_accuracy_log, generalization_gap, output_dir

def save_epoch_accuracies():
    accuracy_log, test_accuracy_log, generalization_gap, output_dir = train(total_epoch=10, mode="xavier_normal") #epoch数はここ
    output_path = os.path.join(output_dir, "epoch_accuracies_uniform.csv")
    with open(output_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch','train_accuracy','test_accuracy','generalization_gap'])
        for epoch in range(len(accuracy_log)):
            writer.writerow([epoch, accuracy_log[epoch], test_accuracy_log[epoch], generalization_gap[epoch]])
    print(f"Epoch accuracies saved in {output_path}")

if __name__ == "__main__":
    save_epoch_accuracies()