import os, csv, numpy as np, torch
import torch.nn.init as init
from torch import nn, optim
from tqdm import tqdm
from datetime import datetime
from dataset.mnist import get_dataloader
from model.resnet import get_resnet

def initialize_weights(model, mode="kaiming_uniform"):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if mode == "kaiming_uniform":
                init.kaiming_uniform_(m.weight, nonlinearity="relu")
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

def train(total_epoch: int = 20, mode="kaiming_uniform"):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
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

    model.eval()
    with torch.no_grad():
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(axis=1)
            train_acc_0 = (preds == labels).sum().item() / labels.size(0)
            break
    test_acc_0 = evaluate_accuracy(model, test_dataloader, device)

    np.save(os.path.join(output_dir, "epoch_0_layer4.npy"), last_activation)
    accuracy_log.append(train_acc_0)
    test_accuracy_log.append(test_acc_0)
    generalization_gap.append(train_acc_0 - test_acc_0)

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

        np.save(os.path.join(output_dir, f"epoch_{epoch+1}_layer4.npy"), last_activation)

    return accuracy_log, test_accuracy_log, generalization_gap, output_dir

def save_epoch_accuracies():
    accuracy_log, test_accuracy_log, generalization_gap, output_dir = train(total_epoch=20, mode="kaiming_uniform")
    output_path = os.path.join(output_dir, "epoch_accuracies_kaiming_uniform.csv")
    with open(output_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch','train_accuracy','test_accuracy','generalization_gap'])
        for epoch in range(len(accuracy_log)):
            writer.writerow([epoch, accuracy_log[epoch], test_accuracy_log[epoch], generalization_gap[epoch]])
    print(f"Epoch accuracies saved in {output_path}")

if __name__ == "__main__":
    # エポックごとの訓練精度・テスト精度を eval モードで全データ再評価して保存
    save_epoch_accuracies()