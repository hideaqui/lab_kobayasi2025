import random
import os
import csv
import numpy as np
import torch
import torch.nn.init as init
from torch import nn, optim
from tqdm import tqdm
from dataset.mnist import get_dataloader
from model.resnet import get_resnet


# ============================================================
# GPU最適化設定（RTX対応）
# ============================================================
seed = 1015
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.benchmark = True        # 入力サイズ固定で高速化
torch.backends.cudnn.deterministic = False   # 速度優先
torch.backends.cuda.matmul.allow_tf32 = True # Ampere以降で高速
torch.set_float32_matmul_precision('high')   # 高精度float32演算モード
# ============================================================


# ============================================================
# モデルの重みを初期化する関数（Xavier一様分布初期化）
# ============================================================
def initialize_weights(model, mode="xavier_uniform"):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if mode == "xavier_uniform":
                init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)


# ============================================================
# 精度評価関数（高速モード）
# ============================================================
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


# ============================================================
# 学習関数
# ============================================================
def train(total_epoch: int = 20, mode="xavier_uniform", seed=seed):
    # デバイス設定
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"使用中のデバイス: {device}")

    # データローダー取得（GPU最適化対応）
    train_dataloader, test_dataloader = get_dataloader(
        root="data",
        batch_size=256,
        num_workers=4,
        seed=seed
    )

    # モデル構築と初期化
    model = get_resnet(pretrained=False).to(device)
    initialize_weights(model, mode=mode)

    # 損失関数・最適化
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # ログ記録リスト
    accuracy_log, test_accuracy_log, generalization_gap = [], [], []

    # フック（中間層活性化記録）
    activations = {}

    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook

    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        getattr(model, layer_name).register_forward_hook(get_hook(layer_name))

    # 出力ディレクトリ
    output_dir = f"./output/{mode}_{seed}"
    os.makedirs(output_dir, exist_ok=True)

    # 初期精度評価（epoch 0）
    train_acc_0 = evaluate_accuracy(model, train_dataloader, device)
    test_acc_0 = evaluate_accuracy(model, test_dataloader, device)
    accuracy_log.append(train_acc_0)
    test_accuracy_log.append(test_acc_0)
    generalization_gap.append(train_acc_0 - test_acc_0)

    # 初期状態保存
    save_weights_and_activations(model, activations, output_dir, epoch=0, mode=mode)

    # ============================================================
    # 学習ループ
    # ============================================================
    for epoch in range(total_epoch):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epoch}")):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 新規追加: ミニバッチ毎に重みを保存
            weights_to_save = {}
            with torch.no_grad():
                for name, param in model.named_parameters():
                    weights_to_save[name] = param.detach().cpu().numpy()
            np.savez(os.path.join(output_dir, f"epoch_{epoch+1}_batch_{batch_idx}_weights.npz"), **weights_to_save)

        scheduler.step()

        # 評価フェーズ
        train_acc = evaluate_accuracy(model, train_dataloader, device)
        test_acc = evaluate_accuracy(model, test_dataloader, device)

        accuracy_log.append(train_acc)
        test_accuracy_log.append(test_acc)
        generalization_gap.append(train_acc - test_acc)

        print(f"Epoch [{epoch+1}/{total_epoch}] | Loss: {running_loss/len(train_dataloader):.4f} "
              f"| Train: {train_acc:.4f} | Test: {test_acc:.4f}")

        # 重みと活性化を保存
        save_weights_and_activations(model, activations, output_dir, epoch=epoch+1, mode=mode)

    return accuracy_log, test_accuracy_log, generalization_gap, output_dir


# ============================================================
# 重み・活性化保存関数
# ============================================================
def save_weights_and_activations(model, activations, output_dir, epoch, mode):
    # 活性化保存（activation名を明示）
    for layer_name, act in activations.items():
        np.save(os.path.join(output_dir, f"epoch_{epoch}_{layer_name}_activation_{mode}.npy"), act)

    # 各層の重み保存
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']:
        layer = getattr(model, layer_name, None)
        if layer is None:
            continue

        if isinstance(layer, nn.Sequential):
            weights = {
                name: mod.weight.detach().cpu().numpy()
                for name, mod in layer.named_modules()
                if isinstance(mod, (nn.Conv2d, nn.Linear))
            }
            np.savez(os.path.join(output_dir, f"epoch_{epoch}_{layer_name}_weights_{mode}.npz"), **weights)
        elif hasattr(layer, 'weight'):
            np.save(os.path.join(output_dir, f"epoch_{epoch}_{layer_name}_weights_{mode}.npy"),
                    layer.weight.detach().cpu().numpy())


# ============================================================
# 精度をCSVに保存
# ============================================================
def save_epoch_accuracies():
    accuracy_log, test_accuracy_log, generalization_gap, output_dir = train(total_epoch=20, mode="xavier_uniform", seed=seed)
    output_path = os.path.join(output_dir, "epoch_accuracies_xavier_uniform.csv")

    with open(output_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_accuracy', 'test_accuracy', 'generalization_gap'])
        for epoch in range(len(accuracy_log)):
            writer.writerow([epoch, accuracy_log[epoch], test_accuracy_log[epoch], generalization_gap[epoch]])

    print(f"✅ Epoch accuracies saved in {output_path}")


# ============================================================
# メイン処理
# ============================================================
if __name__ == "__main__":
    save_epoch_accuracies()