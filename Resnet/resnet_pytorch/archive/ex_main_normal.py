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

# モデルの重みを初期化する関数（正規分布初期化）
def initialize_weights(model, mode="normal"):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if mode == "normal":
                # 平均0, 標準偏差0.02の正規分布で初期化
                init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                # バイアスはゼロで初期化
                init.zeros_(m.bias)

# モデルの精度を評価する関数
def evaluate_accuracy(model, dataloader, device):
    model.eval()  # 評価モードに切り替え
    correct, total = 0, 0
    with torch.no_grad():  # 勾配計算を無効化
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(axis=1)  # 最大値のインデックスを予測ラベルとする
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total  # 精度を返す

 # 学習を行うメイン関数
def train(total_epoch: int = 20, mode="normal", seed=seed):
    # デバイス選択
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用中のデバイス: {device}")
    
    # データローダーの取得（MNIST）
    train_dataloader, test_dataloader = get_dataloader(root="data", batch_size=64)

    # ResNetモデルの取得とデバイスへの転送
    model = get_resnet(pretrained=False).to(device)
    
    # モデルの重みを初期化
    initialize_weights(model, mode=mode)

    # 損失関数と最適化手法の設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 学習過程の精度ログ用リスト
    accuracy_log, test_accuracy_log, generalization_gap = [], [], []
    
    # 中間層の活性化を保存する辞書
    activations = {}
    
    # フック関数を定義し、指定した層の出力を保存
    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.cpu().detach().numpy()
        return hook

    # layer1からlayer4までにフックを登録
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        layer.register_forward_hook(get_hook(layer_name))

    # 出力ディレクトリをseed付きで作成（timestampなし）
    output_dir = f"./output/{mode}_{str(seed)}"
    os.makedirs(output_dir, exist_ok=True)

    # エポック0の精度を評価（初期状態のモデル）
    train_acc_0 = evaluate_accuracy(model, train_dataloader, device)
    test_acc_0 = evaluate_accuracy(model, test_dataloader, device)

    # エポック0の各層の活性化を保存
    for layer_name, act in activations.items():
        np.save(os.path.join(output_dir, f"epoch_0_{layer_name}.npy"), act)
        
    # エポック0の各層の重みを保存
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

    # エポック0の精度をログに追加
    accuracy_log.append(train_acc_0)
    test_accuracy_log.append(test_acc_0)
    generalization_gap.append(train_acc_0 - test_acc_0)

    # 学習ループ開始
    for epoch in range(total_epoch):
        model.train()  # 学習モードに切り替え
        # バッチごとに学習
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 勾配初期化
            out = model(images)    # 順伝播
            loss = criterion(out, labels)  # 損失計算
            loss.backward()        # 逆伝播
            optimizer.step()       # パラメータ更新
        scheduler.step()           # 学習率更新

        # 学習後に訓練データとテストデータの精度を評価
        train_acc = evaluate_accuracy(model, train_dataloader, device)
        test_acc = evaluate_accuracy(model, test_dataloader, device)
        accuracy_log.append(train_acc)
        test_accuracy_log.append(test_acc)
        generalization_gap.append(train_acc - test_acc)

        # 各層の活性化を保存
        for layer_name, act in activations.items():
            np.save(os.path.join(output_dir, f"epoch_{epoch+1}_{layer_name}.npy"), act)

        # 各層の重みを保存
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

    # 学習結果のログと出力ディレクトリを返す
    return accuracy_log, test_accuracy_log, generalization_gap, output_dir

 # エポックごとの精度をCSVに保存する関数
def save_epoch_accuracies():
    accuracy_log, test_accuracy_log, generalization_gap, output_dir = train(total_epoch=20, mode="normal", seed=seed)
    output_path = os.path.join(output_dir, "epoch_accuracies_normal.csv")
    # CSVファイルに書き込み
    with open(output_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        # ヘッダー行
        writer.writerow(['epoch','train_accuracy','test_accuracy','generalization_gap'])
        # 各エポックの精度を行として書き込み
        for epoch in range(len(accuracy_log)):
            writer.writerow([epoch, accuracy_log[epoch], test_accuracy_log[epoch], generalization_gap[epoch]])
    print(f"Epoch accuracies saved in {output_path}")

# スクリプト実行時のエントリーポイント
if __name__ == "__main__":
    save_epoch_accuracies()