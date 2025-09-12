import os
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser

# データディレクトリ
base_dir = "/Users/hide/卒業研究/resnet_pytorch/output_cifar10/20250127_134525"
epochs = [20, 40, 59]  # プロットするエポック
layers = ["layer0", "layer1", "layer2", "layer3", "layer4"]

# エポックごとの b1 の平均を格納する辞書
b1_means_dict = {epoch: [] for epoch in epochs}

# 各エポック・層の `npy` ファイルを処理
for epoch in epochs:
    for layer in layers:
        npy_path = os.path.join(base_dir, f"epoch_{epoch}_{layer}_label0.npy")

        if not os.path.exists(npy_path):
            print(f"⚠️ ファイルが見つかりません: {npy_path}")
            b1_means_dict[epoch].append(None)
            continue

        # データロード
        activations = np.load(npy_path)

        # 2次元に変換 (Reshape)
        num_samples = activations.shape[0]
        feature_dim = np.prod(activations.shape[1:])  # Flatten
        activations_reshaped = activations.reshape(num_samples, feature_dim)

        # パーシステントホモロジーを計算
        diagrams = ripser(activations_reshaped, maxdim=1)['dgms']

        # b1 の平均を計算
        b1_lifetimes = [d - b for b, d in diagrams[1] if np.isfinite(d)]  # 生存時間
        b1_mean = np.mean(b1_lifetimes) if b1_lifetimes else 0
        b1_means_dict[epoch].append(b1_mean)

        print(f"✅ Epoch {epoch}, {layer}: 平均 b1 = {b1_mean:.4f}")

# プロット
plt.figure(figsize=(8, 5))

for epoch in epochs:
    plt.plot(layers, b1_means_dict[epoch], marker='o', linestyle='-', label=f"Epoch {epoch}")

plt.xlabel("Layer")
plt.ylabel("Average $b_1$")
plt.title("Persistent Homology - Average $b_1$ Across Layers")
plt.legend()
plt.grid()
plt.show()