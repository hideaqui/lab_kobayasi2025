import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser

# パスリスト（各レイヤーの `.npy` ファイル）
file_paths = [
    "/Users/hide/卒業研究/resnet_pytorch/output_cifar10/20250127_134525/epoch_59_layer0_label0.npy",
    "/Users/hide/卒業研究/resnet_pytorch/output_cifar10/20250127_134525/epoch_59_layer1_label0.npy",
    "/Users/hide/卒業研究/resnet_pytorch/output_cifar10/20250127_134525/epoch_59_layer2_label0.npy",
    "/Users/hide/卒業研究/resnet_pytorch/output_cifar10/20250127_134525/epoch_59_layer3_label0.npy",
    "/Users/hide/卒業研究/resnet_pytorch/output_cifar10/20250127_134525/epoch_59_layer4_label0.npy",
]

# 各レイヤーのBetti-0 (b0) の平均を保存するリスト
b0_means = []
layer_names = [f"Layer {i}" for i in range(len(file_paths))]

# 各レイヤーのデータを処理
for path in file_paths:
    # `.npy` ファイルをロード（形状が (N, C, H, W) なら (N, C) に変換）
    activations = np.load(path)
    activations = activations.reshape(activations.shape[0], -1)  # 2次元 (N, C*H*W) に変換
    
    # リプサーでPH計算 (最大次元は 1 に設定)
    diagrams = ripser(activations, maxdim=1)['dgms']

    # Betti-0 (b0) の寿命 (death - birth) の平均を計算
    b0_lifetimes = [d - b for b, d in diagrams[0] if np.isfinite(d)]  # B0の生存時間
    b0_mean = np.mean(b0_lifetimes) if len(b0_lifetimes) > 0 else 0
    b0_means.append(b0_mean)

# **プロット**
plt.figure(figsize=(8, 5))
plt.plot(layer_names, b0_means, marker="o", linestyle="-", label="Mean Betti-0 Lifetime")

plt.xlabel("Layers")
plt.ylabel("Mean Betti-0 Lifetime")
plt.title("Change in Mean Betti-0 Across Layers")
plt.legend()
plt.grid()
plt.show()