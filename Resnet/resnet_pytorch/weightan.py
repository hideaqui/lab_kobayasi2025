import os
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

# ======================================
# 設定
# ======================================
base_dir = r"./output"
methods = [
    "uniform_1009",
    "normal_1009",
    "xavier_uniform_1009",
    "xavier_normal_1009",
    "kaiming_uniform_1009",
    "kaiming_normal_1009"
]
labels = [
    "Uniform",
    "Normal",
    "Xavier Uniform",
    "Xavier Normal",
    "Kaiming Uniform",
    "Kaiming Normal"
]

layer_name = "layer1"
max_epoch = 10   # 表示するエポック数（例: 0〜10）

# ======================================
# 6つの初期化法を比較
# ======================================
fig, axes = plt.subplots(len(methods), max_epoch + 1, figsize=(3*(max_epoch+1), 3*len(methods)))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for i, (method, label) in enumerate(zip(methods, labels)):
    print(f"🧩 {label} を解析中...")
    output_dir = os.path.join(base_dir, method)

    for epoch in range(max_epoch + 1):
        weight_path = os.path.join(output_dir, f"epoch_{epoch}_{layer_name}_weights.npz")
        ax = axes[i, epoch] if len(methods) > 1 else axes[epoch]

        if not os.path.exists(weight_path):
            ax.axis("off")
            ax.set_title(f"Epoch {epoch}\n(No data)")
            continue

        # 重みデータを読み込み
        data = np.load(weight_path)
        all_filters = []

        for key in data.files:
            W = data[key]  # (out_ch, in_ch, kH, kW)
            W_flat = W.reshape(W.shape[0], -1)
            all_filters.append(W_flat)

        all_filters = np.concatenate(all_filters, axis=0)

        # RipserでPH解析
        diagrams = ripser(all_filters, maxdim=1)['dgms']
        plot_diagrams(diagrams, ax=ax, show=False)
        ax.set_title(f"Epoch {epoch}")

    axes[i, 0].set_ylabel(label, fontsize=10)

plt.suptitle(f"Persistent Homology of Layer1 weights across initializations", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
