import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ripser import ripser

# 🔹 データの保存ディレクトリ
output_dir = "/Users/hide/卒業研究/resnet_pytorch/output_cifar10/20250126_170439_battiseisokuoff"

# **解析するエポックリスト**
selected_epochs = [1, 10, 20, 30, 40, 50, 60]  # 表示したいエポック

# 🔹 パーシステントダイアグラムをエポックごとにプロット
def plot_persistence_diagrams(epoch_list, num_cols=3):
    """エポックごとのパーシステントダイアグラムを分割表示"""
    num_epochs = len(epoch_list)
    num_rows = (num_epochs + num_cols - 1) // num_cols  # 行数を計算
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes = axes.flatten()

    for idx, epoch in enumerate(epoch_list):
        activation_file = os.path.join(output_dir, f"epoch_{epoch}_layer4.npy")

        if not os.path.exists(activation_file):
            print(f"⚠️ File not found for epoch {epoch}: {activation_file}")
            continue

        # 🔹 活性化データをロード
        activations = np.load(activation_file).reshape(1024, -1)  # (1024, 512)

        # 🔹 パーシステントホモロジー解析
        diagrams = ripser(activations, maxdim=1)['dgms']

        # 🔹 プロット
        ax = axes[idx]
        if len(diagrams[0]) > 0:
            ax.scatter(diagrams[0][:, 0], diagrams[0][:, 1], label="H0 (Connected Components)", color='blue', alpha=0.6)
        if len(diagrams[1]) > 0:
            ax.scatter(diagrams[1][:, 0], diagrams[1][:, 1], label="H1 (Loops)", color='red', alpha=0.6)
        
        ax.plot([0, 1], [0, 1], 'k--')  # 対角線
        ax.set_xlim([np.min(activations), np.max(activations)])  # X軸スケールを調整
        ax.set_ylim([np.min(activations), np.max(activations)])  # Y軸スケールを調整
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(f"Epoch {epoch}")
        ax.legend()

    # 不要な空白プロットを削除
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# 🔹 実行（エポックごとにパーシステントダイアグラムを表示）
plot_persistence_diagrams(selected_epochs)