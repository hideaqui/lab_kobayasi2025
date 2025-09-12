import os
import pandas as pd
import matplotlib.pyplot as plt

# 🔹 フォント設定（日本語の文字化け対策）
plt.rcParams["font.family"] = "Hiragino Maru Gothic Pro"  # Mac
# plt.rcParams["font.family"] = "Meiryo"  # Windows
# plt.rcParams["font.family"] = "IPAexGothic"  # Linux

# データの保存ディレクトリ
output_dir = "/Users/hide/卒業研究/resnet_pytorch/output_cifar10/20250126_170439_battiseisokuoff"
accuracy_csv_path = os.path.join(output_dir, "epoch_accuracies.csv")

# **データの読み込み**
accuracy_df = pd.read_csv(accuracy_csv_path)

# **エポック数と Generalization Gap を抽出**
epochs = accuracy_df["epoch"]
generalization_gap = accuracy_df["generalization_gap"]

# **可視化**
plt.figure(figsize=(8, 5))
plt.plot(epochs, generalization_gap, marker="o", linestyle="-", color="blue", label="汎化ギャップ")

plt.xlabel("エポック数")
plt.ylabel("汎化ギャップ")

plt.legend()
plt.grid()

plt.show()