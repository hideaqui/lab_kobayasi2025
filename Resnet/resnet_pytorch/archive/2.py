import pandas as pd
import matplotlib.pyplot as plt

# CSV ファイルのパス
csv_path = "/Users/hide/卒業研究/resnet_pytorch/output_6000/20250126_021306/epoch_accuracies.csv"
# データの読み込み
df = pd.read_csv(csv_path)

# プロットの設定
plt.figure(figsize=(10, 6))

# 1. Train Accuracy vs. Test Accuracy
plt.subplot(2, 2, 1)
plt.plot(df["epoch"], df["train_accuracy"], label="Train Accuracy", marker="o", linestyle="-", color="blue")
plt.plot(df["epoch"], df["test_accuracy"], label="Test Accuracy", marker="o", linestyle="-", color="red")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train & Test Accuracy Over Epochs")
plt.legend()
plt.grid()

# 2. Generalization Gap
plt.subplot(2, 2, 2)
plt.plot(df["epoch"], df["generalization_gap"], label="Generalization Gap", marker="o", linestyle="-", color="purple")
plt.xlabel("Epoch")
plt.ylabel("Gap")
plt.title("Generalization Gap Over Epochs")
plt.legend()
plt.grid()

# 3. Train Loss
plt.subplot(2, 2, 3)
plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o", linestyle="-", color="green")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss Over Epochs")
plt.legend()
plt.grid()

# 4. Learning Rate
plt.subplot(2, 2, 4)
plt.plot(df["epoch"], df["learning_rate"], label="Learning Rate", marker="o", linestyle="-", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Over Epochs")
plt.legend()
plt.grid()

# プロットの表示
plt.tight_layout()
plt.show()