# 汎化ギャップ可視化スクリプト
# 元: used/gap.py
import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Hiragino Maru Gothic Pro"  # Mac
output_dir = "../../output_cifar10/20250126_170439_battiseisokuoff"
accuracy_csv_path = os.path.join(output_dir, "epoch_accuracies.csv")

accuracy_df = pd.read_csv(accuracy_csv_path)
epochs = accuracy_df["epoch"]
generalization_gap = accuracy_df["generalization_gap"]

plt.figure(figsize=(8, 5))
plt.plot(epochs, generalization_gap, marker="o", linestyle="-", color="blue", label="汎化ギャップ")
plt.xlabel("エポック数")
plt.ylabel("汎化ギャップ")
plt.legend()
plt.grid()
plt.show()
