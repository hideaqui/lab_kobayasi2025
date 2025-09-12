import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSVファイルのパス
file_path = "/Users/hide/卒業研究/resnet_pytorch/output_6000/20250131_004004/epoch_accuracies.csv"

# データを読み込む
df = pd.read_csv(file_path)

# train_accuracy と test_accuracy を変換 (-log(1-x))
df["train_transformed"] = -np.log(1 - df["train_accuracy"])
df["test_transformed"] = -np.log(1 - df["test_accuracy"])

# gap を計算し、-log(x) に変換
df["gap"] = np.abs(df["train_accuracy"] - df["test_accuracy"])
df["gap_transformed"] = -np.log(df["gap"])

# 結果を可視化
plt.figure(figsize=(10, 6))
plt.plot(df["train_transformed"], label="Train (-log(1-x))")
plt.plot(df["test_transformed"], label="Test (-log(1-x))")
plt.plot(df["gap_transformed"], label="Gap (-log(x))", linestyle="dashed")
plt.xlabel("Epoch")
plt.ylabel("Transformed Accuracy")
plt.title("Transformed Accuracy Trends")
plt.legend()
plt.grid()
plt.show()