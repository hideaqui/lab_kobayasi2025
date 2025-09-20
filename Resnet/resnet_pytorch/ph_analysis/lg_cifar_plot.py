import matplotlib.pyplot as plt
import pandas as pd

# 🔹 ファイルのパス（適宜変更してください）
file_path = "/Users/hide/卒業研究/resnet_pytorch/output_cifar10/20250126_170439_battiseisokuoff/regression_results_fc_weight_512x10_persistence_1to60.csv"

# 🔹 CSV データの読み込み
df = pd.read_csv(file_path)

# 🔹 プロットの作成
plt.figure(figsize=(8, 5))
plt.scatter(df["epoch"], df["actual_gap"], label="Actual Gap", marker="o", alpha=0.7)
plt.plot(df["epoch"], df["predicted_gap"], label="Predicted Gap", linestyle="--", color="red")

# 🔹 軸ラベルとタイトル
plt.xlabel("Epoch")
plt.ylabel("Generalization Gap")
plt.legend()
plt.title("Generalization Gap Prediction (Epoch 1-60)")
plt.grid()

# 🔹 グラフの表示
plt.show()