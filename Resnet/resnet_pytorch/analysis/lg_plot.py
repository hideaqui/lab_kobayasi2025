import pandas as pd
import matplotlib.pyplot as plt
import os

# CSVファイルのパス
csv_path = "/Users/hide/lab_kobayasi2025/output/uniform_20251001_004321/regression_results_dimwise.csv"

# データを読み込む
df = pd.read_csv(csv_path)

# プロット
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["actual_gap"], label="Actual Gap", marker="o", linestyle="-", color="blue")
plt.plot(df["epoch"], df["predicted_gap"], label="Predicted Gap", marker="s", linestyle="--", color="red")

plt.xlabel("Epoch")
plt.ylabel("Generalization Gap")
plt.title("Actual vs. Predicted Generalization Gap")
plt.legend()
plt.grid(True)

# 保存
output_plot_path = os.path.join(os.path.dirname(csv_path), "regression_plot.png")
plt.savefig(output_plot_path)
print(f"✅ プロットを保存しました: {output_plot_path}")

# プロットを表示
plt.show()