import pandas as pd
import matplotlib.pyplot as plt

# 日本語フォントの設定（Mac用）
plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'

# CSV ファイルの読み込み
csv_path = "/Users/hide/卒業研究/resnet_pytorch/output_cifar10/20250126_170439_battiseisokuoff/epoch_accuracies.csv"
df = pd.read_csv(csv_path)

# エポック数 1 〜 60 のデータのみ抽出
df = df[(df["epoch"] >= 1) & (df["epoch"] <= 60)]

# 図の作成
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_accuracy"], label="訓練精度", marker="o")
plt.plot(df["epoch"], df["test_accuracy"], label="テスト精度", marker="s")

# 軸ラベル・タイトル
plt.xlabel("エポック数")
plt.ylabel("精度")
plt.xticks(range(1, 61, 5))  # 5エポックごとに目盛りを設定
plt.legend()
plt.grid()

# 図の表示
plt.show()