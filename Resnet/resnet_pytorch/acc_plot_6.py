import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# 🧩 フォント設定（Windows）
# ==============================
plt.rcParams['font.family'] = 'Meiryo'  # Macなら 'Hiragino Maru Gothic Pro'

# ==============================
# 📂 相対パス設定（スクリプトの位置から）
# ==============================
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output"))

# ==============================
# 🔹 初期化手法とラベル
# ==============================
methods = [
    ("uniform_1009", "Uniform"),
    ("normal_1009", "Normal"),
    ("xavier_uniform_1009", "Xavier Uniform"),
    ("xavier_normal_1009", "Xavier Normal"),
    ("kaiming_uniform_1009", "Kaiming Uniform"),
    ("kaiming_normal_1009", "Kaiming Normal"),
]

# ==============================
# 🎨 グラフ作成：2行×3列に並べて比較
# ==============================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, (folder, label) in enumerate(methods):
    folder_path = os.path.join(base_dir, folder)

    # CSVファイルの自動検出
    csv_files = [f for f in os.listdir(folder_path)
                 if f.startswith("epoch_accuracies_") and f.endswith(".csv")]
    if not csv_files:
        print(f"⚠️ CSVファイルが見つかりません: {folder_path}")
        continue

    csv_path = os.path.join(folder_path, csv_files[0])
    print(f"✅ 読み込み中: {csv_path}")

    # データ読み込み
    df = pd.read_csv(csv_path)
    df = df[(df["epoch"] >= 1) & (df["epoch"] <= 60)]

    ax = axes[i]

    # 訓練精度（青・点線）＋ テスト精度（赤・実線）
    ax.plot(df["epoch"], df["train_accuracy"], linestyle="--", color="blue", label="訓練精度")
    ax.plot(df["epoch"], df["test_accuracy"], linestyle="-", color="red", label="テスト精度")

    ax.set_title(label, fontsize=12)
    ax.set_xlabel("エポック数")
    ax.set_ylabel("精度")
    ax.set_xticks(range(0, 20, 5))
    ax.set_ylim(0.9, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")

# ==============================
# 🔹 レイアウトと保存
# ==============================
plt.suptitle("6つの初期化手法ごとの学習・テスト精度比較（MNIST, ResNet）", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 💾 保存
save_path = os.path.join(base_dir, "accuracy_comparison_grid.png")
plt.savefig(save_path, dpi=300)
print(f"✅ 画像を保存しました: {save_path}")

plt.show()
