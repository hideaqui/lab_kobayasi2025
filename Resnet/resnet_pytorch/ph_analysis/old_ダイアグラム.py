import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ripser import ripser

# データの保存ディレクトリ
output_dir = "../../output_cifar10/20250126_170439_battiseisokuoff"
epoch_range = [1, 5, 10, 20, 30, 40, 50, 60,]  # 表示するエポック

# 🔹 日本語フォントの設定（環境に応じて変更）
plt.rcParams["font.family"] = "Hiragino Sans"  # Mac

# 🔹 PH の計算結果を保存
ph_results = {}

print("🔹 Extracting PH persistence diagrams...")

for epoch in tqdm(epoch_range, desc="Processing epochs", unit="epoch"):
    file_path = os.path.join(output_dir, f"epoch_{epoch}_layer4.npy")
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        continue

    # **活性化データをロード**
    activations = np.load(file_path).reshape(1024, -1)

    # `NaN` や `Inf` を含む行・列を削除
    valid_rows = np.all(np.isfinite(activations), axis=1)
    activations = activations[valid_rows, :]
    valid_cols = np.all(np.isfinite(activations), axis=0)
    activations = activations[:, valid_cols]

    if activations.size == 0:
        print(f"⚠️ No valid data after NaN/Inf removal for epoch {epoch}")
        continue

    # **相関行列を計算**
    corr_matrix = np.corrcoef(activations, rowvar=False)

    # **NaN をゼロに置き換え（計算の安定化）**
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # **相関行列から距離行列を作成**
    distance_matrix = 1 - np.abs(corr_matrix)
    
    # **対角成分を明示的に 0 にする**
    np.fill_diagonal(distance_matrix, 0)

    # **デバッグ: 距離行列の統計情報を表示**
    print(f"Epoch {epoch} の distance_matrix:")
    print("最小値:", np.min(distance_matrix))
    print("最大値:", np.max(distance_matrix))
    print("NaN の数:", np.isnan(distance_matrix).sum())

    # **PH を計算**
    diagrams = ripser(distance_matrix, maxdim=1, metric="precomputed")["dgms"]
    ph_results[epoch] = diagrams

    # **H1の確認**
    print(f"✅ Epoch {epoch}: H0={len(diagrams[0])}, H1={len(diagrams[1])}")
print("activations shape:", activations.shape)
print(f"✅ Successfully computed PH for {len(ph_results)} epochs")

# 🔹 ダイアグラムの可視化（2行4列の分割表示）
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, epoch in enumerate(sorted(ph_results.keys())):
    ax = axes[i]
    diagrams = ph_results[epoch]

    # **H0とH1を明示的に表示**
    if len(diagrams[0]) > 0:
        ax.scatter(diagrams[0][:, 0], diagrams[0][:, 1], label="H0", color="blue", s=20, alpha=0.6)
    if len(diagrams[1]) > 0:
        ax.scatter(diagrams[1][:, 0], diagrams[1][:, 1], label="H1", color="orange", s=20, alpha=0.6)

    # **軸の範囲を固定**
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # **対角線 y=x を追加**
    ax.plot([0, 1], [0, 1], "--", color="gray")

    ax.set_title(f"エポック {epoch}")
    ax.legend()

plt.tight_layout()
plt.show()