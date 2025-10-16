import numpy as np
import glob, os
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

# === データフォルダ ===
output_dir = "./output/normal_1015"
pattern = os.path.join(output_dir, "epoch_1_batch_*_weights.npz")

# === ファイルリスト ===
files = sorted(glob.glob(pattern), key=lambda x: int(x.split("_batch_")[1].split("_")[0]))

print(f"📂 読み込み対象 {len(files)} 件")

# === PH 特徴量保存用 ===
ph_features = []

# === ループ ===
for i in range(len(files) - 1):
    f1, f2 = files[i], files[i + 1]
    W1 = np.load(f1)
    W2 = np.load(f2)

    # fc層の重みを抽出（npz内の 'fc.weight'）
    if 'fc.weight' not in W1.files:
        print(f"⚠️ {f1} に fc.weight がありません")
        continue

    w1 = W1['fc.weight']
    w2 = W2['fc.weight']

    # === 差分 (ΔW) ===
    delta_w = w2 - w1
    flat = delta_w.flatten()

    # === 相関行列 ===
    corr = np.corrcoef(flat)
    if np.isnan(corr).any():
        corr = np.nan_to_num(corr, nan=0.0)

    # === 距離行列 ===
    dist = 1 - np.abs(corr)

    # === PH計算 ===
    dgms = ripser(dist, metric='precomputed', maxdim=1)['dgms']

    # === 特徴抽出（ライフタイム平均など） ===
    lifetimes = [d - b for b, d in dgms[1] if np.isfinite(d)]
    lambda_mean = np.mean(lifetimes) if len(lifetimes) > 0 else 0.0
    ph_features.append(lambda_mean)

    print(f"✅ Batch {i} → {i+1} | mean lifetime = {lambda_mean:.4f}")

# === 可視化 ===
plt.figure(figsize=(8,4))
plt.plot(range(len(ph_features)), ph_features, marker='o')
plt.xlabel("Batch index")
plt.ylabel("Mean persistence lifetime (H1)")
plt.title("PH trace from ΔW (fc layer, normal_1015)")
plt.grid()
plt.show()
