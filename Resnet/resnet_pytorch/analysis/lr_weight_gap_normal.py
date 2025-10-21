import csv
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser

# === データフォルダ ===
repo_root = Path(__file__).resolve().parents[2]
output_dir = repo_root / "output" / "normal_1015"
pattern = str(output_dir / "epoch_1_batch_*_weights.npz")
acc_path = output_dir / "epoch_accuracies_normal.csv"

if not output_dir.exists():
    print(f"⚠️ 出力ディレクトリが見つかりません: {output_dir}")

if acc_path.exists():
    try:
        with acc_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            generalization_gaps = [float(row["generalization_gap"]) for row in reader if "generalization_gap" in row]
        print(f"📈 汎化ギャップCSVを読み込みました ({len(generalization_gaps)} 件): {acc_path}")
    except Exception as exc:
        print(f"⚠️ 汎化ギャップCSVの読み込みに失敗しました: {acc_path} ({exc})")
else:
    print(f"⚠️ 汎化ギャップCSVが見つかりません: {acc_path}")

# === ファイルリスト ===
files = sorted(glob.glob(pattern), key=lambda x: int(Path(x).stem.split("_batch_")[1].split("_")[0]))

if len(files) < 2:
    print("⚠️ 解析には少なくとも 2 つの重みファイルが必要です。処理を終了します。")
    sys.exit(0)

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
