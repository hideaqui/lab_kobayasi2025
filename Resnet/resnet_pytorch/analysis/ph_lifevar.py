# パーシステンスライフ分散解析（CIFAR-10用）
# 元: used/bunnsann2.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ripser import ripser
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Hiragino Maru Gothic Pro"  # Mac
output_dir = "../../output_cifar10/20250126_170439_battiseisokuoff"
epoch_range = range(1, 71)

h0_persistence_variances = []
h1_persistence_variances = []
valid_epochs = []

print("🔹 Extracting PH persistence life variances...")
for epoch in tqdm(epoch_range, desc="Processing epochs", unit="epoch"):
    activation_file = os.path.join(output_dir, f"epoch_{epoch}_layer4.npy")
    if not os.path.exists(activation_file):
        print(f"⚠️ File not found for epoch {epoch}: {activation_file}")
        continue
    activations = np.load(activation_file).reshape(1024, -1)
    valid_rows = np.all(np.isfinite(activations), axis=1)
    activations_filtered = activations[valid_rows, :]
    valid_cols = np.all(np.isfinite(activations_filtered), axis=0)
    activations_filtered = activations_filtered[:, valid_cols]
    if activations_filtered.shape[0] == 0 or activations_filtered.shape[1] == 0:
        print(f"⚠️ All rows/cols removed due to NaN/Inf at epoch {epoch}")
        continue
    corr_matrix = np.corrcoef(activations_filtered)
    distance_matrix = 1 - np.abs(corr_matrix)
    diagrams = ripser(distance_matrix, maxdim=1, metric="precomputed")["dgms"]
    h0_lifetimes = [d - b for b, d in diagrams[0] if np.isfinite(d)]
    h1_lifetimes = [d - b for b, d in diagrams[1] if np.isfinite(d)]
    h0_var = np.var(h0_lifetimes) if len(h0_lifetimes) > 0 else 0.0
    h1_var = np.var(h1_lifetimes) if len(h1_lifetimes) > 0 else 0.0
    h0_persistence_variances.append(h0_var)
    h1_persistence_variances.append(h1_var)
    valid_epochs.append(epoch)

plt.figure(figsize=(8, 5))
plt.plot(valid_epochs, h0_persistence_variances, label="H0 のパーシステンスライフの分散", marker="o", linestyle="-")
plt.plot(valid_epochs, h1_persistence_variances, label="H1 のパーシステンスライフの分散", marker="s", linestyle="--")
plt.xlabel("エポック数")
plt.ylabel("パーシステンスの分散")
plt.title("エポックごとの H0 と H1 のパーシステンスの分散の変化")
plt.legend()
plt.grid()
plt.show()
