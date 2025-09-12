# PH特徴量と汎化ギャップの回帰分析スクリプト
# 元: used/lg_mid_pers.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from ripser import ripser

output_dir = "../../output_cifar10/20250126_170439_battiseisokuoff"
csv_path = os.path.join(output_dir, "epoch_accuracies.csv")

df_log = pd.read_csv(csv_path)
df_log = df_log[["epoch", "generalization_gap"]]

features = []
valid_epochs = []
print("🔹 Extracting PH features for regression...")
for epoch in tqdm(df_log["epoch"].values, desc="Processing epochs", unit="epoch"):
    file_path = os.path.join(output_dir, f"epoch_{epoch}_layer4.npy")
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        continue
    activations = np.load(file_path).reshape(1024, -1)
    valid_rows = np.all(np.isfinite(activations), axis=1)
    activations = activations[valid_rows, :]
    valid_cols = np.all(np.isfinite(activations), axis=0)
    activations = activations[:, valid_cols]
    if activations.size == 0:
        print(f"⚠️ No valid data after NaN/Inf removal for epoch {epoch}")
        continue
    corr_matrix = np.corrcoef(activations, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0)
    diagrams = ripser(distance_matrix, maxdim=1, metric="precomputed")["dgms"]
    h0_lifetimes = [d - b for b, d in diagrams[0] if np.isfinite(d)]
    h1_lifetimes = [d - b for b, d in diagrams[1] if np.isfinite(d)]
    h0_mean = np.mean(h0_lifetimes) if len(h0_lifetimes) > 0 else 0.0
    h1_mean = np.mean(h1_lifetimes) if len(h1_lifetimes) > 0 else 0.0
    h1_midlife = np.median([(b + d) / 2 for b, d in diagrams[1] if np.isfinite(d)]) if len(h1_lifetimes) > 0 else 0.0
    features.append([h0_mean, h1_mean, h1_midlife])
    valid_epochs.append(epoch)

df_features = pd.DataFrame(features, columns=["H0_mean", "H1_mean", "H1_midlife"])
df_features["epoch"] = valid_epochs
df_final = df_log.merge(df_features, on="epoch")

X = df_final[["H0_mean", "H1_mean", "H1_midlife"]]
y = df_final["generalization_gap"]
reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"✅ R²: {r2:.4f}, RMSE: {rmse:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, color="blue", alpha=0.6, label="予測値 vs 実測値")
plt.plot([min(y), max(y)], [min(y), max(y)], "--", color="red", label="理想線 (y=x)")
plt.xlabel("実測値（汎化ギャップ）")
plt.ylabel("予測値（回帰モデル）")
plt.title("汎化ギャップの回帰分析")
plt.legend()
plt.grid()
plt.show()
