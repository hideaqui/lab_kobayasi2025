import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# === CSVファイルのパスを適宜修正してください ===
csv_path = "/Users/hide/卒業研究/resnet_pytorch/output/20250123_140605/regression_results_dimwise.csv"

# CSV を読み込み
df = pd.read_csv(csv_path)

# PCA の対象となる列（特徴量）を選択
# 例: 0次元,1次元,2次元 の λ, μ の平均・分散
feature_columns = [
    "dim0_lambda_mean", "dim0_lambda_var", "dim0_mu_mean", "dim0_mu_var",
    "dim1_lambda_mean", "dim1_lambda_var", "dim1_mu_mean", "dim1_mu_var",
    "dim2_lambda_mean", "dim2_lambda_var", "dim2_mu_mean", "dim2_mu_var"
]

# DataFrame から対象列のみ抽出
X = df[feature_columns].values

# 必要に応じて標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA を実行
# n_components を特に指定しない場合、特徴量数と同じになる
n_components = X.shape[1]  # 例: 12
pca = PCA(n_components=n_components)
pca.fit(X_scaled)

# 寄与率 (explained_variance_ratio_) を取得
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# 結果を表示
print("各主成分の寄与率:")
for i, ratio in enumerate(explained_variance_ratio, start=1):
    print(f"  PC{i}: {ratio:.4f}")
print("\n累積寄与率:")
for i, ratio in enumerate(cumulative_variance_ratio, start=1):
    print(f"  PC1～PC{i}まで: {ratio:.4f}")

# グラフで確認 (オプション)
plt.figure(figsize=(6,4))
plt.plot(range(1, n_components+1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.title("Cumulative Explained Variance Ratio by PCA")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.ylim(0,1.05)
plt.grid(True)
plt.show()