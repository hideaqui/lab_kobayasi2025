import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 🔹 データの読み込み設定
output_dir = "/Users/hide/卒業研究/resnet_pytorch/output_cifar10/20250126_170439_battiseisokuoff"
accuracy_csv_path = os.path.join(output_dir, "epoch_accuracies.csv")

# **解析するエポックの範囲（1エポック目から60エポックまで）**
epoch_range = range(1, 61)

# 🔹 Generalization Gap の取得
print("📌 Loading generalization gap data...")
accuracy_df = pd.read_csv(accuracy_csv_path)
accuracy_df = accuracy_df[accuracy_df["epoch"].isin(epoch_range)]
generalization_gaps = accuracy_df["generalization_gap"].values
valid_epochs = accuracy_df["epoch"].values
print(f"✅ {len(valid_epochs)} epochs found in range {epoch_range.start}-{epoch_range.stop-1}.")

# 🔹 特徴量のリストとバー情報の保存用リスト
all_features = []
all_persistence_bars = []

print("🔹 Extracting features from FC layer weights...")
for epoch in tqdm(valid_epochs, desc="Processing epochs", unit="epoch"):
    weight_file = os.path.join(output_dir, f"epoch_{epoch}_fc_weight.npy")

    if not os.path.exists(weight_file):
        print(f"⚠️ File not found for epoch {epoch}: {weight_file}")
        continue

    # **重みをロード (10, 512) → 転置して (512, 10) に変換**
    fc_weight = np.load(weight_file).T  # (512, 10)

    # 🔹 512個の `10` 次元ベクトルの距離行列を計算（ユークリッド距離）
    distance_matrix = squareform(pdist(fc_weight, metric='euclidean'))

    # 🔹 パーシステントホモロジー解析
    diagrams = ripser(distance_matrix, distance_matrix=True, maxdim=2)['dgms']

    # 🔹 各次元のバーを統合
    lifetimes_all = []
    midpoints_all = []
    epoch_persistence_bars = []

    for dim in range(3):  # 次元 0, 1, 2 を統合
        diagram_dim = diagrams[dim]
        finite_bars = [(b, d) for (b, d) in diagram_dim if np.isfinite(d)]

        if len(finite_bars) > 0:
            lifetimes_all.extend([d - b for b, d in finite_bars])
            midpoints_all.extend([(b + d) / 2 for b, d in finite_bars])

            # 🔹 各バー情報を保存
            epoch_persistence_bars.extend([
                {"epoch": epoch, "dim": dim, "birth": b, "death": d, "lifetime": d - b, "midpoint": (b + d) / 2}
                for b, d in finite_bars
            ])

    # 🔹 λ, μ の全次元統合平均
    lambda_mean_all = np.mean(lifetimes_all) if lifetimes_all else 0.0
    mu_mean_all = np.mean(midpoints_all) if midpoints_all else 0.0

    all_features.append([lambda_mean_all, mu_mean_all])
    all_persistence_bars.extend(epoch_persistence_bars)

    # 🔹 エポックごとのバーを CSV に保存
    epoch_bars_df = pd.DataFrame(epoch_persistence_bars)
    epoch_bars_path = os.path.join(output_dir, f"epoch_{epoch}_persistence_bars.csv")
    epoch_bars_df.to_csv(epoch_bars_path, index=False)
    print(f"✅ 保存: {epoch_bars_path}")

# 🔹 NumPy配列に変換
all_features = np.array(all_features)
valid_epochs = np.array(valid_epochs)

# 🔹 回帰分析
print("📌 Performing linear regression...")
X = all_features  # 説明変数（λとμの統合平均）
y = generalization_gaps  # 目標値（汎化ギャップ）

model = LinearRegression()
model.fit(X, y)
predicted_gaps = model.predict(X)

# 🔹 回帰結果
r2_score = model.score(X, y)
mse = mean_squared_error(y, predicted_gaps)
rmse = np.sqrt(mse)

print("✅ Regression completed.")
print("回帰係数:", model.coef_)
print("バイアス (intercept):", model.intercept_)
print("R-squared (決定係数):", r2_score)
print("MSE (平均二乗誤差):", mse)
print("RMSE (二乗平均平方根誤差):", rmse)

# 🔹 結果の保存
result_df = pd.DataFrame({
    "epoch": valid_epochs,
    "actual_gap": y,
    "predicted_gap": predicted_gaps,
    "lambda_mean_all": all_features[:, 0],
    "mu_mean_all": all_features[:, 1]
})

result_path = os.path.join(output_dir, "regression_results_fc_weight_512x10_persistence_1to60_lam_mu_merged.csv")
result_df.to_csv(result_path, index=False)
print(f"✅ 回帰結果を保存しました: {result_path}")

# 🔹 全エポックのパーシステントバーを統合した CSV を作成
all_bars_df = pd.DataFrame(all_persistence_bars)
all_bars_path = os.path.join(output_dir, "persistence_bars_all_epochs.csv")
all_bars_df.to_csv(all_bars_path, index=False)
print(f"✅ 全エポックのバー情報を統合して保存: {all_bars_path}")

# 🔹 可視化
plt.figure(figsize=(8, 5))
plt.scatter(result_df["epoch"], result_df["actual_gap"], label="Actual Gap", marker="o", alpha=0.7)
plt.plot(result_df["epoch"], result_df["predicted_gap"], label="Predicted Gap", linestyle="--", color="red")

plt.xlabel("Epoch")
plt.ylabel("Generalization Gap")
plt.legend()
plt.title("Generalization Gap Prediction (Epoch 1-60, FC Weight (512x10) with Merged Persistence)")
plt.grid()
plt.show()