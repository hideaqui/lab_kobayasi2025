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
output_dir = "/Users/hide/卒業研究/resnet_pytorch/output_6000/20250126_021306"
accuracy_csv_path = os.path.join(output_dir, "epoch_accuracies.csv")

# 解析するエポックの範囲
epoch_range = range(25, 50)

# 🔹 Generalization Gap の取得
print("📌 Loading generalization gap data...")
accuracy_df = pd.read_csv(accuracy_csv_path)
accuracy_df = accuracy_df[accuracy_df["epoch"].isin(epoch_range)]
generalization_gaps = accuracy_df["generalization_gap"].values
valid_epochs = accuracy_df["epoch"].values
print(f"✅ {len(valid_epochs)} epochs found in range {epoch_range.start}-{epoch_range.stop}.")

# 🔹 特徴量のリスト
all_features = []

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

    # 🔹 特徴量の抽出（各次元の λ, μ の平均・分散）
    epoch_features = []
    for dim in range(3):
        diagram_dim = diagrams[dim]
        finite_bars = [(b, d) for (b, d) in diagram_dim if np.isfinite(d)]

        if len(finite_bars) == 0:
            lam_mean, lam_var, mu_mean, mu_var = 0.0, 0.0, 0.0, 0.0
        else:
            lifetimes = [d - b for b, d in finite_bars]
            midpoints = [(b + d) / 2 for b, d in finite_bars]
            lam_mean, lam_var = np.mean(lifetimes), np.var(lifetimes)
            mu_mean, mu_var = np.mean(midpoints), np.var(midpoints)

        epoch_features.extend([lam_mean, lam_var, mu_mean, mu_var])

    all_features.append(epoch_features)

# 🔹 NumPy配列に変換
all_features = np.array(all_features)
valid_epochs = np.array(valid_epochs)

# 🔹 回帰分析
print("📌 Performing linear regression...")
X = all_features  # 特徴量
y = generalization_gaps  # 目標値

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
columns = ["dim0_lambda_mean", "dim0_lambda_var", "dim0_mu_mean", "dim0_mu_var",
           "dim1_lambda_mean", "dim1_lambda_var", "dim1_mu_mean", "dim1_mu_var",
           "dim2_lambda_mean", "dim2_lambda_var", "dim2_mu_mean", "dim2_mu_var"]

result_df = pd.DataFrame({"epoch": valid_epochs, "actual_gap": y, "predicted_gap": predicted_gaps})
for i, col_name in enumerate(columns):
    result_df[col_name] = all_features[:, columns.index(col_name)]  # 各次元の特徴量を追加

result_path = os.path.join(output_dir, "regression_results_fc_weight_512x10_persistence.csv")
result_df.to_csv(result_path, index=False)
print(f"✅ 回帰結果を保存しました: {result_path}")

# 🔹 可視化
plt.figure(figsize=(8, 5))
plt.scatter(result_df["epoch"], result_df["actual_gap"], label="Actual Gap", marker="o", alpha=0.7)
plt.plot(result_df["epoch"], result_df["predicted_gap"], label="Predicted Gap", linestyle="--", color="red")

plt.xlabel("Epoch")
plt.ylabel("Generalization Gap")
plt.legend()
plt.title("Generalization Gap Prediction (Epoch 25-200, FC Weight (512x10) with Persistence)")
plt.grid()
plt.show()