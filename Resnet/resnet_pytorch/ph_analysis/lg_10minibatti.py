import os
import numpy as np
import pandas as pd
import torch
from ripser import ripser
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 🔹 設定
output_dir = "/Users/hide/卒業研究/resnet_pytorch/output_6000/20250125_235615"
accuracy_csv_path = os.path.join(output_dir, "batchwise_accuracies.csv")
epoch_range = range(10, 41)  # **解析対象のエポック (10～40)**
batch_number = 10  # 各エポックの先頭バッチ

# 🔹 Generalization Gap の取得（エポック 10～40 のみ）
accuracy_df = pd.read_csv(accuracy_csv_path)
accuracy_df = accuracy_df[(accuracy_df["epoch"].isin(epoch_range)) & (accuracy_df["batch"] == batch_number)]
generalization_gaps = accuracy_df["generalization_gap"].values
valid_epochs = accuracy_df["epoch"].values

# 🔹 特徴量リスト
all_features = []

for epoch in valid_epochs:
    activation_file = os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_number}_layer4.npy")

    if not os.path.exists(activation_file):
        print(f"⚠️ ファイルが見つかりません: {activation_file}")
        continue

    # **活性化データをロード (Shape: (N, 512, 1, 1) → (N, 512))**
    activation_data = np.load(activation_file).reshape(-1, 512)

    # 🔹 相関行列を計算（Pearson 相関）
    correlation_matrix = np.corrcoef(activation_data, rowvar=False)

    # 🔹 相関係数を距離に変換 & 対称化
    distance_matrix = 1 - correlation_matrix  # 相関が高いほど距離が小さい
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # **対称化**
    np.fill_diagonal(distance_matrix, 0)  # **対角成分をゼロにする**

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

# 🔹 回帰分析（正則化なし）
y = generalization_gaps[:len(valid_epochs)]  # Generalization gap

model = LinearRegression()
model.fit(all_features, y)
predicted_gaps = model.predict(all_features)

# 🔹 回帰結果
r2_score = model.score(all_features, y)
mse = mean_squared_error(y, predicted_gaps)
rmse = np.sqrt(mse)

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
    result_df[col_name] = all_features[:, i]

result_path = os.path.join(output_dir, "regression_results_corr_10-40_fixed.csv")
result_df.to_csv(result_path, index=False)
print(f"✅ 回帰結果を保存しました: {result_path}")

# 🔹 可視化
plt.figure(figsize=(8, 5))
plt.scatter(result_df["epoch"], result_df["actual_gap"], label="Actual Gap", marker="o", alpha=0.7)
plt.plot(result_df["epoch"], result_df["predicted_gap"], label="Predicted Gap", linestyle="--", color="red")

plt.xlabel("Epoch")
plt.ylabel("Generalization Gap")
plt.legend()
plt.title("Generalization Gap Prediction (Epoch 10-40, Correlation)")
plt.grid()
plt.show()