import numpy as np
import pandas as pd
from ripser import ripser
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# ★実際のパスに置き換えてください
output_dir = "/Users/hide/卒業研究/resnet_pytorch/output/20250123_140605"
accuracy_csv_path = os.path.join(output_dir, "epoch_accuracies.csv")

num_epochs = 21  # 例: 0 ~ 20

# epoch_accuracies.csv から generalization_gap を取得
accuracy_df = pd.read_csv(accuracy_csv_path)
generalization_gaps = accuracy_df["generalization_gap"].values[:num_epochs]

# 特徴量 (各エポックの次元別 λ, μ の平均・分散) を格納するリスト
# 例: [ λ0_mean, λ0_var, μ0_mean, μ0_var,  λ1_mean, λ1_var, μ1_mean, μ1_var, ... , λ2_mean, λ2_var, μ2_mean, μ2_var ]
all_features = []

valid_epochs = []

for epoch in range(num_epochs):
    activation_file = os.path.join(output_dir, f"epoch_{epoch}_layer4.npy")

    if not os.path.exists(activation_file):
        print(f"⚠️ ファイルが見つかりません: {activation_file}")
        continue

    valid_epochs.append(epoch)

    # 活性化データをロードして reshape (N, C, 1, 1) → (N, C)
    activation_data = np.load(activation_file)
    activation_data_reshaped = activation_data.reshape(activation_data.shape[0], -1)

    # ripser で 0~2次元のバーコードを取得
    diagrams = ripser(activation_data_reshaped, maxdim=2)['dgms']

    # 各次元で λ, μ の平均/分散を計算
    epoch_features = []

    for dim in range(3):
        # (b, d) リストを取得 (∞除外)
        diagram_dim = diagrams[dim]
        finite_bars = [(b, d) for (b, d) in diagram_dim if np.isfinite(d)]

        if len(finite_bars) == 0:
            # バーが存在しない場合
            lam_mean = 0.0
            lam_var  = 0.0
            mu_mean  = 0.0
            mu_var   = 0.0
        else:
            # ライフタイム (d - b)
            lifetimes = [d - b for b, d in finite_bars]
            # バーの中心 ((b + d)/2)
            midpoints = [(b + d)/2 for b, d in finite_bars]

            lam_mean = np.mean(lifetimes)   # λ の平均
            lam_var  = np.var(lifetimes)    # λ の分散
            mu_mean  = np.mean(midpoints)   # μ の平均 (バーの中心)
            mu_var   = np.var(midpoints)    # μ の分散

        # (λの平均, λの分散, μの平均, μの分散) の順で格納
        epoch_features.extend([lam_mean, lam_var, mu_mean, mu_var])

    all_features.append(epoch_features)

# NumPy配列に変換
all_features = np.array(all_features)
valid_epochs = np.array(valid_epochs)

# 対応する generalization_gap を取得 (スキップした epoch を除外)
y = generalization_gaps[valid_epochs]

# 線形回帰モデル
model = LinearRegression()
model.fit(all_features, y)

# 回帰結果
coeffs = model.coef_
intercept = model.intercept_
predicted_gaps = model.predict(all_features)

r2_score = model.score(all_features, y)
mse = mean_squared_error(y, predicted_gaps)
rmse = np.sqrt(mse)

print("回帰係数:", coeffs)
print("バイアス (intercept):", intercept)
print("R-squared (決定係数):", r2_score)
print("MSE (平均二乗誤差):", mse)
print("RMSE (二乗平均平方根誤差):", rmse)

# 結果を DataFrame 化
dims = ["dim0", "dim1", "dim2"]
columns = []
for d in dims:
    columns.extend([f"{d}_lambda_mean", f"{d}_lambda_var", f"{d}_mu_mean", f"{d}_mu_var"])

result_df = pd.DataFrame({
    "epoch": valid_epochs,
    "actual_gap": y,
    "predicted_gap": predicted_gaps
})

# 特徴量を列として追加
for i, col_name in enumerate(columns):
    result_df[col_name] = all_features[:, i]

# CSV保存
result_path = os.path.join(output_dir, "regression_results_dimwise.csv")
result_df.to_csv(result_path, index=False)
print(f"✅ 回帰結果を保存しました: {result_path}")



