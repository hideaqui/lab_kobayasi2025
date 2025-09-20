import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ripser import ripser
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# **フォント設定（日本語対応）**
plt.rcParams["font.family"] = "Hiragino Sans"  # Mac
# plt.rcParams["font.family"] = "Meiryo"  # Windows
# plt.rcParams["font.family"] = "IPAexGothic"  # Linux

# **データの保存ディレクトリ**
output_dir = "/Users/hide/卒業研究/resnet_pytorch/output_cifar10/20250126_170439_battiseisokuoff"

# **エポック範囲の定義**
epoch_range = range(1, 60)  

# **汎化ギャップデータの取得**
accuracy_csv_path = os.path.join(output_dir, "epoch_accuracies.csv")
print("📌 汎化ギャップデータを読み込み中...")
accuracy_df = pd.read_csv(accuracy_csv_path)
accuracy_df = accuracy_df[accuracy_df["epoch"].isin(epoch_range)]
generalization_gaps = accuracy_df["generalization_gap"].values
valid_epochs = accuracy_df["epoch"].values

print(f"✅ {len(valid_epochs)} エポック分のデータを取得しました。")

# **PH の特徴量を格納するリスト**
ph_features = []  
valid_epochs_filtered = []  

# **PH の計算**
print("🔹 相関行列を用いた PH の特徴量を計算中...")
for epoch in tqdm(epoch_range, desc="エポック処理中", unit="epoch"):
    activation_file = os.path.join(output_dir, f"epoch_{epoch}_layer4.npy")

    if not os.path.exists(activation_file):
        print(f"⚠️ ファイルが見つかりません: {activation_file}")
        continue

    # **活性化データをロード**
    activations = np.load(activation_file).reshape(1024, -1)

    # **NaN や Inf を含む行・列を削除**
    valid_rows = np.all(np.isfinite(activations), axis=1)
    activations_filtered = activations[valid_rows, :]

    valid_cols = np.all(np.isfinite(activations_filtered), axis=0)
    activations_filtered = activations_filtered[:, valid_cols]

    if activations_filtered.shape[0] == 0 or activations_filtered.shape[1] == 0:
        print(f"⚠️ エポック {epoch}: データが全て削除されたためスキップ。")
        continue

    # **標準偏差が 0 の特徴を削除**
    std = np.std(activations_filtered, axis=0)
    valid_features = std > 0
    activations_filtered = activations_filtered[:, valid_features]

    if activations_filtered.shape[1] == 0:
        print(f"⚠️ エポック {epoch}: 全ての特徴が削除されたためスキップ。")
        continue

    # **相関行列を計算**
    corr_matrix = np.corrcoef(activations_filtered, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # **相関行列から距離行列を作成**
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.nan_to_num(distance_matrix, nan=1.0)

    # **PH を計算**
    diagrams = ripser(distance_matrix, maxdim=1, metric="precomputed")["dgms"]

    # **PH 特徴量の抽出**
    all_lifetimes = []
    all_midpoints = []

    for dim in range(len(diagrams)):
        diagram_dim = diagrams[dim]
        finite_bars = [(b, d) for (b, d) in diagram_dim if np.isfinite(d)]

        if len(finite_bars) > 0:
            lifetimes = [d - b for b, d in finite_bars]  
            midpoints = [(b + d) / 2 for b, d in finite_bars]  
            all_lifetimes.extend(lifetimes)
            all_midpoints.extend(midpoints)

    lambda_mean = np.mean(all_lifetimes) if len(all_lifetimes) > 0 else 0.0
    mu_mean = np.mean(all_midpoints) if len(all_midpoints) > 0 else 0.0
    ph_features.append([lambda_mean, mu_mean])
    valid_epochs_filtered.append(epoch)

# **NumPy 配列に変換**
ph_features = np.array(ph_features)
valid_epochs_filtered = np.array(valid_epochs_filtered)

# **回帰分析**
print("📌 線形回帰モデルを学習中...")
X = ph_features
y = generalization_gaps[:len(valid_epochs_filtered)]

model = LinearRegression()
model.fit(X, y)
predicted_gaps = model.predict(X)

# **回帰結果**
r2 = r2_score(y, predicted_gaps)
mse = mean_squared_error(y, predicted_gaps)
rmse = np.sqrt(mse)

print("✅ 回帰分析が完了しました。")
print("回帰係数 (λ, μ):", model.coef_)
print("バイアス (intercept):", model.intercept_)
print(f"決定係数 (R²): {r2:.4f}")
print(f"平均二乗誤差 (MSE): {mse:.4f}")
print(f"二乗平均平方根誤差 (RMSE): {rmse:.4f}")

# **結果の保存（日本語対応のエンコーディング）**
result_df = pd.DataFrame({
    "エポック": valid_epochs_filtered,
    "実際の汎化ギャップ": y,
    "予測された汎化ギャップ": predicted_gaps,
    "パーシステンスライフ (λ)": ph_features[:, 0],
    "パーシステンスミッドライフ (μ)": ph_features[:, 1],
    "回帰係数 (λ)": model.coef_[0],
    "回帰係数 (μ)": model.coef_[1],
    "バイアス": model.intercept_,
    "決定係数 (R²)": r2
})

result_path = os.path.join(output_dir, "回帰分析結果_layer4_persistence.csv")
result_df.to_csv(result_path, index=False, encoding="utf-8-sig")  # `utf-8-sig` を指定
print(f"✅ 回帰結果を保存しました: {result_path}")

# **可視化: 汎化ギャップの実測値 vs 予測値**
plt.figure(figsize=(8, 5))
plt.scatter(result_df["エポック"], result_df["実際の汎化ギャップ"], label="実測値", marker="o", alpha=0.7, color="blue")
plt.plot(result_df["エポック"], result_df["予測された汎化ギャップ"], label="予測値", linestyle="--", color="red")

plt.xlabel("エポック数")
plt.ylabel("汎化ギャップ")
plt.legend()
plt.grid()
plt.show()
latex_code = model.summary2().as_latex()
print(latex_code)
conf_int = model.conf_int()
coef_df = pd.DataFrame({
    "パラメータ": ["バイアス (const)", "パーシステンスライフ (λ)", "パーシステンスミッドライフ (μ)"],
    "回帰係数": model.params,  
    "標準誤差": model.bse,  
    "t 値": model.tvalues,  
    "p 値": model.pvalues,  
    "95% 信頼区間下限": conf_int[:, 0],  
    "95% 信頼区間上限": conf_int[:, 1]
})