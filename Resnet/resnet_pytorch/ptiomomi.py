import sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Hiragino Sans"
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm  # statsmodels をインポート
from tqdm import tqdm
from ripser import ripser
from sklearn.metrics import mean_squared_error, r2_score

# **データの保存ディレクトリ**
output_dir = "/Users/hide/lab_kobayasi2025/output/normal_20251001_032131"
accuracy_csv_path = os.path.join(output_dir, "epoch_accuracies_normal.csv")

# **解析するエポックの範囲**
epoch_range = range(1, 61)

# **汎化ギャップデータの取得**
print("📌 汎化ギャップデータを読み込み中...")
accuracy_df = pd.read_csv(accuracy_csv_path)
accuracy_df = accuracy_df[accuracy_df["epoch"].isin(epoch_range)]
generalization_gaps = accuracy_df["generalization_gap"].values
valid_epochs = accuracy_df["epoch"].values

print(f"✅ {len(valid_epochs)} エポック分のデータを取得しました。")

# **特徴量を保存するリスト**
all_features = []
valid_epochs_filtered = []

print("🔹 FC 層の重みの相関（512×512）から PH を抽出中...")
for epoch in tqdm(valid_epochs, desc="エポック処理中", unit="epoch"):
    weight_file = os.path.join(output_dir, f"epoch_{epoch}_fc_weight.npy")

    if not os.path.exists(weight_file):
        print(f"⚠️ ファイルが見つかりません: {weight_file}")
        continue

    # **重みをロード (10, 512) → 転置して (512, 10) に変換**
    fc_weight = np.load(weight_file)  # (10, 512)
    fc_weight = fc_weight.T  # (512, 10)

    # **相関行列を計算（512×512）**
    correlation_matrix = np.corrcoef(fc_weight, rowvar=True)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

    # **距離行列を作成（1 - 相関係数の絶対値）**
    distance_matrix = 1 - np.abs(correlation_matrix)
    np.fill_diagonal(distance_matrix, 0)

    # **パーシステントホモロジーを計算**
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
    all_features.append([lambda_mean, mu_mean])
    valid_epochs_filtered.append(epoch)

# **NumPy 配列に変換**
all_features = np.array(all_features)
valid_epochs_filtered = np.array(valid_epochs_filtered)

# **回帰分析（statsmodels を使用）**
print("📌 OLS 回帰モデルを学習中...")
X = sm.add_constant(all_features)  # 切片（バイアス）を追加
y = generalization_gaps[:len(valid_epochs_filtered)]

model = sm.OLS(y, X).fit()  # statsmodels で回帰分析
predicted_gaps = model.predict(X)

# **回帰結果**
r2 = model.rsquared
mse = mean_squared_error(y, predicted_gaps)
rmse = np.sqrt(mse)

print("✅ 回帰分析が完了しました。")
print(model.summary())  # 詳細な回帰結果を出力

# **結果を DataFrame にまとめる**
conf_int = model.conf_int()
coef_df = pd.DataFrame({
    "パラメータ": ["バイアス (const)", "パーシステンスライフ (λ)", "パーシステンスミッドライフ (μ)"],
    "回帰係数": model.params,  # `.values` を削除
    "標準誤差": model.bse,  # `.values` を削除
    "t 値": model.tvalues,  # `.values` を削除
    "p 値": model.pvalues,  # `.values` を削除
    "95% 信頼区間下限": conf_int.iloc[:, 0],  # `.values` を削除
    "95% 信頼区間上限": conf_int.iloc[:, 1]  # `.values` を削除
})

# **回帰結果を CSV に保存**
coef_result_path = os.path.join(output_dir, "回帰分析統計量_fc_weight_512x512_persistence.csv")
coef_df.to_csv(coef_result_path, index=False, encoding="utf-8-sig")
print(f"✅ 回帰分析統計量を保存しました: {coef_result_path}")

# **可視化**
plt.figure(figsize=(8, 5))
plt.scatter(valid_epochs_filtered, y, label="実測値", marker="o", alpha=0.7, color="blue")
plt.plot(valid_epochs_filtered, predicted_gaps, label="予測値", linestyle="--", color="red")

plt.xlabel("エポック数")
plt.ylabel("汎化ギャップ")
plt.legend()
plt.grid()
plt.show()