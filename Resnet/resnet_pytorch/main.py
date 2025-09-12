import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from ripser import ripser

# 結果を保存するフォルダの指定
output_dir = "lg_output"
os.makedirs(output_dir, exist_ok=True)

# 回帰分析の結果を保存するファイル
regression_output_path = os.path.join(output_dir, "regression_results.txt")

# トポロジー情報を保存するファイル
topology_output_path = os.path.join(output_dir, "topology_data.npy")

# Example data (replace with your actual data)
X_features = np.random.rand(100, 10)  # 100 samples, 10 features
y_accuracy = np.random.rand(100)  # 100 target values

# データの前処理と回帰
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)

# 線形回帰の適用
reg = LinearRegression()
reg.fit(X_features, y_accuracy)

# 決定係数などの情報を保存
with open(regression_output_path, "w") as f:
    f.write(f"回帰係数: {reg.coef_}\n")
    f.write(f"切片: {reg.intercept_}\n")
    f.write(f"決定係数 (R^2): {reg.score(X_features, y_accuracy)}\n")

# Ripserを用いたトポロジー解析
diagrams = ripser(X_features, maxdim=3)['dgms']
np.save(topology_output_path, diagrams)

print(f"回帰分析の結果を {regression_output_path} に保存しました。")
print(f"トポロジー情報を {topology_output_path} に保存しました。")