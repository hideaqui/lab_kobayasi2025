import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def perform_regression(file_path, start_epoch, end_epoch):
    # 🔹 CSVファイルの読み込み
    df = pd.read_csv(file_path)

    # 🔹 指定されたエポック範囲のデータをフィルタリング
    df_filtered = df[(df["epoch"] >= start_epoch) & (df["epoch"] <= end_epoch)]

    # 🔹 説明変数（lambda_mean_all, mu_mean_all）と目的変数（actual_gap）
    X = df_filtered[["lambda_mean_all", "mu_mean_all"]].values
    y = df_filtered["actual_gap"].values

    # 🔹 線形回帰モデルの学習
    model = LinearRegression()
    model.fit(X, y)

    # 🔹 予測
    y_pred = model.predict(X)

    # 🔹 評価指標の計算
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # 🔹 結果表示
    print(f"✅ 回帰結果 (Epoch {start_epoch} - {end_epoch}):")
    print(f"  - R²: {r2:.4f}")
    print(f"  - MSE: {mse:.6f}")
    print(f"  - RMSE: {rmse:.6f}")
    print(f"  - 回帰係数 (lambda_mean_all, mu_mean_all): {model.coef_}")
    print(f"  - バイアス (intercept): {model.intercept_}")

    # 🔹 可視化
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df_filtered["lambda_mean_all"], df_filtered["mu_mean_all"], df_filtered["actual_gap"], color='blue', alpha=0.6, label="Actual")
    ax.scatter(df_filtered["lambda_mean_all"], df_filtered["mu_mean_all"], y_pred, color='red', alpha=0.6, label="Predicted")

    ax.set_xlabel("lambda_mean_all")
    ax.set_ylabel("mu_mean_all")
    ax.set_zlabel("actual_gap")
    ax.set_title(f"3D Regression: actual_gap vs lambda_mean_all & mu_mean_all (Epoch {start_epoch} - {end_epoch})")
    ax.legend()

    plt.show()

if __name__ == "__main__":
    file_path = "/Users/hide/卒業研究/resnet_pytorch/output_cifar10/20250126_170439_battiseisokuoff/regression_results_layer4_persistence_1to60_lam_mu_merged.csv"
    start_epoch = 20  # 開始エポック
    end_epoch = 60  # 終了エポック
    perform_regression(file_path, start_epoch, end_epoch)