import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# パラメータ設定
n = 100  # 生成するサンプル数
mu = np.array([1, 2, 3])  # 期待値ベクトル
Sigma = np.array([
    [2, -np.sqrt(2 / 3), 0],
    [-np.sqrt(2 / 3), 2, -np.sqrt(1 / 3)],
    [0, -np.sqrt(1 / 3), 2]
])  # 共分散行列

# (1) 一様乱数を 4n 個生成
U = np.random.rand(4*n)

# (2) コレスキー分解 (Sigma = L L^T)
L = np.linalg.cholesky(Sigma)

# -- Box–Muller変換 その1 --
#  U[0:n], U[n:2n] => (Z1, Z2)
r1 = np.sqrt(-2.0 * np.log(U[0:n]))  # (n,)
theta1 = 2.0 * np.pi * U[n:2*n]      # (n,)
Z1 = r1 * np.cos(theta1)  # (n,) 独立標準正規
Z2 = r1 * np.sin(theta1)  # (n,) 独立標準正規

# -- Box–Muller変換 その2 --
#  U[2n:3n], U[3n:4n] => (Z3, Z4)
r2 = np.sqrt(-2.0 * np.log(U[2*n:3*n]))
theta2 = 2.0 * np.pi * U[3*n:4*n]
Z3 = r2 * np.cos(theta2)
Z4 = r2 * np.sin(theta2)

# (3) (Z1, Z2, Z3) => 3次元標準正規ベクトル(n組)
Y = np.stack([Z1, Z2, Z3], axis=1)  # shape=(n,3)

# (4) Y にコレスキー行列 L をかけ => 共分散Sigma の3次元正規
Y = Y @ L.T  # shape=(n,3)

# (5) Z4 を CDF 変換 => U_star (一様分布), さらに -log(U_star) => Exp(1)
U_star = norm.cdf(Z4)    # shape=(n,), 一様(0,1)
Z_exp  = -np.log(U_star) # shape=(n,), Exp(1) (自由度2のガンマ乱数)

# (6) t_{3,2}(mu, Sigma) = mu + Y / sqrt(Z_exp)
X = mu + Y / np.sqrt(Z_exp)[:, None]

# (7) 結果を DataFrame にまとめる
df = pd.DataFrame(X, columns=["x1", "x2", "x3"])