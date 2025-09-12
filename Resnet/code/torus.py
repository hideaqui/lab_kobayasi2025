import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

# トーラス上の点を生成
def sample_torus(n_points=5000, R=2, r=1):
    theta = 2 * np.pi * np.random.rand(n_points)
    phi = 2 * np.pi * np.random.rand(n_points)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.vstack([x, y, z]).T

# 点を取得
points = sample_torus()

# 3Dプロット
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2)
ax.set_title("Sampled Points on Torus")
plt.show()

# Ripserを使ってパーシステンスダイアグラムを計算
diagrams = ripser(points)['dgms']

# パーシステンスダイアグラムを表示
plot_diagrams(diagrams, show=True)