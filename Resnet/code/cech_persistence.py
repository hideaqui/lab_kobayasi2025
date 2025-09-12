import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

# 3点の座標
points = np.array([[0.2, 0.4], [0.7, 0.33], [0.5, 0.8]])

# パーシステントホモロジーの計算
results = ripser(points, maxdim=2)  # 最大2次元まで解析
diagrams = results['dgms']

# 可視化
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# ① パーシステンスダイアグラム
plot_diagrams(diagrams, ax=ax[0], show=False)
ax[0].set_title("Persistence Diagram")

# ② パーシステンスバーコード
plot_diagrams(diagrams, ax=ax[1], show=False, plot_only=False)
ax[1].set_title("Persistence Barcode")

# 表示
plt.tight_layout()
plt.show()