from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np

# 合成データセットの生成（例: 2つの円）
t = np.linspace(0, 2*np.pi, 100)
x1 = np.sin(t)
y1 = np.cos(t)

x2 = np.sin(t) + 5
y2 = np.cos(t) + 5

data = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2))))

# 永続ホモロジーの計算
diagrams = ripser(data)['dgms']

# 永続図の可視化
plot_diagrams(diagrams, show=True)

