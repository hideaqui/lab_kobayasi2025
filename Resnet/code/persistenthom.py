import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

def plot_persistence_diagram():
    np.random.seed(2)  # 乱数シードを固定

    # **5点を円周上に配置**
    num_points = 5
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False) + np.random.uniform(-0.3, 0.3, num_points)
    radius = 2.0  # **円の半径**
    center = np.array([0, 0])
    points = np.column_stack([center[0] + radius * np.cos(theta),
                              center[1] + radius * np.sin(theta)])

    # **パーシステンスダイアグラムの生成**
    results = ripser(points, maxdim=2)  # 最大2次元まで解析
    diagrams = results['dgms']

    # **パーシステンスダイアグラムの可視化**
    plt.figure(figsize=(6, 6))
    plot_diagrams(diagrams, show=True)
    plt.title("Persistence Diagram")
    plt.show()

# **描画実行**
plot_persistence_diagram()