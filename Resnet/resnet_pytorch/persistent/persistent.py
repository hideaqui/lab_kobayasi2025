import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# Čechフィルトレーションの描画関数
def draw_cech_filtration():
    # 3点の座標
    points = np.array([[0.2, 0.4], [0.8, 0.4], [0.5, 0.8]])

    # 各フィルトレーションの段階（円の半径を増加）
    radii = [0.0, 0.2, 0.35]

    # 各ステップで形成される辺と三角形（シンプレックス）
    edges = [[], [(0, 1)], [(0, 1), (1, 2), (2, 0)]]
    filled_simplices = [[], [], [(0, 1, 2)]]

    # 図の設定
    fig, ax = plt.subplots(2, 3, figsize=(9, 6))

    # 各段階を描画（半径 0 から開始）
    for i in range(3):
        # 上段: Čech複体の構築（円）
        ax[0, i].set_xlim(0, 1)
        ax[0, i].set_ylim(0, 1)
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        ax[0, i].set_aspect('equal')

        for p in points:
            circle = Circle(p, radii[i], color='gray', alpha=0.5, edgecolor='black', lw=1)
            ax[0, i].add_patch(circle)
            ax[0, i].scatter(*p, color='black', zorder=3)

        # 下段: Čech複体の単体の成長
        ax[1, i].set_xlim(0, 1)
        ax[1, i].set_ylim(0, 1)
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
        ax[1, i].set_aspect('equal')

        for p in points:
            ax[1, i].scatter(*p, color='black', zorder=3)

        for e in edges[i]:
            line = Line2D([points[e[0]][0], points[e[1]][0]],
                          [points[e[0]][1], points[e[1]][1]], color='black')
            ax[1, i].add_line(line)

        for tri in filled_simplices[i]:
            ax[1, i].fill(points[tri, 0], points[tri, 1], color='gray', alpha=0.5)

    # 矢印の描画（ステップの遷移を表す）【逆方向: 右から左へ】
    for i in range(2):
        ax[0, i + 1].annotate("", xy=(-0.1, 0.5), xytext=(-0.5, 0.5),
                              arrowprops=dict(arrowstyle="->", lw=2), xycoords=ax[0, i + 1].transAxes)
        ax[1, i + 1].annotate("", xy=(-0.1, 0.5), xytext=(-0.5, 0.5),
                              arrowprops=dict(arrowstyle="->", lw=2), xycoords=ax[1, i + 1].transAxes)

    # 各ステップのタイトル（フィルトレーションの成長）
    ax[0, 0].set_title(r"$r_0$", fontsize=12)
    ax[0, 1].set_title(r"$r_1$", fontsize=12)
    ax[0, 2].set_title(r"$r_2$", fontsize=12)

    plt.tight_layout()
    plt.show()

# 描画実行
draw_cech_filtration()