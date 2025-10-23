from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gudhi.representations.kernel_methods import PersistenceWeightedGaussianKernel
from persim import plot_diagrams
from sklearn.kernel_ridge import KernelRidge

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from lr_analysis_common import compute_layer_data  # type: ignore  # pylint: disable=import-error
else:
    from .lr_analysis_common import compute_layer_data


def _prepare_diagram(diagram_list: List[np.ndarray]) -> np.ndarray:
    """Concatenate finite bars across homology dimensions."""
    points: List[np.ndarray] = []
    for diagram in diagram_list:
        if diagram.size == 0:
            continue
        finite = diagram[np.isfinite(diagram[:, 1])]
        if finite.size == 0:
            continue
        points.append(finite)
    if points:
        return np.vstack(points)
    # Avoid empty diagram issues by inserting a single dummy point on the diagonal.
    return np.array([[0.0, 0.0]])


def _plot_regression(
    epochs: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: str,
    init_label: str,
    layer_index: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(epochs, actual, label="実測値", color="tab:blue", alpha=0.7)
    ax.plot(epochs, predicted, label="PWGK予測値", color="tab:red", linestyle="--")
    ax.set_xlabel("エポック数")
    ax.set_ylabel("汎化ギャップ")
    ax.set_title(f"{init_label} 第{layer_index}層: PWGKによる汎化ギャップ回帰")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    if epochs.size > 0:
        min_epoch = int(epochs.min())
        max_epoch = int(epochs.max())
        tick_start = max(0, (min_epoch // 5) * 5)
        ticks = np.arange(tick_start, max_epoch + 5, 5)
        if len(ticks) > 0:
            ax.set_xticks(ticks)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_inverse_pd_grid(
    diagram_map: Dict[int, List[np.ndarray]],
    representative_epochs: List[int],
    target_gaps: List[float],
    output_path: str,
    init_label: str,
    layer_index: int,
    font_family: str = "Hiragino Sans",
) -> None:
    if not representative_epochs:
        return

    plt.rcParams["font.family"] = font_family
    cols = min(5, len(representative_epochs))
    rows = int(np.ceil(len(representative_epochs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(rows, cols)
    fig.suptitle(f"{init_label} 第{layer_index}層: 汎化ギャップ別PD（PWGK局所回帰）", fontsize=16)

    for idx, (ax, epoch, gap) in enumerate(zip(axes.ravel(), representative_epochs, target_gaps)):
        diagrams = diagram_map.get(epoch)
        if diagrams is None:
            ax.axis("off")
            ax.set_title(f"汎化ギャップ={gap:.4f}\n(データなし)", fontsize=10)
            continue
        plot_diagrams(diagrams, ax=ax, show=False)
        ax.set_title(f"汎化ギャップ={gap:.4f}\n代表エポック={epoch}", fontsize=10)

    total_axes = rows * cols
    for idx in range(len(representative_epochs), total_axes):
        ax = axes.ravel()[idx]
        ax.axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_pwgk_regression(
    output_dir: str,
    init_label: str,
    *,
    layers: Sequence[int] = (1, 2, 3, 4),
    font_family: str = "Hiragino Sans",
    max_features: Optional[int] = 512,
    bandwidth: float = 1.0,
    alpha: float = 1e-6,
    inverse_bandwidth: Optional[float] = None,
) -> None:
    accuracy_df, layer_results = compute_layer_data(
        output_dir=output_dir,
        init_label=init_label,
        layers=layers,
        max_features=max_features,
        font_family=font_family,
    )

    for layer_idx, data in layer_results.items():
        layer_dir = data["layer_dir"]
        merged_df = data["merged_df"]
        diagrams_valid: List[List[np.ndarray]] = data["valid_diagrams"]  # type: ignore[assignment]

        epochs = merged_df["epoch"].to_numpy()
        gaps = merged_df["generalization_gap"].to_numpy()
        if len(diagrams_valid) != len(epochs):
            print(f"[PWGK] {init_label} layer {layer_idx}: mismatch between diagrams and results, skipping.")
            continue

        diagrams = [_prepare_diagram(diag) for diag in diagrams_valid]
        pwkg = PersistenceWeightedGaussianKernel(bandwidth=bandwidth)
        gram = pwkg.fit_transform(diagrams)

        model = KernelRidge(alpha=alpha, kernel="precomputed")
        model.fit(gram, gaps)
        predictions = model.predict(gram)

        from sklearn.metrics import r2_score

        # 決定係数（R²）の計算
        r2 = r2_score(gaps, predictions)
        print(f"[PWGK] {init_label} layer {layer_idx}: R² = {r2:.4f}")

        # 結果データフレームにR²を追加
        merged_df["pwgk_r2"] = r2

        result_df = merged_df.copy()
        result_df["pwgk_predicted_gap"] = predictions
        result_df["pwgk_residual"] = result_df["generalization_gap"] - predictions

        csv_path = os.path.join(layer_dir, "pwgk_regression_results.csv")
        result_df.to_csv(csv_path, index=False)

        plot_path = os.path.join(layer_dir, "pwgk_generalization_gap_regression.png")
        _plot_regression(epochs, gaps, predictions, plot_path, init_label, layer_idx)

        # -----------------
        # Inverse regression (local linear)
        # -----------------
        if inverse_bandwidth is None:
            gap_std = np.std(gaps) if gaps.size > 1 else 0.0
            sigma = max(gap_std / 2.0, 1e-6)
        else:
            sigma = inverse_bandwidth

        embeddings = gram  # rows act as implicit embeddings
        target_gaps = np.linspace(gaps.min(), gaps.max(), min(5, len(gaps)))
        representative_epochs: List[int] = []
        representative_gaps: List[float] = []
        distances: List[float] = []
        r2_locals: List[float] = []

        for target_gap in target_gaps:
            centered = gaps - target_gap
            weights = np.exp(-0.5 * (centered / sigma) ** 2)
            W = np.diag(weights)
            X = np.column_stack((np.ones_like(gaps), centered))
            XT_W = X.T @ W
            try:
                beta = np.linalg.pinv(XT_W @ X) @ XT_W @ embeddings
            except np.linalg.LinAlgError:
                continue
            embedding_hat = beta[0]  # local linear estimate at target gap
            dists = np.linalg.norm(embeddings - embedding_hat, axis=1)

            # --- 局所線形回帰の決定係数 R² を計算 ---
            y_hat = X @ beta.T  # 各サンプルの局所予測値
            weighted_mean = np.sum(weights * gaps) / np.sum(weights)
            ss_res = np.sum(weights * (gaps - y_hat[:, 0]) ** 2)
            ss_tot = np.sum(weights * (gaps - weighted_mean) ** 2)
            r2_local = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            print(f"  [Local R²] target_gap={target_gap:.4f} → R²={r2_local:.4f}")
            r2_locals.append(float(r2_local))

            idx_min = int(np.argmin(dists))
            representative_epochs.append(int(epochs[idx_min]))
            representative_gaps.append(float(gaps[idx_min]))
            distances.append(float(dists[idx_min]))

        inverse_csv = os.path.join(layer_dir, "pwgk_inverse_regression.csv")
        inverse_df = pd.DataFrame(
            {
                "target_gap": target_gaps[: len(representative_epochs)],
                "representative_epoch": representative_epochs,
                "representative_gap": representative_gaps,
                "distance": distances,
                "r2_local": r2_locals,
            }
        )
        inverse_df.to_csv(inverse_csv, index=False)

        diagram_map = {int(epoch): diag for epoch, diag in zip(epochs, diagrams_valid)}
        inverse_plot_path = os.path.join(layer_dir, "pwgk_inverse_pd_grid.png")
        _plot_inverse_pd_grid(
            diagram_map,
            representative_epochs,
            list(target_gaps[: len(representative_epochs)]),
            inverse_plot_path,
            init_label,
            layer_idx,
            font_family=font_family,
        )

        print(f"[PWGK] Completed {init_label} layer {layer_idx}: saved regression and inverse analysis to {layer_dir}")
