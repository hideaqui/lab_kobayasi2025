import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from persim import plot_diagrams
from ripser import ripser
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm


@dataclass
class EpochResult:
    epoch: int
    h0_persistence_life: float
    h1_persistence: float
    h1_persistence_midlife: float
    remaining_samples: int
    remaining_features: int
    removed_nonfinite_rows: int
    removed_nonfinite_features: int
    removed_zero_std_features: int
    truncated_features: int


def _flatten_activations(activations: np.ndarray) -> np.ndarray:
    """Reshape activation tensor to (batch, features)."""
    return activations.reshape(activations.shape[0], -1)


def _clean_activation_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Remove non-finite samples/features and zero-variance features."""
    info = {
        "removed_nonfinite_rows": 0,
        "removed_nonfinite_features": 0,
        "removed_zero_std_features": 0,
        "truncated_features": 0,
    }

    # Remove rows with any NaN/Inf.
    row_mask = np.all(np.isfinite(matrix), axis=1)
    info["removed_nonfinite_rows"] = int(matrix.shape[0] - np.sum(row_mask))
    matrix = matrix[row_mask]

    if matrix.size == 0:
        return matrix, info

    # Remove features with any NaN/Inf.
    col_mask = np.all(np.isfinite(matrix), axis=0)
    info["removed_nonfinite_features"] = int(matrix.shape[1] - np.sum(col_mask))
    matrix = matrix[:, col_mask]

    if matrix.size == 0:
        return matrix, info

    # Remove zero-variance features.
    std = matrix.std(axis=0, ddof=0)
    var_mask = std > 0
    info["removed_zero_std_features"] = int(matrix.shape[1] - np.sum(var_mask))
    matrix = matrix[:, var_mask]

    return matrix, info


def _limit_features(matrix: np.ndarray, max_features: Optional[int]) -> Tuple[np.ndarray, int]:
    """Optionally reduce feature dimensionality to keep Ripser tractable."""
    if max_features is None or matrix.shape[1] <= max_features:
        return matrix, 0

    indices = np.linspace(0, matrix.shape[1] - 1, max_features, dtype=int)
    reduced = matrix[:, indices]
    removed = matrix.shape[1] - reduced.shape[1]
    return reduced, int(removed)


def _compute_distance_matrix(features: np.ndarray) -> Optional[np.ndarray]:
    """Return correlation-based distance matrix or None if invalid."""
    if features.shape[0] < 2 or features.shape[1] < 2:
        return None

    corr = np.corrcoef(features, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.isfinite(corr).all():
        return None

    distance = 1.0 - np.abs(corr)
    np.fill_diagonal(distance, 0.0)
    distance = np.nan_to_num(distance, nan=1.0, posinf=1.0, neginf=1.0)

    if not np.isfinite(distance).all():
        return None

    return distance


def _mean_lifetime(diagrams: List[np.ndarray], dim: int) -> float:
    if dim >= len(diagrams):
        return 0.0
    lifetimes = diagrams[dim]
    if lifetimes.size == 0:
        return 0.0
    finite = lifetimes[np.isfinite(lifetimes[:, 1])]
    if finite.size == 0:
        return 0.0
    lengths = finite[:, 1] - finite[:, 0]
    if lengths.size == 0:
        return 0.0
    return float(np.mean(lengths))


def _mean_midlife(diagrams: List[np.ndarray], dim: int) -> float:
    if dim >= len(diagrams):
        return 0.0
    bars = diagrams[dim]
    if bars.size == 0:
        return 0.0
    finite = bars[np.isfinite(bars[:, 1])]
    if finite.size == 0:
        return 0.0
    midlife = (finite[:, 0] + finite[:, 1]) / 2.0
    if midlife.size == 0:
        return 0.0
    return float(np.mean(midlife))


def _summarize_diagrams(diagrams: List[np.ndarray]) -> Tuple[float, float, float]:
    h0_life = _mean_lifetime(diagrams, dim=0)
    h1_life = _mean_lifetime(diagrams, dim=1)
    h1_mid = _mean_midlife(diagrams, dim=1)
    return h0_life, h1_life, h1_mid


def _plot_regression(
    df: pd.DataFrame,
    layer_index: int,
    output_path: str,
    init_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["epoch"], df["generalization_gap"], label="Actual", color="tab:blue", alpha=0.7)
    ax.plot(df["epoch"], df["predicted_generalization_gap"], label="Predicted", color="tab:red", linestyle="--")
    ax.set_xlabel("エポック数")
    ax.set_ylabel("汎化ギャップ")
    ax.set_title(f"{init_label} 第{layer_index}層: 汎化ギャップ回帰")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    if not df["epoch"].empty:
        min_epoch = int(df["epoch"].min())
        max_epoch = int(df["epoch"].max())
        tick_start = max(0, (min_epoch // 5) * 5)
        ticks = np.arange(tick_start, max_epoch + 5, 5)
        if len(ticks) > 0:
            ax.set_xticks(ticks)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_persistent_diagrams(
    diagrams_per_epoch: Sequence[Optional[List[np.ndarray]]],
    epochs: Sequence[int],
    layer_index: int,
    output_path: str,
    init_label: str,
) -> None:
    if not diagrams_per_epoch:
        return

    cols = 5
    rows = math.ceil(len(diagrams_per_epoch) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(rows, cols)
    fig.suptitle(f"{init_label} 第{layer_index}層: パーシステント図の遷移", fontsize=16)

    for idx, (ax, epoch, diagrams) in enumerate(zip(axes.ravel(), epochs, diagrams_per_epoch)):
        if diagrams is None:
            ax.axis("off")
            ax.set_title(f"Epoch {epoch} (Skipped)", fontsize=10)
            continue
        plot_diagrams(diagrams, ax=ax, show=False)
        ax.set_title(f"Epoch {epoch}", fontsize=10)

    # Hide unused axes if total < rows*cols.
    total_axes = rows * cols
    for idx in range(len(diagrams_per_epoch), total_axes):
        ax = axes.ravel()[idx]
        ax.axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_analysis(
    output_dir: str,
    init_label: str,
    layers: Sequence[int] = (1, 2, 3, 4),
    font_family: str = "Hiragino Sans",
    max_features: Optional[int] = 512,
) -> None:
    """Run persistence-based analysis for each layer inside the output directory."""
    plt.rcParams["font.family"] = font_family

    accuracy_df, layer_results = compute_layer_data(
        output_dir=output_dir,
        init_label=init_label,
        layers=layers,
        max_features=max_features,
        font_family=font_family,
    )

    for layer, data in layer_results.items():
        layer_dir = data["layer_dir"]

        merged_df: pd.DataFrame = data["merged_df"]
        reg = data["regressor"]
        metrics = data["metrics"]
        diagrams = data["diagrams"]
        epochs_considered = data["epochs"]

        csv_path = os.path.join(layer_dir, "regression_results.csv")
        merged_df.to_csv(csv_path, index=False)

        plot_path = os.path.join(layer_dir, "generalization_gap_regression.png")
        _plot_regression(merged_df, layer, plot_path, init_label)

        diagrams_path = os.path.join(layer_dir, "persistent_diagrams.png")
        _plot_persistent_diagrams(diagrams, epochs_considered, layer, diagrams_path, init_label)

        metrics_path = os.path.join(layer_dir, "regression_metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            metrics_file.write(f"R2: {metrics['r2']:.6f}\n")
            metrics_file.write(f"MSE: {metrics['mse']:.6f}\n")
            metrics_file.write(f"RMSE: {metrics['rmse']:.6f}\n")
            metrics_file.write(f"Intercept: {reg.intercept_:.6f}\n")
            metrics_file.write(f"Coef H0 persistence life: {reg.coef_[0]:.6f}\n")
            metrics_file.write(f"Coef H1 persistence: {reg.coef_[1]:.6f}\n")
            metrics_file.write(f"Coef H1 persistence midlife: {reg.coef_[2]:.6f}\n")

        print(f"Layer {layer} complete: saved CSV, regression plot, and persistent diagrams to {layer_dir}")


def compute_layer_data(
    output_dir: str,
    init_label: str,
    layers: Sequence[int] = (1, 2, 3, 4),
    max_features: Optional[int] = 512,
    font_family: str = "Hiragino Sans",
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, object]]]:
    """Return per-layer analysis artifacts without saving plots/csv."""
    plt.rcParams["font.family"] = font_family

    accuracy_path = os.path.join(output_dir, f"epoch_accuracies_{init_label}.csv")
    if not os.path.exists(accuracy_path):
        raise FileNotFoundError(f"Accuracy CSV not found: {accuracy_path}")

    accuracy_df = pd.read_csv(accuracy_path)
    if "epoch" not in accuracy_df.columns or "generalization_gap" not in accuracy_df.columns:
        raise ValueError(f"CSV must contain 'epoch' and 'generalization_gap': {accuracy_path}")

    accuracy_df["epoch"] = accuracy_df["epoch"].astype(int)
    epoch_order = sorted(accuracy_df["epoch"].unique())

    layer_results: Dict[int, Dict[str, object]] = {}

    for layer in layers:
        layer_dir = os.path.join(output_dir, f"layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)

        results: List[EpochResult] = []
        diagrams_for_plot: List[Optional[List[np.ndarray]]] = []
        diagrams_valid: List[List[np.ndarray]] = []
        epochs_considered: List[int] = []

        print(f"\n=== Processing {init_label} Layer {layer} ===")
        activation_template = f"epoch_{{epoch}}_layer{layer}_activation_{init_label}.npy"

        for epoch in tqdm(epoch_order, desc=f"Layer {layer} epochs", unit="epoch"):
            activation_path = os.path.join(output_dir, activation_template.format(epoch=epoch))
            if not os.path.exists(activation_path):
                print(f"Missing activation file skipped: {activation_path}")
                diagrams_for_plot.append(None)
                epochs_considered.append(epoch)
                continue

            activations = np.load(activation_path)
            features = _flatten_activations(activations)
            cleaned, info = _clean_activation_matrix(features)
            cleaned, truncated = _limit_features(cleaned, max_features=max_features)
            info["truncated_features"] = truncated

            if info["removed_nonfinite_rows"] > 0 or info["removed_nonfinite_features"] > 0:
                print(
                    f"Epoch {epoch} Layer {layer}: removed "
                    f"{info['removed_nonfinite_rows']} non-finite samples and "
                    f"{info['removed_nonfinite_features']} non-finite features."
                )

            if info["removed_zero_std_features"] > 0:
                print(
                    f"Epoch {epoch} Layer {layer}: removed {info['removed_zero_std_features']} zero-variance features."
                )

            if info["truncated_features"] > 0:
                print(
                    f"Epoch {epoch} Layer {layer}: truncated {info['truncated_features']} features "
                    f"to cap dimensionality at {max_features}."
                )

            if cleaned.size == 0:
                print(f"Epoch {epoch} Layer {layer}: no data remaining after cleaning, skipping.")
                diagrams_for_plot.append(None)
                epochs_considered.append(epoch)
                continue

            distance = _compute_distance_matrix(cleaned)
            if distance is None:
                print(f"Epoch {epoch} Layer {layer}: invalid distance matrix, skipping.")
                diagrams_for_plot.append(None)
                epochs_considered.append(epoch)
                continue

            try:
                ripser_result = ripser(distance, maxdim=1, distance_matrix=True)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Epoch {epoch} Layer {layer}: ripser failed with {exc!r}, skipping.")
                diagrams_for_plot.append(None)
                epochs_considered.append(epoch)
                continue

            diagrams = ripser_result["dgms"]
            h0_life, h1_life, h1_mid = _summarize_diagrams(diagrams)

            results.append(
                EpochResult(
                    epoch=epoch,
                    h0_persistence_life=h0_life,
                    h1_persistence=h1_life,
                    h1_persistence_midlife=h1_mid,
                    remaining_samples=int(cleaned.shape[0]),
                    remaining_features=int(cleaned.shape[1]),
                    removed_nonfinite_rows=info["removed_nonfinite_rows"],
                    removed_nonfinite_features=info["removed_nonfinite_features"],
                    removed_zero_std_features=info["removed_zero_std_features"],
                    truncated_features=info["truncated_features"],
                )
            )
            diagrams_for_plot.append(diagrams)
            epochs_considered.append(epoch)
            diagrams_valid.append(diagrams)

        if not results:
            print(f"No valid epochs for {init_label} layer {layer}.")
            continue

        result_df = pd.DataFrame([r.__dict__ for r in results])
        merged_df = result_df.merge(accuracy_df[["epoch", "generalization_gap"]], on="epoch", how="left")

        features_cols = ["h0_persistence_life", "h1_persistence", "h1_persistence_midlife"]
        X = merged_df[features_cols].values
        y = merged_df["generalization_gap"].values

        reg = LinearRegression()
        reg.fit(X, y)
        predictions = reg.predict(X)

        merged_df["predicted_generalization_gap"] = predictions
        merged_df["intercept"] = reg.intercept_
        merged_df["coef_h0_persistence_life"] = reg.coef_[0]
        merged_df["coef_h1_persistence"] = reg.coef_[1]
        merged_df["coef_h1_persistence_midlife"] = reg.coef_[2]

        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = math.sqrt(mse)

        merged_df["r2"] = r2
        merged_df["mse"] = mse
        merged_df["rmse"] = rmse

        layer_results[layer] = {
            "layer_dir": layer_dir,
            "merged_df": merged_df,
            "diagrams": diagrams_for_plot,
            "valid_diagrams": diagrams_valid,
            "epochs": epochs_considered,
            "regressor": reg,
            "metrics": {"r2": r2, "mse": mse, "rmse": rmse},
        }

    return accuracy_df, layer_results
