from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_rmse_plot(results_df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = _prepare_output_dir(output_dir) / "rmse_comparison.png"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(results_df["model"], results_df["rmse"], color=["#4C78A8", "#F58518"])
    ax.set_title("RMSE by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_precision_recall_plot(results_df: pd.DataFrame, output_dir: str | Path, k: int) -> Path:
    output_path = _prepare_output_dir(output_dir) / "precision_recall_comparison.png"

    labels = results_df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, results_df["precision_at_k"], width=width, label=f"Precision@{k}", color="#54A24B")
    ax.bar(x + width / 2, results_df["recall_at_k"], width=width, label=f"Recall@{k}", color="#E45756")

    ax.set_title(f"Precision@{k} and Recall@{k} by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_diversity_plot_if_available(results_df: pd.DataFrame, output_dir: str | Path) -> Path | None:
    has_coverage = "item_coverage" in results_df.columns
    has_gini = "item_gini" in results_df.columns
    if not (has_coverage or has_gini):
        return None

    output_path = _prepare_output_dir(output_dir) / "diversity_comparison.png"

    labels = results_df["model"].tolist()
    x = np.arange(len(labels))

    metric_cols: List[str] = []
    if has_coverage:
        metric_cols.append("item_coverage")
    if has_gini:
        metric_cols.append("item_gini")

    width = 0.35 if len(metric_cols) == 2 else 0.6
    offsets = np.linspace(-(len(metric_cols) - 1) * width / 2, (len(metric_cols) - 1) * width / 2, len(metric_cols))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#72B7B2", "#B279A2"]
    for idx, col in enumerate(metric_cols):
        ax.bar(x + offsets[idx], results_df[col], width=width, label=col, color=colors[idx % len(colors)])

    ax.set_title("Diversity Metrics by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_all_plots(results_df: pd.DataFrame, output_dir: str | Path, k: int) -> dict[str, str]:
    paths: dict[str, str] = {}
    rmse_plot = save_rmse_plot(results_df, output_dir)
    pr_plot = save_precision_recall_plot(results_df, output_dir, k)
    diversity_plot = save_diversity_plot_if_available(results_df, output_dir)

    paths["rmse_plot"] = str(rmse_plot)
    paths["precision_recall_plot"] = str(pr_plot)
    if diversity_plot is not None:
        paths["diversity_plot"] = str(diversity_plot)

    return paths
