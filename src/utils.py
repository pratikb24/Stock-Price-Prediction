"""
utils.py
========
Plotting and small helper utilities.
"""

from __future__ import annotations

import os
import random
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend so plots save without a display
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """Seed Python and NumPy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def plot_series(series, title="Stock Closing Price", ylabel="Price", save_path=None):
    """Plot a single time series."""
    plt.figure(figsize=(12, 5))
    plt.plot(series, color="#1f77b4", linewidth=1.2)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()


def plot_predictions(y_true, preds: dict, title="Forecast vs Actual", save_path=None):
    """
    Overlay actual values with one or more model predictions.

    preds : {model_name: prediction_array}
    """
    plt.figure(figsize=(13, 6))
    plt.plot(y_true, label="Actual", color="black", linewidth=1.6)
    for name, p in preds.items():
        plt.plot(p, label=name, linewidth=1.2, alpha=0.85)
    plt.title(title)
    plt.xlabel("Time (test window)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()


def plot_metric_bars(metrics_by_model: dict, metric="RMSE", save_path=None):
    """Bar chart comparing a chosen metric across models."""
    names = list(metrics_by_model.keys())
    values = [metrics_by_model[n][metric] for n in names]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, values, color="#4c72b0")
    plt.title(f"Model comparison — {metric}")
    plt.ylabel(metric)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()
