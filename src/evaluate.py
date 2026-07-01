"""
evaluate.py
===========
Forecast-accuracy metrics used throughout the project:

    RMSE  - Root Mean Squared Error      (same units as price; headline metric)
    MAE   - Mean Absolute Error
    MAPE  - Mean Absolute Percentage Err (~1% is the project target)
    R2    - Coefficient of determination
"""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (%). Guards against divide-by-zero."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2_score(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def evaluate_all(y_true, y_pred) -> dict:
    """Return all metrics as a dict."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def print_metrics(name: str, y_true, y_pred):
    """Pretty-print a one-line metrics summary for a model."""
    m = evaluate_all(y_true, y_pred)
    print(
        f"{name:<14} | RMSE: {m['RMSE']:8.3f} | MAE: {m['MAE']:8.3f} "
        f"| MAPE: {m['MAPE']:6.3f}% | R2: {m['R2']:6.3f}"
    )
    return m
