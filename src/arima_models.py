"""
arima_models.py
===============
Classical linear time-series models: AR, MA, ARMA and ARIMA.

All four are special cases of statsmodels' ARIMA(p, d, q):
    AR(p)        -> ARIMA(p, 0, 0)
    MA(q)        -> ARIMA(0, 0, q)
    ARMA(p, q)   -> ARIMA(p, 0, q)
    ARIMA(p,d,q) -> ARIMA(p, d, q)

The module provides:
    * thin wrappers to fit each model
    * a rolling (walk-forward) one-step-ahead forecaster, which is the
      realistic way to evaluate a model on a hold-out set
    * an AIC grid-search to auto-select (p, q) orders
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # silence statsmodels convergence chatter


def _fit_arima(series, order):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(series, order=order)
    return model.fit()


# --------------------------------------------------------------------------- #
# Single-fit convenience wrappers
# --------------------------------------------------------------------------- #
def fit_ar(series, p=2):
    """Autoregressive model AR(p) = ARIMA(p, 0, 0)."""
    return _fit_arima(series, (p, 0, 0))


def fit_ma(series, q=2):
    """Moving-average model MA(q) = ARIMA(0, 0, q)."""
    return _fit_arima(series, (0, 0, q))


def fit_arma(series, p=2, q=2):
    """ARMA(p, q) = ARIMA(p, 0, q)."""
    return _fit_arima(series, (p, 0, q))


def fit_arima(series, p=5, d=1, q=0):
    """ARIMA(p, d, q)."""
    return _fit_arima(series, (p, d, q))


# --------------------------------------------------------------------------- #
# Auto order selection
# --------------------------------------------------------------------------- #
def auto_order(series, d=1, p_range=range(0, 4), q_range=range(0, 4)):
    """
    Grid-search (p, q) for a fixed d by minimising AIC.
    Returns the best (p, d, q) tuple.
    """
    best_aic = np.inf
    best_order = (1, d, 0)
    for p in p_range:
        for q in q_range:
            if p == 0 and q == 0:
                continue
            try:
                res = _fit_arima(series, (p, d, q))
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
            except Exception:
                continue
    return best_order


# --------------------------------------------------------------------------- #
# Walk-forward (rolling) one-step-ahead forecasting
# --------------------------------------------------------------------------- #
def rolling_forecast(train, test, order):
    """
    Walk-forward validation: re-fit (cheaply, via append) and predict one step
    at a time across the whole test set. This mirrors how the model would be
    used in production and gives an honest error estimate.

    Parameters
    ----------
    train, test : array-like (1-D)
    order       : (p, d, q)

    Returns
    -------
    predictions : np.ndarray  (same length as `test`)
    """
    from statsmodels.tsa.arima.model import ARIMA

    history = list(np.asarray(train, dtype=float))
    test = np.asarray(test, dtype=float)
    predictions = np.empty(len(test))

    # Fit once on the training history...
    model = ARIMA(history, order=order)
    fitted = model.fit()

    for t in range(len(test)):
        yhat = fitted.forecast(steps=1)[0]
        predictions[t] = yhat
        # ...then extend the model with the true observation (fast update).
        fitted = fitted.append([test[t]], refit=False)

    return predictions


def simple_forecast(fitted_model, steps):
    """Multi-step forecast from an already-fitted model."""
    return np.asarray(fitted_model.forecast(steps=steps))
