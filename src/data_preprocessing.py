"""
data_preprocessing.py
=====================
Data cleaning, normalization and stationarity checks for stock-price series.

This module covers the "Approach" step from the project brief:
    * Data cleaning  -> handle missing values, duplicates, ordering
    * Normalization  -> Min-Max scaling for the neural network
    * Stationarity   -> Augmented Dickey-Fuller (ADF) test + auto differencing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def load_data(path: str, date_col: str = "Date") -> pd.DataFrame:
    """Load a CSV of OHLCV data and parse the date column as the index."""
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean an OHLCV dataframe:
      * drop exact duplicate rows / duplicate dates
      * coerce price columns to numeric
      * forward/backward fill missing values
      * drop any rows still fully empty
    """
    df = df.copy()

    # Remove duplicate timestamps (keep first occurrence)
    df = df[~df.index.duplicated(keep="first")]

    # Coerce numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Stock prices are continuous: forward-fill gaps (holidays/halts), then back-fill leading NaNs
    df = df.ffill().bfill()

    # Drop anything still missing
    df = df.dropna(how="any")
    return df


def get_target_series(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Extract a single target column (default: closing price) as a Series."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available: {list(df.columns)}")
    return df[column].astype(float)


# --------------------------------------------------------------------------- #
# Stationarity
# --------------------------------------------------------------------------- #
def adf_test(series: pd.Series, signif: float = 0.05) -> dict:
    """
    Run the Augmented Dickey-Fuller test for stationarity.

    Returns a dict with the test statistic, p-value, critical values and a
    boolean `is_stationary` (p-value < signif rejects the unit-root null).
    """
    from statsmodels.tsa.stattools import adfuller

    series = series.dropna()
    result = adfuller(series, autolag="AIC")
    out = {
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < signif,
    }
    return out


def find_differencing_order(series: pd.Series, max_d: int = 2, signif: float = 0.05) -> int:
    """
    Determine the smallest differencing order `d` that makes the series
    stationary according to the ADF test. This is the `d` in ARIMA(p, d, q).
    """
    d = 0
    current = series.dropna().copy()
    while d <= max_d:
        if adf_test(current, signif)["is_stationary"]:
            return d
        current = current.diff().dropna()
        d += 1
    return max_d


def difference(series: pd.Series, d: int = 1) -> pd.Series:
    """Apply `d` rounds of differencing."""
    out = series.copy()
    for _ in range(d):
        out = out.diff()
    return out.dropna()


# --------------------------------------------------------------------------- #
# Normalization & splitting
# --------------------------------------------------------------------------- #
def train_test_split_series(series: pd.Series, train_ratio: float = 0.8):
    """Chronological split (no shuffling — this is time-series data)."""
    n = len(series)
    split = int(n * train_ratio)
    return series.iloc[:split], series.iloc[split:]


def scale_train_test(train: np.ndarray, test: np.ndarray, feature_range=(0, 1)):
    """
    Fit a MinMaxScaler on the TRAIN portion only (to avoid look-ahead leakage)
    and transform both train and test. Returns (train_scaled, test_scaled, scaler).
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    train = np.asarray(train).reshape(-1, 1)
    test = np.asarray(test).reshape(-1, 1)
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled, scaler


def inverse_scale(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    """Invert Min-Max scaling back to original price units."""
    values = np.asarray(values).reshape(-1, 1)
    return scaler.inverse_transform(values).ravel()
