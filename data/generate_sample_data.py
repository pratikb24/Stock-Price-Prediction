"""
generate_sample_data.py
========================
Generate a realistic, fully-offline synthetic OHLCV stock-price dataset so the
project runs end-to-end without needing an internet connection or an API key.

The series is built with Geometric Brownian Motion (GBM) and layered with a
slow long-term trend plus mild seasonality, which gives the AR/MA/ARMA/ARIMA
and LSTM models something structured (yet noisy) to learn.

Run:
    python data/generate_sample_data.py
Produces:
    data/sample_stock_data.csv  (Date, Open, High, Low, Close, Volume)
"""

import os
import numpy as np
import pandas as pd


def generate_stock_data(
    n_days: int = 1500,
    start_price: float = 150.0,
    mu: float = 0.0004,        # daily drift
    sigma: float = 0.018,      # daily volatility
    seed: int = 42,
    start_date: str = "2018-01-01",
) -> pd.DataFrame:
    """Generate a synthetic OHLCV dataframe with business-day frequency."""
    rng = np.random.default_rng(seed)

    # --- Close price via Geometric Brownian Motion -------------------------
    daily_returns = rng.normal(loc=mu, scale=sigma, size=n_days)

    # add a gentle, slowly-varying seasonal component (e.g. ~quarterly cycle)
    t = np.arange(n_days)
    seasonal = 0.0009 * np.sin(2 * np.pi * t / 63.0)        # ~quarter
    seasonal += 0.0004 * np.sin(2 * np.pi * t / 21.0)       # ~month
    daily_returns = daily_returns + seasonal

    log_price = np.log(start_price) + np.cumsum(daily_returns)
    close = np.exp(log_price)

    # --- Build OHLC around the close ---------------------------------------
    open_ = close * (1 + rng.normal(0, 0.004, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n_days)))
    volume = rng.integers(1_000_000, 8_000_000, n_days)

    dates = pd.bdate_range(start=start_date, periods=n_days)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": np.round(open_, 2),
            "High": np.round(high, 2),
            "Low": np.round(low, 2),
            "Close": np.round(close, 2),
            "Volume": volume,
        }
    )
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "sample_stock_data.csv")
    df = generate_stock_data()
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(df)} rows to {out_path}")
    print(df.head())
    print("...")
    print(df.tail())


if __name__ == "__main__":
    main()
