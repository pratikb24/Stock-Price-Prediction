from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import data_preprocessing as dp
from src import arima_models as am
from src import evaluate as ev
from src import utils


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid ARIMA-LSTM stock forecaster")
    p.add_argument("--data", default="data/sample_stock_data.csv",
                   help="Path to OHLCV CSV")
    p.add_argument("--column", default="Close", help="Target column to forecast")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--look-back", type=int, default=30,
                   help="LSTM sliding-window length")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--quick", action="store_true",
                   help="Fast run for testing (few epochs, small ARIMA grid)")
    return p.parse_args()


def main():
    args = parse_args()
    utils.set_seed(42)
    utils.ensure_dir(args.results_dir)

    if args.quick:
        args.epochs = 3

    print("=" * 70)
    print("STEP 1 — Load & clean data")
    print("=" * 70)
    df = dp.load_data(args.data)
    df = dp.clean_data(df)
    series = dp.get_target_series(df, args.column)
    print(f"Loaded {len(series)} observations of '{args.column}' "
          f"({series.index.min().date()} → {series.index.max().date()})")
    utils.plot_series(series.values, title=f"{args.column} price",
                      save_path=os.path.join(args.results_dir, "01_price_series.png"))


    print("\n" + "=" * 70)
    print("STEP 2 — Stationarity check (ADF)")
    print("=" * 70)
    adf_raw = dp.adf_test(series)
    print(f"Raw series      : ADF={adf_raw['adf_statistic']:.4f}, "
          f"p={adf_raw['p_value']:.4f}, stationary={adf_raw['is_stationary']}")
    d = dp.find_differencing_order(series, max_d=2)
    print(f"Chosen differencing order d = {d}")
    if d > 0:
        adf_diff = dp.adf_test(dp.difference(series, d))
        print(f"Differenced (d={d}): ADF={adf_diff['adf_statistic']:.4f}, "
              f"p={adf_diff['p_value']:.4f}, stationary={adf_diff['is_stationary']}")


    print("\n" + "=" * 70)
    print("STEP 3 — Train/test split")
    print("=" * 70)
    train, test = dp.train_test_split_series(series, args.train_ratio)
    print(f"Train: {len(train)}   Test: {len(test)}")
    train_vals, test_vals = train.values, test.values

    metrics = {}
    preds = {}

    print("\n" + "=" * 70)
    print("STEP 4 — Classical models (walk-forward one-step-ahead)")
    print("=" * 70)

    classical = {
        "AR":   (2, 0, 0),
        "MA":   (0, 0, 2),
        "ARMA": (2, 0, 2),
    }

    try:
        if args.quick:
            arima_order = (2, d, 1)
        else:
            print("Searching ARIMA order via AIC grid ...")
            arima_order = am.auto_order(train_vals, d=d,
                                        p_range=range(0, 4), q_range=range(0, 4))
    except Exception as e:
        print(f"Could not run ARIMA order search ({e}). "
              f"Is statsmodels installed? Falling back to (2, {d}, 1).")
        arima_order = (2, d, 1)
    classical["ARIMA"] = arima_order
    print(f"ARIMA order selected: {arima_order}")

    for name, order in classical.items():
        try:
            yhat = am.rolling_forecast(train_vals, test_vals, order)
            metrics[name] = ev.print_metrics(name, test_vals, yhat)
            preds[name] = yhat
        except Exception as e:
            print(f"{name:<14} | skipped ({e})")

    print("\n" + "=" * 70)
    print("STEP 5 — Standalone LSTM")
    print("=" * 70)
    try:
        from src import lstm_model as lm

        lm.set_tf_seed(42)
        train_scaled, test_scaled, scaler = dp.scale_train_test(train_vals, test_vals)

        full_scaled = np.concatenate([train_scaled.ravel(), test_scaled.ravel()])
        X_train, y_train = lm.create_sequences(train_scaled.ravel(), args.look_back)

        model = lm.build_lstm(look_back=args.look_back)
        lm.train_lstm(model, X_train, y_train, epochs=args.epochs,
                      batch_size=args.batch_size, verbose=0)

        n_train = len(train_scaled)
        lstm_pred_scaled = []
        for t in range(len(test_scaled)):
            window = full_scaled[n_train + t - args.look_back: n_train + t]
            window = window.reshape(1, args.look_back, 1)
            lstm_pred_scaled.append(lm.predict_lstm(model, window)[0])
        lstm_pred = dp.inverse_scale(scaler, np.array(lstm_pred_scaled))

        metrics["LSTM"] = ev.print_metrics("LSTM", test_vals, lstm_pred)
        preds["LSTM"] = lstm_pred
    except ImportError:
        print("TensorFlow not installed — skipping LSTM. "
              "Install with `pip install tensorflow` to enable it.")

    print("\n" + "=" * 70)
    print("STEP 6 — Hybrid ARIMA-LSTM")
    print("=" * 70)
    try:
        from src.hybrid_model import hybrid_arima_lstm

        result = hybrid_arima_lstm(
            train_vals, test_vals,
            arima_order=arima_order,
            look_back=args.look_back,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
        )
        metrics["ARIMA-LSTM"] = ev.print_metrics("ARIMA-LSTM", test_vals, result["hybrid"])
        preds["ARIMA-LSTM"] = result["hybrid"]
    except ImportError:
        print("TensorFlow not installed — skipping hybrid model.")


    print("\n" + "=" * 70)
    print("STEP 7 — Save results")
    print("=" * 70)

    utils.plot_predictions(
        test_vals, preds,
        title="Forecast vs Actual (test window)",
        save_path=os.path.join(args.results_dir, "02_forecast_vs_actual.png"),
    )
    if metrics:
        utils.plot_metric_bars(
            metrics, metric="RMSE",
            save_path=os.path.join(args.results_dir, "03_rmse_comparison.png"),
        )
        utils.plot_metric_bars(
            metrics, metric="MAPE",
            save_path=os.path.join(args.results_dir, "04_mape_comparison.png"),
        )

    summary = pd.DataFrame(metrics).T
    summary = summary[["RMSE", "MAE", "MAPE", "R2"]] if not summary.empty else summary
    summary_path = os.path.join(args.results_dir, "metrics_summary.csv")
    summary.to_csv(summary_path)
    print("\nFinal metrics:")
    print(summary.round(4).to_string())
    print(f"\nSaved metrics to {summary_path}")
    print(f"Saved plots to {args.results_dir}/")

    if "ARIMA-LSTM" in metrics:
        best = min(metrics, key=lambda k: metrics[k]["RMSE"])
        print(f"\nBest model by RMSE: {best} "
              f"(RMSE={metrics[best]['RMSE']:.3f}, MAPE={metrics[best]['MAPE']:.3f}%)")


if __name__ == "__main__":
    main()
