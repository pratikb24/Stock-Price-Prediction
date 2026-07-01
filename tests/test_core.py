import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import data_preprocessing as dp
from src import evaluate as ev
from src import utils
from src import lstm_model as lm


def test_clean_data():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"]
            ),
            "Close": [100.0, 100.0, np.nan, 102.0],
        }
    ).set_index("Date")
    cleaned = dp.clean_data(df)
    assert cleaned.index.is_unique, "duplicate dates not removed"
    assert cleaned["Close"].isna().sum() == 0, "NaNs not filled"
    print("[PASS] clean_data")


def test_scaling_roundtrip():
    train = np.linspace(10, 20, 50)
    test = np.linspace(20, 25, 10)
    tr_s, te_s, scaler = dp.scale_train_test(train, test)
    assert tr_s.min() >= -1e-9 and tr_s.max() <= 1 + 1e-9
    back = dp.inverse_scale(scaler, tr_s)
    assert np.allclose(back, train, atol=1e-6), "inverse scaling mismatch"
    print("[PASS] scaling roundtrip")


def test_split_is_chronological():
    s = pd.Series(np.arange(100))
    train, test = dp.train_test_split_series(s, 0.8)
    assert len(train) == 80 and len(test) == 20
    assert train.iloc[-1] < test.iloc[0], "split not chronological"
    print("[PASS] chronological split")


def test_metrics():
    y = np.array([100.0, 110.0, 120.0])
    yhat = np.array([102.0, 108.0, 119.0])
    assert abs(ev.rmse(y, yhat) - np.sqrt((4 + 4 + 1) / 3)) < 1e-9
    assert ev.mae(y, yhat) > 0
    assert ev.mape(y, yhat) > 0
    assert ev.r2_score(y, y) == 1.0
    m = ev.evaluate_all(y, yhat)
    assert set(m) == {"RMSE", "MAE", "MAPE", "R2"}
    print("[PASS] metrics")


def test_create_sequences():
    data = np.arange(10)
    X, y = lm.create_sequences(data, look_back=3)
    assert X.shape == (7, 3, 1)
    assert y.shape == (7,)
    assert list(X[0].ravel()) == [0, 1, 2]
    assert y[0] == 3
    print("[PASS] create_sequences")


def test_plots_save(tmp="results"):
    utils.ensure_dir(tmp)
    y = np.sin(np.linspace(0, 6, 50))
    utils.plot_series(y, save_path=os.path.join(tmp, "_test_series.png"))
    utils.plot_predictions(y, {"m1": y * 0.9}, save_path=os.path.join(tmp, "_test_pred.png"))
    assert os.path.exists(os.path.join(tmp, "_test_series.png"))
    os.remove(os.path.join(tmp, "_test_series.png"))
    os.remove(os.path.join(tmp, "_test_pred.png"))
    print("[PASS] plotting")


if __name__ == "__main__":
    test_clean_data()
    test_scaling_roundtrip()
    test_split_is_chronological()
    test_metrics()
    test_create_sequences()
    test_plots_save()
    print("\nAll core tests passed.")
