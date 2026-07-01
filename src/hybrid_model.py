"""
hybrid_model.py
===============
Hybrid ARIMA-LSTM forecaster.

Idea (Zhang, 2003): a time series is a combination of a *linear* component and
a *nonlinear* component.

    y_t = L_t (linear) + N_t (nonlinear)

    1. ARIMA models the linear structure  -> gives predictions L_hat and
       residuals  e_t = y_t - L_hat_t.
    2. The residuals still contain nonlinear patterns ARIMA cannot capture.
       An LSTM is trained on those residuals to predict N_hat_t.
    3. The final forecast recombines both parts:

           y_hat_t = L_hat_t + N_hat_t

This typically beats either model alone, which is why the project's headline
result (low RMSE / ~1% MAPE) comes from the ARIMA-LSTM combination.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.arima_models import rolling_forecast
from src.lstm_model import create_sequences, build_lstm, train_lstm, predict_lstm, set_tf_seed


def hybrid_arima_lstm(
    train: np.ndarray,
    test: np.ndarray,
    arima_order=(5, 1, 0),
    look_back: int = 30,
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = 42,
    verbose: int = 1,
):
    """
    Fit the hybrid ARIMA-LSTM model and produce one-step-ahead forecasts for
    the test window.

    Returns
    -------
    dict with keys:
        'hybrid'   : final combined predictions (len == len(test))
        'arima'    : the ARIMA-only predictions over the test window
        'residual_pred' : the LSTM residual predictions
    """
    train = np.asarray(train, dtype=float).ravel()
    test = np.asarray(test, dtype=float).ravel()

    # ----------------------------------------------------------------- #
    # 1. ARIMA: linear component
    # ----------------------------------------------------------------- #
    from statsmodels.tsa.arima.model import ARIMA

    fitted = ARIMA(train, order=arima_order).fit()

    # In-sample fitted values on the training set -> training residuals
    arima_fitted_train = np.asarray(fitted.fittedvalues, dtype=float)
    train_residuals = train - arima_fitted_train

    # Walk-forward ARIMA predictions for the test window
    arima_test_pred = rolling_forecast(train, test, arima_order)
    test_residuals_actual = test - arima_test_pred  # realised residuals

    # ----------------------------------------------------------------- #
    # 2. LSTM on the residual series (nonlinear component)
    # ----------------------------------------------------------------- #
    set_tf_seed(seed)

    # Scale residuals (fit on TRAIN residuals only)
    res_scaler = MinMaxScaler(feature_range=(0, 1))
    train_res_scaled = res_scaler.fit_transform(train_residuals.reshape(-1, 1)).ravel()

    X_train, y_train = create_sequences(train_res_scaled, look_back=look_back)

    model = build_lstm(look_back=look_back)
    train_lstm(
        model, X_train, y_train,
        epochs=epochs, batch_size=batch_size, verbose=verbose,
    )

    # ----------------------------------------------------------------- #
    # 3. Walk-forward residual prediction for the test window
    #    The window of past residuals is extended with the *actual* residual
    #    once each true value becomes known (consistent walk-forward).
    # ----------------------------------------------------------------- #
    full_res_scaled = list(train_res_scaled)
    test_res_actual_scaled = res_scaler.transform(
        test_residuals_actual.reshape(-1, 1)
    ).ravel()

    residual_pred_scaled = np.empty(len(test))
    for t in range(len(test)):
        window = np.array(full_res_scaled[-look_back:]).reshape(1, look_back, 1)
        residual_pred_scaled[t] = predict_lstm(model, window)[0]
        # reveal the true residual for the next step's window
        full_res_scaled.append(test_res_actual_scaled[t])

    residual_pred = res_scaler.inverse_transform(
        residual_pred_scaled.reshape(-1, 1)
    ).ravel()

    # ----------------------------------------------------------------- #
    # 4. Recombine: linear + nonlinear
    # ----------------------------------------------------------------- #
    hybrid_pred = arima_test_pred + residual_pred

    return {
        "hybrid": hybrid_pred,
        "arima": arima_test_pred,
        "residual_pred": residual_pred,
        "lstm_model": model,
        "arima_model": fitted,
    }
