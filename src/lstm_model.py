"""
lstm_model.py
=============
LSTM (Long Short-Term Memory) network for sequential stock-price forecasting.

Workflow:
    1. Convert a 1-D scaled price series into supervised (X, y) sequences
       using a sliding window of `look_back` timesteps.
    2. Train a stacked LSTM with dropout regularisation.
    3. Predict one step ahead across the test window.

TensorFlow/Keras is imported lazily so that the ARIMA-only parts of the
project still work if TensorFlow is not installed.
"""

from __future__ import annotations

import numpy as np


def set_tf_seed(seed: int = 42):
    """Make TensorFlow runs reproducible."""
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)


def create_sequences(data: np.ndarray, look_back: int = 60):
    """
    Turn a 1-D array into overlapping (X, y) pairs.

    X[i] = data[i : i+look_back]   (look_back timesteps)
    y[i] = data[i+look_back]       (the next value)

    Returns
    -------
    X : (n_samples, look_back, 1)
    y : (n_samples,)
    """
    data = np.asarray(data).ravel()
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(data[i + look_back])
    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y)
    return X, y


def build_lstm(look_back: int = 60, units: int = 50, dropout: float = 0.2):
    """
    Build and compile a stacked LSTM regression model.

    Architecture:
        LSTM(units, return_sequences=True) -> Dropout
        LSTM(units)                        -> Dropout
        Dense(25, relu) -> Dense(1)
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    model = Sequential(
        [
            Input(shape=(look_back, 1)),
            LSTM(units, return_sequences=True),
            Dropout(dropout),
            LSTM(units, return_sequences=False),
            Dropout(dropout),
            Dense(25, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm(
    model,
    X_train,
    y_train,
    epochs: int = 30,
    batch_size: int = 32,
    validation_split: float = 0.1,
    verbose: int = 1,
):
    """Train the LSTM with early stopping on validation loss."""
    from tensorflow.keras.callbacks import EarlyStopping

    es = EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True, verbose=0
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[es],
        verbose=verbose,
    )
    return history


def predict_lstm(model, X):
    """Return flat predictions for input sequences X."""
    return model.predict(X, verbose=0).ravel()
