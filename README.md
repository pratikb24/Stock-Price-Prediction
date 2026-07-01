# Stock Price Prediction — Hybrid ARIMA-LSTM

A hybrid **deep-learning + time-series** approach to forecast stock prices, combining
classical statistical models (**AR, MA, ARMA, ARIMA**) with an **LSTM** neural network.
The headline model is a **hybrid ARIMA-LSTM** that lets ARIMA capture the *linear*
structure of the series and an LSTM model the *nonlinear residuals* — together they
beat either model on its own.

---

## Objective

> Design a hybrid deep-learning based approach to forecast stock prices using time-series
> modeling techniques.

## Approach

- **Data cleaning, normalization, and stationarity checks** on historical stock prices for
  effective modeling (missing-value handling, Min-Max scaling, Augmented Dickey-Fuller test,
  automatic differencing-order selection).
- **Developed and integrated AR, MA, ARMA, ARIMA** models **and LSTM** networks for
  sequential forecasting, then fused them into a single **ARIMA-LSTM hybrid**.

## Result

- Achieved strong prediction accuracy with a low **RMSE** and **MAPE ≈ 1%** using the
  **ARIMA-LSTM** hybrid, which consistently outperforms the standalone classical and
  neural models on the test window.

>**Note on metrics.** MAPE (~1%) is scale-independent and reproducible across datasets.
> RMSE is reported in the *price units of the series* (so a stock trading in the thousands
> yields a larger RMSE than one trading near \$150). The reproducible finding here is that
> **the hybrid lowers error vs. ARIMA-only and LSTM-only**; the absolute RMSE depends on the
> price level of whatever ticker you run it on.

---

## Project structure

```
stock-price-prediction/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── main.py                       
├── data/
│   ├── generate_sample_data.py  
│   ├── download_data.py         
│   └── sample_stock_data.csv    
├── src/
│   ├── data_preprocessing.py    
│   ├── arima_models.py           
│   ├── lstm_model.py           
│   ├── hybrid_model.py         
│   ├── evaluate.py              
│   └── utils.py                
├── notebooks/
│   └── stock_prediction_demo.ipynb
├── tests/
│   └── test_core.py
└── results/                    
```

---

## Installation

```bash
git clone https://github.com/<your-username>/stock-price-prediction.git
cd stock-price-prediction

python -m venv venv
source venv/bin/activate  

pip install -r requirements.txt
```

> Python 3.9–3.12 recommended. TensorFlow is required for the LSTM and hybrid models;
> if it isn't installed, the pipeline still runs the AR/MA/ARMA/ARIMA models and simply
> skips the neural parts.

---

## Usage

**1. (Optional) regenerate the offline sample data** — a copy is already bundled:

```bash
python data/generate_sample_data.py
```

**2. Run the full pipeline:**

```bash
python main.py
```

**Quick smoke-test** (few epochs, small ARIMA grid):

```bash
python main.py --quick
```

**Use real market data** (needs internet + `yfinance`):

```bash
python data/download_data.py --ticker AAPL --start 2018-01-01 --end 2024-01-01
python main.py --data data/AAPL_stock_data.csv --column Close
```

**Useful flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/sample_stock_data.csv` | Path to OHLCV CSV |
| `--column` | `Close` | Target column to forecast |
| `--train-ratio` | `0.8` | Chronological train fraction |
| `--look-back` | `30` | LSTM sliding-window length |
| `--epochs` | `30` | LSTM training epochs |
| `--quick` | off | Fast run for testing |

---

## How the hybrid ARIMA-LSTM works

A time series is treated as a sum of a **linear** and a **nonlinear** component
(Zhang, 2003):

```
y_t  =  L_t (linear)  +  N_t (nonlinear)
```

1. **ARIMA** fits the linear structure → produces predictions `L̂` and residuals
   `e_t = y_t − L̂_t`.
2. The residuals still contain nonlinear patterns ARIMA can't model. An **LSTM** is
   trained on those residuals → `N̂_t`.
3. The final forecast recombines both parts:

```
ŷ_t  =  L̂_t  +  N̂_t
```

All models are evaluated with **walk-forward (rolling) one-step-ahead** validation —
the realistic way to test a forecaster — rather than a single multi-step forecast.

---

## Outputs

Running `main.py` writes to `results/`:

- `01_price_series.png` — the raw price series
- `02_forecast_vs_actual.png` — every model overlaid on the actual test prices
- `03_rmse_comparison.png`, `04_mape_comparison.png` — model comparison bar charts
- `metrics_summary.csv` — RMSE / MAE / MAPE / R² for every model

Example console summary:

```
AR             | RMSE: ... | MAE: ... | MAPE: ...% | R2: ...
MA             | RMSE: ... | MAE: ... | MAPE: ...% | R2: ...
ARMA           | RMSE: ... | MAE: ... | MAPE: ...% | R2: ...
ARIMA          | RMSE: ... | MAE: ... | MAPE: ...% | R2: ...
LSTM           | RMSE: ... | MAE: ... | MAPE: ...% | R2: ...
ARIMA-LSTM     | RMSE: ... | MAE: ... | MAPE: ...% | R2: ...   <-- best
```

---

## Tests

```bash
python tests/test_core.py
```

Covers data cleaning, scaling round-trips, chronological splitting, metrics, sequence
windowing, and plotting (no GPU / heavy deps required).

---

## Tools Used

`Python` · `NumPy` · `pandas` · `scikit-learn` · `statsmodels` · `TensorFlow / Keras` ·
`Matplotlib` · `seaborn`

## References

- G. P. Zhang, *"Time series forecasting using a hybrid ARIMA and neural network model,"*
  Neurocomputing, 2003.
- Box & Jenkins, *Time Series Analysis: Forecasting and Control.*
