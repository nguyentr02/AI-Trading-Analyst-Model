# Crypto AI: signals and tracker

Downloads Binance candles, trains a model to estimate the chance price rises over the next
day (4h model) or 3 days (1d model), backtests it honestly, and shows everything in a dashboard
alongside your holdings. It **does not place trades**. You make every decision.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
train.bat
```

Training downloads the price history into `data/` and writes the models to `models/`.

## Use it

| Double-click | What it does |
|---|---|
| `dashboard.bat` | Opens the dashboard at http://localhost:8501 |
| `train.bat` | Downloads new data, retrains, prints backtest (do it weekly or monthly) |
| `check_signals.bat` | Logs current signals and any changes to `signals_output.txt` |

To get alerts automatically, schedule `check_signals.bat` with Windows Task Scheduler to run
every 4 hours, a few minutes after each Binance 4h candle closes (00:00, 04:00, 08:00 … UTC).

Command line: `.venv\Scripts\python -m cryptoai [train|signals|backtest]`

## How it works

- `cryptoai/data.py`: public Binance OHLCV through ccxt, cached in `data/`. No API key needed.
- `cryptoai/features.py`: about 25 indicators (returns, RSI, MACD, EMA distance, Bollinger, ATR, volatility, volume).
- `cryptoai/model.py`: gradient-boosted trees, one model per timeframe, trained on all coins pooled.
  Evaluated **walk-forward**: each test period is predicted by a model trained only on earlier data.
- `cryptoai/backtest.py`: long-or-flat with 0.1% fee per side, compared with buy and hold.
- `cryptoai/config.py`: coins, timeframes, horizon, thresholds. Edit here.

## Read this before trusting it

Out-of-sample accuracy is about 51–52% (AUC about 0.52–0.53). That is a small edge, which is normal
for honest price prediction. Anyone who shows you 70%+ is usually leaking future data into the test.
Backtest results differ a lot between coins and are partly luck. Treat signals as one input,
and never risk money you can't afford to lose.
