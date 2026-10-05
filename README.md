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
| `daily.bat` | Daily learning: fetches new candles, retrains, keeps the new model only if it still has an edge, logs signals to `logs/daily.log` |

The AI learns every day: `schedule_daily.ps1` registers `daily.bat` with Windows Task Scheduler to run at
00:10 UTC (07:10 in UTC+7), right after Binance's daily candle closes. If the PC is off, it runs when it
is next on. Every run is recorded in `models/training_log.csv` and shown under **Backtest > Learning history**.
Remove it with `powershell -ExecutionPolicy Bypass -File schedule_daily.ps1 -Remove`.

To get alerts automatically, schedule `check_signals.bat` with Windows Task Scheduler to run
every 4 hours, a few minutes after each Binance 4h candle closes (00:00, 04:00, 08:00 … UTC).

Command line: `.venv\Scripts\python -m cryptoai [train|signals|backtest|market|daily]`

## How it works

- `cryptoai/data.py`: public Binance data, cached in `data/`. No API key needed. Spot candles with taker-buy
  volume and trade count, plus futures funding rates and premium index (downloaded but currently unused by the
  model, see `UNUSED_FEATURES` in `config.py`).
- `cryptoai/features.py`: about 40 inputs. Each coin's own chart (returns, RSI, MACD, EMA distance, Bollinger,
  ATR, volatility, volume, long-term trend, distance from recent high, day and hour) plus market context:
  what BTC is doing, and each coin's strength against BTC and the other tracked coins.
- `cryptoai/model.py`: gradient-boosted trees, one model per timeframe, trained on all coins pooled.
  Evaluated **walk-forward**: each test period is predicted by a model trained only on earlier data.
- `cryptoai/backtest.py`: long-or-flat with 0.1% fee per side, compared with buy and hold.
- `cryptoai/config.py`: coins, timeframes, horizon, thresholds. Edit here.

## Read this before trusting it

Out-of-sample accuracy is about 52% (AUC about 0.53). That is a small edge, which is normal
for honest price prediction. Anyone who shows you 70%+ is usually leaking future data into the test.
Backtest results differ a lot between coins and are partly luck. Treat signals as one input,
and never risk money you can't afford to lose.
