"""Central settings. Edit these to change what the AI tracks."""
import os
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
PORTFOLIO_FILE = ROOT / "portfolio.json"
SIGNAL_LOG = ROOT / "signals_log.csv"

EXCHANGE = "binance"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]

# CoinGecko id for each symbol, used for market cap and supply data.
COINGECKO_IDS = {"BTC/USDT": "bitcoin", "ETH/USDT": "ethereum", "BNB/USDT": "binancecoin", "SOL/USDT": "solana"}

# How often the dashboard's Market tab redraws, in seconds. Binance pushes new prices once per second.
LIVE_REFRESH = 1
# Candle timeframes the models are built on. The live service learns when one of these closes.
TIMEFRAMES = ["4h", "1d"]
# Shorter charts whose recent patterns are fed to the models marked "intraday" below.
INTRADAY_TIMEFRAMES = ["15m", "1h"]

# The models. Each predicts whether price is higher `horizon` candles of `timeframe` from now.
# "4h" and "1d" keep their original names so existing model files and logs stay valid.
MODELS = {
    "4h_next": {"timeframe": "4h", "horizon": 1, "label": "Next 4 hours", "intraday": True},
    "4h": {"timeframe": "4h", "horizon": 6, "label": "Next 1 day", "intraday": True},
    "1d": {"timeframe": "1d", "horizon": 3, "label": "Next 3 days", "intraday": False},
}

# Features that are computed but not fed to the model. These were tested on 2026-10-05 and lowered
# walk-forward AUC (see docs/research/reading-crypto-charts-for-day-trading.md, "Results in this repo").
# Remove names from this list to test them again.
UNUSED_FEATURES = [
    "taker_ratio", "taker_ratio_6", "taker_ratio_24", "taker_z", "trades_z",
    "funding", "funding_3d", "funding_z", "premium", "premium_7d", "premium_z", "market_funding_3d",
]

# How much history to download the first time.
HISTORY_START = "2019-01-01T00:00:00Z"

# Daily learning: a newly trained model replaces the current one only if its walk-forward AUC
# is at least this (0.5 = no better than a coin flip). Otherwise the previous model is kept.
MIN_AUC = 0.505
TRAINING_LOG = MODEL_DIR / "training_log.csv"

# Trading assumptions used in the backtest.
FEE = 0.001  # 0.1% per side (Binance spot taker)
ENTER_PROB = 0.55  # go long when P(up) rises above this
EXIT_PROB = 0.48  # go flat when P(up) falls below this

LOG_DIR = ROOT / "logs"
LIVE_STATUS = LOG_DIR / "live_status.json"

for d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(exist_ok=True)


@contextmanager
def atomic(path):
    """Yield a temporary path, then move it over `path` in one step.

    The live service and the dashboard share these files; this stops either from reading a half-written one.
    On Windows the swap fails while another process has the file open, so it retries for a few seconds.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # per process, so two writers never collide
    yield tmp
    for attempt in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.1)
