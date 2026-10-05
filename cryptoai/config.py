"""Central settings. Edit these to change what the AI tracks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
PORTFOLIO_FILE = ROOT / "portfolio.json"
SIGNAL_LOG = ROOT / "signals_log.csv"

EXCHANGE = "binance"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
TIMEFRAMES = ["4h", "1d"]

# How far ahead the model predicts, in candles (4h x 6 = 1 day, 1d x 3 = 3 days).
HORIZON = {"4h": 6, "1d": 3}

# How much history to download the first time.
HISTORY_START = "2019-01-01T00:00:00Z"

# Trading assumptions used in the backtest.
FEE = 0.001  # 0.1% per side (Binance spot taker)
ENTER_PROB = 0.55  # go long when P(up) rises above this
EXIT_PROB = 0.48  # go flat when P(up) falls below this

for d in (DATA_DIR, MODEL_DIR):
    d.mkdir(exist_ok=True)
