"""Download and cache OHLCV candles from the exchange (public data, no API key needed)."""
import time

import ccxt
import pandas as pd

from . import config

_exchange = None


def exchange():
    global _exchange
    if _exchange is None:
        _exchange = getattr(ccxt, config.EXCHANGE)({"enableRateLimit": True})
    return _exchange


def _cache_path(symbol, timeframe):
    return config.DATA_DIR / f"{symbol.replace('/', '_')}_{timeframe}.csv"


def load_cached(symbol, timeframe):
    path = _cache_path(symbol, timeframe)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    return df


def update(symbol, timeframe):
    """Fetch any candles newer than the cache, save, and return the full history."""
    ex = exchange()
    df = load_cached(symbol, timeframe)
    if df.empty:
        since = ex.parse8601(config.HISTORY_START)
    else:
        # Re-fetch the last cached candle; it may have been incomplete when saved.
        since = int(df.index[-1].timestamp() * 1000)

    rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        if len(batch) < 1000:
            break
        since = batch[-1][0] + 1
        time.sleep(ex.rateLimit / 1000)

    if rows:
        new = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        new["time"] = pd.to_datetime(new["time"], unit="ms", utc=True)
        new = new.set_index("time")
        df = pd.concat([df, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(_cache_path(symbol, timeframe))
    return df


def drop_open_candle(df, timeframe):
    """Remove the still-forming last candle so the model only sees closed candles."""
    if df.empty:
        return df
    tf_ms = exchange().parse_timeframe(timeframe) * 1000
    now = pd.Timestamp.now(tz="UTC")
    if df.index[-1] + pd.Timedelta(milliseconds=tf_ms) > now:
        return df.iloc[:-1]
    return df


def closed(timeframe, refresh=True, symbols=config.SYMBOLS):
    """Closed candles for every symbol, as {symbol: DataFrame}."""
    return {
        s: drop_open_candle(update(s, timeframe) if refresh else load_cached(s, timeframe), timeframe)
        for s in symbols
    }


def prices(symbols):
    """Latest prices for a list of symbols."""
    tickers = exchange().fetch_tickers(symbols)
    return {s: t["last"] for s, t in tickers.items()}
