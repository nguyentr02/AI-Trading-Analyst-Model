"""Download and cache market data from Binance (public data, no API key needed).

Spot candles include taker-buy volume and trade count. Futures funding rates and the perpetual
premium index are joined onto the candles in `closed`.
"""
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


def _load(path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col="time")
    df.index = pd.to_datetime(df.index, format="ISO8601", utc=True).as_unit("ms")
    return df


def load_cached(symbol, timeframe):
    return _load(_cache_path(symbol, timeframe))


def _paged(call, params, since, limit, time_of):
    """Fetch every row from `since` onward from a Binance endpoint that pages by startTime."""
    ex = exchange()
    rows = []
    while True:
        batch = call({**params, "startTime": since, "limit": limit})
        if not batch:
            break
        rows += batch
        if len(batch) < limit:
            break
        since = int(time_of(batch[-1])) + 1
        time.sleep(ex.rateLimit / 1000)
    return rows


def _since(df):
    if df.empty:
        return exchange().parse8601(config.HISTORY_START)
    # Re-fetch the last cached row; it may have been incomplete when saved.
    return int(df.index[-1].timestamp() * 1000)


def _merge_save(df, new, path):
    df = pd.concat([df, new]) if not df.empty else new
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.as_unit("ms")
    df.to_csv(path)
    return df


KLINE_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]


def update(symbol, timeframe):
    """Fetch any candles newer than the cache, save, and return the full history."""
    df = load_cached(symbol, timeframe)
    if not df.empty and "taker_buy_base" not in df:
        df = pd.DataFrame()  # cache from before taker volume was stored: download again
    rows = _paged(exchange().publicGetKlines, {"symbol": symbol.replace("/", ""), "interval": timeframe},
                  _since(df), 1000, lambda r: r[0])
    if not rows:
        return df
    # Binance kline fields: open time, O, H, L, C, volume, close time, quote volume, trades, taker buy base, ...
    new = pd.DataFrame([[r[0], *r[1:6], r[7], r[8], r[9]] for r in rows], columns=["time", *KLINE_COLS])
    new["time"] = pd.to_datetime(new["time"].astype("int64"), unit="ms", utc=True)
    new = new.set_index("time").astype(float)
    return _merge_save(df, new, _cache_path(symbol, timeframe))


def funding(symbol, refresh=True):
    """Perpetual futures funding rate history (one row per settlement, usually every 8h)."""
    path = config.DATA_DIR / f"funding_{symbol.replace('/', '_')}.csv"
    df = _load(path)
    if not refresh:
        return df
    rows = _paged(exchange().fapiPublicGetFundingRate, {"symbol": symbol.replace("/", "")},
                  _since(df), 1000, lambda r: r["fundingTime"])
    if not rows:
        return df
    new = pd.DataFrame({"time": [int(r["fundingTime"]) for r in rows],
                        "funding": [float(r["fundingRate"]) for r in rows]})
    new["time"] = pd.to_datetime(new["time"], unit="ms", utc=True)
    return _merge_save(df, new.set_index("time"), path)


def premium(symbol, timeframe, refresh=True):
    """Perpetual premium index candles: (perp price - spot index) / index, closing value per candle."""
    path = config.DATA_DIR / f"premium_{symbol.replace('/', '_')}_{timeframe}.csv"
    df = _load(path)
    if not refresh:
        return df
    rows = _paged(exchange().fapiPublicGetPremiumIndexKlines,
                  {"symbol": symbol.replace("/", ""), "interval": timeframe}, _since(df), 1500, lambda r: r[0])
    if not rows:
        return df
    new = pd.DataFrame({"time": [int(r[0]) for r in rows], "premium": [float(r[4]) for r in rows]})
    new["time"] = pd.to_datetime(new["time"], unit="ms", utc=True)
    return _merge_save(df, new.set_index("time"), path)


def with_derivatives(df, symbol, timeframe, refresh=True):
    """Join funding (latest settlement at or before each candle's close) and premium onto candles."""
    if df.empty:
        return df
    close_time = df.index + pd.Timedelta(timeframe)
    fr = funding(symbol, refresh)
    if not fr.empty:
        fr = fr.sort_index()
        pos = fr.index.searchsorted(close_time, side="right") - 1
        df = df.assign(funding=[fr["funding"].iloc[i] if i >= 0 else float("nan") for i in pos])
    pr = premium(symbol, timeframe, refresh)
    if not pr.empty:
        df = df.join(pr["premium"])
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
    """Closed candles with funding and premium for every symbol, as {symbol: DataFrame}."""
    out = {}
    for s in symbols:
        df = drop_open_candle(update(s, timeframe) if refresh else load_cached(s, timeframe), timeframe)
        out[s] = with_derivatives(df, s, timeframe, refresh)
    return out


def prices(symbols):
    """Latest prices for a list of symbols."""
    tickers = exchange().fetch_tickers(symbols)
    return {s: t["last"] for s, t in tickers.items()}
