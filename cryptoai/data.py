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


_loaded = {}  # path -> (modification time, DataFrame): parsing the big 15m files takes about a second each


def _load(path):
    """Read a cache file, reusing the parsed copy while the file is unchanged. Callers must not modify it."""
    if not path.exists():
        return pd.DataFrame()
    mtime = path.stat().st_mtime_ns
    hit = _loaded.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    df = pd.read_csv(path, index_col="time")
    df.index = pd.to_datetime(df.index, format="ISO8601", utc=True).as_unit("ms")
    _loaded[path] = (mtime, df)
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
    old = df
    df = pd.concat([df, new]) if not df.empty else new
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.as_unit("ms")
    if (not old.empty and path.exists() and list(new.columns) == list(old.columns)
            and new.index.min() >= old.index[-1]):
        # Usual case: only the last cached row (re-fetched) and newer rows changed. Rewrite just the tail of the
        # file instead of the whole history (whole rewrites of the 15m files took ~20 s per update).
        tail = df[df.index >= old.index[-1]]
        with open(path, "rb+") as f:
            f.seek(0, 2)
            end = f.tell()
            f.seek(max(end - 4096, 0))
            chunk = f.read()
            cut = chunk.rstrip(b"\r\n").rfind(b"\n")  # start of the last data line
            f.seek(max(end - 4096, 0) + cut + 1)
            f.truncate()
            f.write(tail.to_csv(header=False, lineterminator="\n").encode())
    else:
        with config.atomic(path) as tmp:
            df.to_csv(tmp)
    _loaded[path] = (path.stat().st_mtime_ns, df)  # keep the parsed copy current; no re-read needed
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


def closed(timeframe, refresh=True, symbols=config.SYMBOLS, refresh_derivatives=None):
    """Closed candles with funding and premium for every symbol, as {symbol: DataFrame}.

    `refresh_derivatives=False` downloads new candles but uses the cached funding and premium (unused by the
    models, and slow to fetch), for quick decisions at a candle close.
    """
    refresh_derivatives = refresh if refresh_derivatives is None else refresh_derivatives
    out = {}
    for s in symbols:
        df = drop_open_candle(update(s, timeframe) if refresh else load_cached(s, timeframe), timeframe)
        out[s] = with_derivatives(df, s, timeframe, refresh_derivatives)
    return out


def intraday(refresh=True, since=None, symbols=config.SYMBOLS):
    """15m and 1h candles for every symbol, as {symbol: {timeframe: DataFrame}}, optionally from `since`."""
    out = {}
    for s in symbols:
        out[s] = {}
        for tf in config.INTRADAY_TIMEFRAMES:
            df = update(s, tf) if refresh else load_cached(s, tf)
            out[s][tf] = df[df.index >= since] if since is not None else df
    return out


def latest(symbol, timeframe, limit=50):
    """The most recent candles straight from Binance, in memory only (nothing is written to the cache).

    The last row is the candle still forming.
    """
    rows = exchange().publicGetKlines({"symbol": symbol.replace("/", ""), "interval": timeframe, "limit": limit})
    df = pd.DataFrame([[r[0], *r[1:6], r[7], r[8], r[9]] for r in rows], columns=["time", *KLINE_COLS])
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True).dt.as_unit("ms")
    return df.set_index("time").astype(float)


def prices(symbols):
    """Latest prices for a list of symbols."""
    tickers = exchange().fetch_tickers(symbols)
    return {s: t["last"] for s, t in tickers.items()}
