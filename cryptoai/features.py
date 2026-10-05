"""Turn raw candles into model inputs. Every feature only uses data up to that candle.

`build` describes one coin's own chart. `build_all` adds market context: what BTC is doing and how
each coin is performing against BTC and the other tracked coins, and optionally `intraday` patterns from
the 15m and 1h charts in the hours before each candle closes.
"""
import numpy as np
import pandas as pd


def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def build(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)

    for n in (1, 3, 6, 12, 24):
        f[f"ret_{n}"] = c.pct_change(n)

    f["rsi_14"] = _rsi(c, 14)
    f["rsi_7"] = _rsi(c, 7)

    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd"] = macd / c
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c

    for n in (20, 50, 200):
        f[f"dist_ema{n}"] = c / c.ewm(span=n).mean() - 1

    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (sma20 - 2 * std20)) / (4 * std20)
    f["bb_width"] = 4 * std20 / sma20

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c

    logret = np.log(c).diff()
    f["vol_10"] = logret.rolling(10).std()
    f["vol_50"] = logret.rolling(50).std()
    f["vol_ratio"] = f["vol_10"] / f["vol_50"]

    f["volume_z"] = (v - v.rolling(50).mean()) / v.rolling(50).std()
    f["range_pos"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min())
    f["candle_body"] = (c - df["open"]) / (h - l).replace(0, np.nan)

    # Longer-term trend and position.
    for n in (50, 100):
        f[f"ret_{n}"] = c.pct_change(n)
    f["from_high_100"] = c / h.rolling(100).max() - 1
    f["ema50_slope"] = c.ewm(span=50).mean().pct_change(10)

    f["dow"] = df.index.dayofweek
    f["hour"] = df.index.hour

    per_day = pd.Timedelta("1D") / (df.index[1] - df.index[0])  # candles per day: 6 on 4h, 1 on 1d

    # Order flow: share of volume from aggressive buyers (market orders hitting the ask).
    if "taker_buy_base" in df:
        tb = df["taker_buy_base"]
        f["taker_ratio"] = tb / v
        for n in (6, 24):
            f[f"taker_ratio_{n}"] = tb.rolling(n).sum() / v.rolling(n).sum()
        f["taker_z"] = _z(f["taker_ratio"], 50)
        f["trades_z"] = _z(df["trades"], 50)

    # Derivatives positioning: funding rate and perpetual premium over spot.
    if "funding" in df:
        fr = df["funding"]
        f["funding"] = fr
        f["funding_3d"] = fr.rolling(max(int(3 * per_day), 1)).mean()
        f["funding_z"] = _z(fr, int(90 * per_day))
    if "premium" in df:
        pr = df["premium"]
        f["premium"] = pr
        f["premium_7d"] = pr.rolling(int(7 * per_day)).mean()
        f["premium_z"] = _z(pr, int(90 * per_day))

    return f.replace([np.inf, -np.inf], np.nan)


def _z(s, n):
    return (s - s.rolling(n, min_periods=n // 3).mean()) / s.rolling(n, min_periods=n // 3).std()


MARKET = "BTC/USDT"
MARKET_COLS = ["ret_1", "ret_6", "ret_24", "dist_ema50", "dist_ema200", "vol_50", "rsi_14"]


def intraday(bars):
    """Patterns from the 15m and 1h charts, indexed by bar CLOSE time (known only once the bar has closed).

    `bars` is {"15m": candles, "1h": candles} for one coin.
    """
    m, h = bars["15m"].copy(), bars["1h"].copy()
    m.index = m.index + pd.Timedelta("15min")
    h.index = h.index + pd.Timedelta("1h")
    r = np.log(m["close"]).diff()
    f = pd.DataFrame(index=m.index)

    # Momentum over the last 15 minutes to 2 hours.
    for n, name in ((1, "15m"), (2, "30m"), (4, "1h"), (8, "2h")):
        f[f"m_ret_{name}"] = m["close"].pct_change(n)
    f["m_rsi_14"] = _rsi(m["close"])

    # Volatility: realised over 4h and 24h, and the share of it that came from up-moves.
    f["m_rv_4h"] = np.sqrt((r ** 2).rolling(16).sum())
    f["m_rv_24h"] = np.sqrt((r ** 2).rolling(96).sum())
    f["m_rv_ratio"] = f["m_rv_4h"] / (f["m_rv_24h"] / np.sqrt(6))
    up, down = (r.clip(lower=0) ** 2).rolling(96).sum(), (r.clip(upper=0) ** 2).rolling(96).sum()
    f["m_semivar_ratio"] = up / (up + down)
    f["m_skew_24h"] = r.rolling(96).skew()

    # Path shape: steady trend (efficiency near 1) or back-and-forth chop (near 0).
    f["m_efficiency_4h"] = np.log(m["close"]).diff(16).abs() / r.abs().rolling(16).sum()
    f["m_efficiency_24h"] = np.log(m["close"]).diff(96).abs() / r.abs().rolling(96).sum()
    f["m_up_share_4h"] = (r > 0).astype(float).rolling(16).mean()
    f["m_from_high_4h"] = m["close"] / m["high"].rolling(16).max() - 1
    f["m_from_low_4h"] = m["close"] / m["low"].rolling(16).min() - 1

    # Did activity and aggressive buying pick up in the last hour?
    f["m_vol_last_hour_share"] = m["volume"].rolling(4).sum() / m["volume"].rolling(16).sum()
    f["m_taker_last_hour"] = m["taker_buy_base"].rolling(4).sum() / m["volume"].rolling(4).sum()

    g = pd.DataFrame(index=h.index)
    g["h_rsi_14"] = _rsi(h["close"])
    g["h_dist_ema20"] = h["close"] / h["close"].ewm(span=20).mean() - 1
    macd = h["close"].ewm(span=12).mean() - h["close"].ewm(span=26).mean()
    g["h_macd_hist"] = (macd - macd.ewm(span=9).mean()) / h["close"]
    return f.replace([np.inf, -np.inf], np.nan), g.replace([np.inf, -np.inf], np.nan)


def _as_of(src, times):
    """Latest row of `src` at or before each time in `times`."""
    src = src[~src.index.duplicated()].sort_index()
    return src.reindex(src.index.union(times)).ffill().reindex(times)


def build_all(raw, intraday_bars=None):
    """Features for every coin in `raw` ({symbol: candles}), including market context.

    `raw` must contain BTC/USDT, and all coins should cover the same recent period. With `intraday_bars`
    ({symbol: {"15m": candles, "1h": candles}}), 15m/1h patterns up to each candle's close are added.
    """
    own = {s: build(df) for s, df in raw.items()}
    if intraday_bars is not None:
        for s, f in own.items():
            close_times = raw[s].index + (raw[s].index[1] - raw[s].index[0])
            f15, f1h = intraday(intraday_bars[s])
            extra = pd.concat([_as_of(f15, close_times), _as_of(f1h, close_times)], axis=1)
            extra.index = raw[s].index
            own[s] = f.join(extra)
    btc = own[MARKET][MARKET_COLS].add_prefix("btc_")
    rank = pd.DataFrame({s: df["close"].pct_change(24) for s, df in raw.items()}).rank(axis=1, pct=True)
    if all("funding" in f for f in own.values()):
        market_funding = pd.DataFrame({s: f["funding_3d"] for s, f in own.items()}).mean(axis=1)
    else:
        market_funding = None

    out = {}
    for s, f in own.items():
        f = f.join(btc)
        f["rel_btc_24"] = f["ret_24"] - f["btc_ret_24"]
        f["rel_btc_6"] = f["ret_6"] - f["btc_ret_6"]
        f["xs_rank_24"] = rank[s]
        if market_funding is not None:
            f["market_funding_3d"] = market_funding
        out[s] = f
    return out


def target(df, horizon):
    """1 if price is higher `horizon` candles from now, else 0. NaN where the future is unknown."""
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    y = (fwd > 0).astype(float)
    y[fwd.isna()] = np.nan
    return y
