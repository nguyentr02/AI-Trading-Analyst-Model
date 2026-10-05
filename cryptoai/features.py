"""Turn raw candles into model inputs. Every feature only uses data up to that candle."""
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

    return f.replace([np.inf, -np.inf], np.nan)


def target(df, horizon):
    """1 if price is higher `horizon` candles from now, else 0. NaN where the future is unknown."""
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    y = (fwd > 0).astype(float)
    y[fwd.isna()] = np.nan
    return y
