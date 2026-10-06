"""Altcoins: a live scan for the biggest movers, what testing found, and an experimental AI reading.

Nothing here trades. Testing on 2026-10-06 (experiments/altcoins_near_zec.py, experiments/trend_universe.py)
found that the AI reads NEAR and ZEC only slightly better than a coin flip, that the 50-day trend rule beat it
by a wide margin on them, and that the trend rule on the most-traded alts did far worse than on BTC/ETH/BNB/SOL.
"""
import time

import joblib
import pandas as pd

from . import config, data, droprisk, features, model

WATCH = ["NEAR/USDT", "ZEC/USDT"]  # alts with downloaded history, for the experimental reading
MIN_VOLUME = 20e6  # USDT per day: below this, prices are easy to push around
NOT_ALTS = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "EUR", "EURI", "AEUR", "USDE", "XUSD", "BFUSD", "PAXG", "WBTC",
            "WBETH", "USD1", "RLUSD", "USDS", "BTC", "ETH", "BNB", "SOL"}

# Results of the tests, as run on 2026-10-06 (check period 2025-01-01 to 2026-10-06, $1000 per coin, after fees).
NEAR_ZEC = pd.DataFrame([
    ("AI trained on BTC/ETH/BNB/SOL only", 1784, 2205, 1.10, 0.521),
    ("AI trained with NEAR and ZEC too (chosen on 2022-2024)", 938, 2185, 0.75, 0.518),
    ("50-day trend rule", 1461, 18411, 1.83, None),
    ("Buy & hold", 1001, 22953, 1.54, None),
], columns=["Strategy", "NEAR $", "ZEC $", "Sharpe (both)", "AI accuracy (AUC)"])
UNIVERSE = pd.DataFrame([
    ("BTC, ETH, BNB, SOL (the live coins)", 0.98, 0.93, 0.528),
    ("Top 4 by trading volume", 0.15, 0.39, 0.142),
    ("Top 8", 0.40, 0.25, 0.050),
    ("Top 12", 0.35, 0.19, 0.017),
    ("Top 16 (chosen on 2022-2024)", 0.55, 0.28, 0.070),
], columns=["50-day rule on", "Sharpe 2022-2024", "Sharpe 2025-2026", "Return 2025-2026"])


def scan(top=15):
    """Liquid alt/USDT pairs on Binance ranked by 30-day daily volatility. Takes ~10 s (one request per coin)."""
    ex = data.exchange()
    rows = []
    for sym, t in ex.fetch_tickers().items():
        base = sym.split("/")[0]
        if (not sym.endswith("/USDT") or ":" in sym or base in NOT_ALTS or base.endswith(("UP", "DOWN", "BULL", "BEAR"))
                or (t.get("quoteVolume") or 0) < MIN_VOLUME or not t.get("last")):
            continue
        rows.append((sym, t["last"], (t.get("percentage") or 0) / 100, t["quoteVolume"]))
    out = []
    for sym, price, chg, vol in sorted(rows, key=lambda r: -r[3])[:60]:
        try:
            d = pd.DataFrame(ex.fetch_ohlcv(sym, "1d", limit=91), columns=["t", "o", "h", "l", "c", "v"]).iloc[:-1]
        except Exception:
            continue
        if len(d) < 60:
            continue
        ret = d["c"].pct_change().dropna()
        out.append({"symbol": sym, "price": price, "change_24h": chg, "volume_24h": vol,
                    "change_30d": d["c"].iloc[-1] / d["c"].iloc[-31] - 1,
                    "change_90d": d["c"].iloc[-1] / d["c"].iloc[0] - 1,
                    "daily_vol": ret.tail(30).std(), "above_50d": bool(d["c"].iloc[-1] > d["c"].tail(50).mean()),
                    "days_10pct": int((ret.abs() > 0.10).sum())})
        time.sleep(0.05)
    return pd.DataFrame(out).sort_values("daily_vol", ascending=False).head(top)


def readings(symbols=WATCH):
    """Experimental: the live models (trained on BTC/ETH/BNB/SOL) applied to each alt, plus its trend status.

    Downloads any new candles for the alts first. The cross-coin rank feature is computed among the 4 coins plus
    the alts, slightly different from training, which is one more reason to treat this as experimental.
    """
    p1 = model.load("4h")
    pdrop = joblib.load(droprisk.MODEL_FILE) if droprisk.MODEL_FILE.exists() else None
    for s in symbols:
        for tf in ("4h", "1d", *config.INTRADAY_TIMEFRAMES):
            data.update(s, tf)
    every = [*config.SYMBOLS, *symbols]
    raw = {s: data.drop_open_candle(data.load_cached(s, "4h"), "4h").tail(400) for s in every}
    since = min(df.index[0] for df in raw.values()) - pd.Timedelta("2D")
    feats = features.build_all(raw, data.intraday(False, since=since, symbols=every))
    out = []
    for s in symbols:
        last = feats[s].iloc[[-1]]
        d = data.drop_open_candle(data.load_cached(s, "1d"), "1d")["close"]
        avg50 = d.tail(50).mean()
        row = {"symbol": s, "candle": last.index[0], "price": float(raw[s]["close"].iloc[-1]),
               "above_50d": bool(d.iloc[-1] > avg50), "from_50d": float(d.iloc[-1] / avg50 - 1)}
        if p1 is not None:
            row["p_up_1d"] = float(p1["model"].predict_proba(last.reindex(columns=p1["features"]))[:, 1][0])
        if pdrop is not None:
            row["p_drop"] = float(pdrop["model"].predict_proba(last.reindex(columns=pdrop["features"]))[:, 1][0])
        out.append(row)
    return pd.DataFrame(out)
