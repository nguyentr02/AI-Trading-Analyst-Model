"""Altcoins: a live scan for the biggest movers, what testing found, and an experimental AI reading.

Nothing here trades. Testing on 2026-10-06 (experiments/altcoins_near_zec.py, experiments/trend_universe.py)
found that the AI reads NEAR and ZEC only slightly better than a coin flip, that the 50-day trend rule beat it
by a wide margin on them, and that the trend rule on the most-traded alts did far worse than on BTC/ETH/BNB/SOL.
"""
import json
import time

import joblib
import pandas as pd

from . import config, data, droprisk, features, model

# Altcoins with full history downloaded: the experimental AI reading and the altcoin paper trial use these.
# NEAR and ZEC were tested (experiments/altcoins_near_zec.py); the others are large, liquid, long-listed alts.
WATCH = ["NEAR/USDT", "ZEC/USDT", "XRP/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT"]
COLORS = {"NEAR/USDT": "#00C08B", "ZEC/USDT": "#F4B728", "XRP/USDT": "#23292F", "DOGE/USDT": "#C2A633",
          "AVAX/USDT": "#E84142", "LINK/USDT": "#2A5ADA"}
SIGNALS_FILE = config.LOG_DIR / "alt_signals.json"
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


def _predict(bundle, row):
    return float(bundle["model"].predict_proba(row.reindex(columns=bundle["features"]))[:, 1][0])


def compute(symbols=WATCH, refresh=True):
    """Experimental: the live models (trained on BTC/ETH/BNB/SOL) applied to each alt, plus its trend status.

    Per alt, at the last closed candles: P(up, next 1 day) from the 4h model, P(up, next 3 days) from the daily
    model, the drop warning, and the daily close vs its 50-day average. Downloads new candles first (refresh) and
    saves the result to logs/alt_signals.json, which the dashboard and the altcoin paper trial read. The cross-coin
    rank feature is computed among the 4 coins plus the alts, slightly different from training, which is one more
    reason to treat this as experimental.
    """
    if refresh:
        for s in symbols:
            for tf in ("4h", "1d", *config.INTRADAY_TIMEFRAMES):
                data.update(s, tf)
    every = [*config.SYMBOLS, *symbols]
    m4, m1d = model.load("4h"), model.load("1d")
    mdrop = joblib.load(droprisk.MODEL_FILE) if droprisk.MODEL_FILE.exists() else None
    raw4 = {s: data.drop_open_candle(data.load_cached(s, "4h"), "4h").tail(400) for s in every}
    since = min(df.index[0] for df in raw4.values()) - pd.Timedelta("2D")
    f4 = features.build_all(raw4, data.intraday(False, since=since, symbols=every))
    raw1d = {s: data.drop_open_candle(data.load_cached(s, "1d"), "1d").tail(400) for s in every}
    f1d = features.build_all(raw1d, None)
    coins = {}
    for s in symbols:
        last4, last1d = f4[s].iloc[[-1]], f1d[s].iloc[[-1]]
        d = raw1d[s]["close"]
        avg = {n: float(d.tail(n).mean()) for n in (20, 50, 100, 200)}
        coins[s] = {"candle_4h": str(last4.index[0]), "candle_1d": str(last1d.index[0]),
                    "price": float(raw4[s]["close"].iloc[-1]), "daily_close": float(d.iloc[-1]),
                    "above": {str(n): bool(d.iloc[-1] > a) for n, a in avg.items()},
                    "from_50d": float(d.iloc[-1] / avg[50] - 1),
                    "p_up_1d": _predict(m4, last4) if m4 else None,
                    "p_up_3d": _predict(m1d, last1d) if m1d else None,
                    "p_drop": _predict(mdrop, last4) if mdrop else None}
    out = {"updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "coins": coins}
    with config.atomic(SIGNALS_FILE) as tmp:
        tmp.write_text(json.dumps(out, indent=2))
    return out


def load():
    try:
        return json.loads(SIGNALS_FILE.read_text())
    except (OSError, ValueError):
        return None


def readings(max_age=pd.Timedelta("4h30min")):
    """The latest readings as a table, one row per alt: the saved ones (the live service refreshes them at every
    4h close) if recent and complete, otherwise freshly computed."""
    saved = load()
    if (saved is None or set(saved["coins"]) != set(WATCH)
            or pd.Timestamp.now(tz="UTC") - pd.Timestamp(saved["updated"]) > max_age):
        saved = compute()
    rows = []
    for s, r in saved["coins"].items():
        rows.append({"symbol": s, "candle": pd.Timestamp(r["candle_4h"]), "price": r["price"],
                     "above_50d": r["above"]["50"], "from_50d": r["from_50d"], "p_up_1d": r["p_up_1d"],
                     "p_up_3d": r["p_up_3d"], "p_drop": r["p_drop"]})
    return pd.DataFrame(rows)
