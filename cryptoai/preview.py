"""Live preview: re-run the current models on the live price, every minute.

The models only retrain when a candle closes. In between, this treats the candle still forming as if it
closed now, adds the latest closed 15m and 1h bars, and asks each model for its P(up). The result is a
PROVISIONAL signal: a half-formed candle (partial volume, a close that is still moving) is not what the
models were trained on, so it shows where the signal is heading, not a confirmed signal. Alerts and the
portfolio advice stay on the confirmed signals.

Written to logs/live_preview.json; the live service refreshes it every PREVIEW_EVERY seconds.
"""
import json
from datetime import datetime, timezone

import pandas as pd

from . import config, data, model, signals

HISTORY = 650  # candles of history per coin: enough for the 200-candle averages plus warm-up


def _with_live(cache, live):
    """Cached candles plus any newer ones from Binance (the newest may still be forming)."""
    cols = [c for c in live.columns if c in cache.columns] if len(cache) else list(live.columns)
    merged = pd.concat([cache[cols], live[cols]]) if len(cache) else live
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def compute():
    """Provisional P(up) for every model and coin, from the live price. Returns the dict that is saved."""
    now = pd.Timestamp.now(tz="UTC")
    fetched = {}

    def live(sym, tf, limit):
        if (sym, tf) not in fetched:
            fetched[(sym, tf)] = data.latest(sym, tf, limit)
        return fetched[(sym, tf)]

    out = {"as_of": now.isoformat(timespec="seconds"), "models": {}}
    for name, spec in config.MODELS.items():
        if model.load(name) is None:
            continue
        tf = spec["timeframe"]
        raw = {}
        for sym in config.SYMBOLS:
            cache = data.load_cached(sym, tf).iloc[-HISTORY:]
            raw[sym] = _with_live(cache, live(sym, tf, 5))  # last row = the forming candle
        bars = None
        if spec["intraday"]:
            since = min(df.index[0] for df in raw.values()) - pd.Timedelta("2D")
            bars = {}
            for sym in config.SYMBOLS:
                bars[sym] = {}
                for itf in config.INTRADAY_TIMEFRAMES:
                    cache = data.load_cached(sym, itf)
                    fresh = data.drop_open_candle(live(sym, itf, 60), itf)  # only bars that have closed
                    bars[sym][itf] = _with_live(cache[cache.index >= since], fresh)
        probs = model.predict_history(raw, name, bars=bars)
        step = pd.Timedelta(tf)
        out["models"][name] = {}
        for sym, df in raw.items():
            opened = df.index[-1]
            prob = float(probs[sym].iloc[-1])
            out["models"][name][sym] = {
                "prob_up": round(prob, 3), "signal": signals.label(prob), "price": float(df["close"].iloc[-1]),
                "candle_open": opened.isoformat(), "candle_progress": round(min((now - opened) / step, 1.0), 3),
            }
    with config.atomic(config.LIVE_PREVIEW) as tmp:
        tmp.write_text(json.dumps(out, indent=2))
    return out


def load():
    """The latest saved preview, or None."""
    if not config.LIVE_PREVIEW.exists():
        return None
    try:
        return json.loads(config.LIVE_PREVIEW.read_text())
    except (OSError, ValueError):
        return None


def age_seconds(preview):
    return (datetime.now(timezone.utc) - datetime.fromisoformat(preview["as_of"])).total_seconds()
