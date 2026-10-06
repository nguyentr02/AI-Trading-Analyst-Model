"""Produce the current signal for each symbol and log changes (these are the "alerts").

The log's "timeframe" column holds the model name from config.MODELS ("4h_next", "4h" or "1d").
"""
import pandas as pd

from . import config, data, model


def label(prob):
    if prob >= config.ENTER_PROB:
        return "BULLISH"
    if prob <= config.EXIT_PROB:
        return "BEARISH"
    return "NEUTRAL"


def current(name, refresh=True, fast=False):
    """Latest signal per coin. `fast` skips refreshing funding/premium (unused by the models) to decide quickly."""
    if model.load(name) is None:
        raise RuntimeError(f"No {name} model yet. Run: python -m cryptoai train")
    tf = config.MODELS[name]["timeframe"]
    raw = {s: df.iloc[-400:] for s, df in data.closed(tf, refresh, refresh_derivatives=refresh and not fast).items()}
    probs = model.predict_history(raw, name, refresh)
    rows = []
    for sym, df in raw.items():
        prob = probs[sym].loc[df.index[-1]]
        rows.append(
            {
                "symbol": sym,
                "timeframe": name,
                "candle_close": df.index[-1] + pd.Timedelta(tf),
                "price": df["close"].iloc[-1],
                "prob_up": round(float(prob), 3),
                "signal": label(prob),
            }
        )
    return pd.DataFrame(rows)


def _last_logged():
    if not config.SIGNAL_LOG.exists():
        return {}
    log = pd.read_csv(config.SIGNAL_LOG)
    last = log.groupby(["symbol", "timeframe"]).tail(1)
    return {(r.symbol, r.timeframe): r.signal for r in last.itertuples()}


def check_and_log(names=tuple(config.MODELS)):
    """Compute signals, append them to the log, and return only the ones that changed."""
    prev = _last_logged()
    now = pd.concat([current(n) for n in names], ignore_index=True)
    now["changed"] = [prev.get((r.symbol, r.timeframe)) != r.signal for r in now.itertuples()]
    now["checked_at"] = pd.Timestamp.now(tz="UTC").floor("s")
    now.to_csv(config.SIGNAL_LOG, mode="a", header=not config.SIGNAL_LOG.exists(), index=False)
    return now, now[now["changed"]]
