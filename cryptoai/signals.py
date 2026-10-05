"""Produce the current signal for each symbol and log changes (these are the "alerts")."""
import pandas as pd

from . import config, data, model


def label(prob):
    if prob >= config.ENTER_PROB:
        return "BULLISH"
    if prob <= config.EXIT_PROB:
        return "BEARISH"
    return "NEUTRAL"


def current(timeframe, refresh=True):
    bundle = model.load(timeframe)
    if bundle is None:
        raise RuntimeError(f"No {timeframe} model yet. Run: python -m cryptoai train")
    rows = []
    for sym in config.SYMBOLS:
        df = data.update(sym, timeframe) if refresh else data.load_cached(sym, timeframe)
        df = data.drop_open_candle(df, timeframe)
        prob = model.predict_history(df.iloc[-400:], timeframe).iloc[-1]
        rows.append(
            {
                "symbol": sym,
                "timeframe": timeframe,
                "candle_close": df.index[-1] + pd.Timedelta(timeframe),
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


def check_and_log(timeframes=config.TIMEFRAMES):
    """Compute signals, append them to the log, and return only the ones that changed."""
    prev = _last_logged()
    now = pd.concat([current(tf) for tf in timeframes], ignore_index=True)
    now["changed"] = [prev.get((r.symbol, r.timeframe)) != r.signal for r in now.itertuples()]
    now["checked_at"] = pd.Timestamp.now(tz="UTC").floor("s")
    now.to_csv(config.SIGNAL_LOG, mode="a", header=not config.SIGNAL_LOG.exists(), index=False)
    return now, now[now["changed"]]
