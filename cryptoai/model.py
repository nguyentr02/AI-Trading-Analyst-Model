"""Train, evaluate (walk-forward) and run the prediction models.

One model per entry in config.MODELS (e.g. "4h" = 4h candles, next-1-day horizon), each trained on all
symbols pooled together for more data.
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from . import config, data, features


def spec(name):
    return config.MODELS[name]


def _new_model():
    # Deliberately conservative: shallow trees + strong regularisation. Market data is noisy
    # and a flexible model will memorise the past instead of learning anything useful.
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )


def build_features(name, raw, refresh=False, bars=None):
    """Features for every coin in `raw`, adding 15m/1h patterns if this model uses them.

    `bars` ({symbol: {"15m": candles, "1h": candles}}) can be passed in, e.g. with the latest minutes'
    bars for a live preview; otherwise they are read from the cache.
    """
    if not spec(name)["intraday"]:
        return features.build_all(raw, None)
    if bars is None:
        since = min(df.index[0] for df in raw.values()) - pd.Timedelta("2D")  # warm-up for 24h windows
        bars = data.intraday(refresh, since=since)
    return features.build_all(raw, bars)


def dataset(name, refresh=True):
    """Features + target for every symbol, stacked into one frame."""
    raw = data.closed(spec(name)["timeframe"], refresh)
    feats = build_features(name, raw, refresh)
    frames = []
    for sym, df in raw.items():
        if len(df) < 300:
            continue
        X = feats[sym]
        X["y"] = features.target(df, spec(name)["horizon"])
        X["fwd_ret_1"] = df["close"].pct_change().shift(-1)  # next-candle return, for backtests
        X["symbol"] = sym
        frames.append(X.iloc[200:])  # skip warm-up rows where long indicators are undefined
    return pd.concat(frames).sort_index()


def feature_cols(ds):
    return [c for c in ds.columns if c not in ("y", "fwd_ret_1", "symbol", *config.UNUSED_FEATURES)]


def walk_forward(ds, name, n_folds=8):
    """Out-of-sample probabilities: each fold is predicted by a model trained only on earlier data."""
    cols = feature_cols(ds)
    labeled = ds.dropna(subset=["y"])
    times = labeled.index.unique().sort_values()
    start = len(times) // 3
    edges = np.linspace(start, len(times), n_folds + 1).astype(int)
    gap = spec(name)["horizon"]  # embargo so training labels never overlap the test period

    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        test_t = times[a:b]
        train_end = times[max(a - gap, 1)]
        train = labeled[labeled.index < train_end]
        test = labeled[labeled.index.isin(test_t)]
        m = _new_model().fit(train[cols], train["y"])
        p = test[["symbol", "y", "fwd_ret_1"]].copy()
        p["prob"] = m.predict_proba(test[cols])[:, 1]
        out.append(p)
    return pd.concat(out)


def train(name, refresh=True, min_auc=None):
    """Retrain on all data up to the last closed candle and save the model.

    With `min_auc`, the new model is saved only if its walk-forward AUC reaches it; otherwise the
    current model is kept. Every run is appended to the training log either way.
    """
    ds = dataset(name, refresh)
    oos = walk_forward(ds, name)
    metrics = {
        "model": name,
        "timeframe": spec(name)["timeframe"],
        "horizon": spec(name)["horizon"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_candle": ds.index.max().isoformat(),
        "rows": int(ds["y"].notna().sum()),
        "oos_rows": len(oos),
        "oos_accuracy": round(accuracy_score(oos["y"], oos["prob"] > 0.5), 4),
        "oos_auc": round(roc_auc_score(oos["y"], oos["prob"]), 4),
        "baseline_up_rate": round(float(oos["y"].mean()), 4),
    }
    metrics["accepted"] = min_auc is None or metrics["oos_auc"] >= min_auc
    _log(metrics)
    if not metrics["accepted"]:
        return metrics

    labeled = ds.dropna(subset=["y"])
    cols = feature_cols(ds)
    final = _new_model().fit(labeled[cols], labeled["y"])

    with config.atomic(config.MODEL_DIR / f"model_{name}.joblib") as tmp:
        joblib.dump({"model": final, "features": cols}, tmp)
    with config.atomic(config.MODEL_DIR / f"oos_{name}.csv") as tmp:
        oos.to_csv(tmp)
    with config.atomic(config.MODEL_DIR / f"metrics_{name}.json") as tmp:
        tmp.write_text(json.dumps(metrics, indent=2))
    return metrics


def _log(metrics):
    row = pd.DataFrame([metrics])
    if config.TRAINING_LOG.exists():
        old = load_log(parse=False)
        if list(old.columns) != list(row.columns):  # a newer version added columns: rewrite with all of them
            with config.atomic(config.TRAINING_LOG) as tmp:
                pd.concat([old, row], ignore_index=True).to_csv(tmp, index=False)
            return
    row.to_csv(config.TRAINING_LOG, mode="a", header=not config.TRAINING_LOG.exists(), index=False)


def load_log(parse=True):
    if not config.TRAINING_LOG.exists():
        return pd.DataFrame()
    log = pd.read_csv(config.TRAINING_LOG)
    if "model" not in log:
        log["model"] = log["timeframe"]  # rows from before there were several models per timeframe
    log["model"] = log["model"].fillna(log["timeframe"])
    if parse:
        for c in ("trained_at", "last_candle"):
            log[c] = pd.to_datetime(log[c], format="ISO8601", utc=True)
    return log


def load(name):
    path = config.MODEL_DIR / f"model_{name}.joblib"
    return joblib.load(path) if path.exists() else None


def load_metrics(name):
    path = config.MODEL_DIR / f"metrics_{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_oos(name):
    path = config.MODEL_DIR / f"oos_{name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["time"], index_col="time")


def predict_history(raw, name, refresh=False, bars=None):
    """P(up) per closed candle for each coin in `raw` ({symbol: candles}, must include BTC/USDT).

    Returns a DataFrame with one column per symbol. Past values are in-sample, so use them only for display.
    """
    bundle = load(name)
    if bundle is None:
        return None
    probs = {}
    for sym, X in build_features(name, raw, refresh, bars).items():
        X = X.reindex(columns=bundle["features"])
        probs[sym] = pd.Series(bundle["model"].predict_proba(X)[:, 1], index=X.index)
    return pd.DataFrame(probs)
