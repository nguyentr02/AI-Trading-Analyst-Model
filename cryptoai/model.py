"""Train, evaluate (walk-forward) and run the prediction model.

One model per timeframe, trained on all symbols pooled together for more data.
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from . import config, data, features


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


def dataset(timeframe, refresh=True):
    """Features + target for every symbol, stacked into one frame."""
    raw = data.closed(timeframe, refresh)
    feats = features.build_all(raw)
    frames = []
    for sym, df in raw.items():
        if len(df) < 300:
            continue
        X = feats[sym]
        X["y"] = features.target(df, config.HORIZON[timeframe])
        X["fwd_ret_1"] = df["close"].pct_change().shift(-1)  # next-candle return, for backtests
        X["symbol"] = sym
        frames.append(X.iloc[200:])  # skip warm-up rows where long indicators are undefined
    return pd.concat(frames).sort_index()


def feature_cols(ds):
    return [c for c in ds.columns if c not in ("y", "fwd_ret_1", "symbol", *config.UNUSED_FEATURES)]


def walk_forward(ds, timeframe, n_folds=8):
    """Out-of-sample probabilities: each fold is predicted by a model trained only on earlier data."""
    cols = feature_cols(ds)
    labeled = ds.dropna(subset=["y"])
    times = labeled.index.unique().sort_values()
    start = len(times) // 3
    edges = np.linspace(start, len(times), n_folds + 1).astype(int)
    gap = config.HORIZON[timeframe]  # embargo so training labels never overlap the test period

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


def train(timeframe, refresh=True, min_auc=None):
    """Retrain on all data up to the last closed candle and save the model.

    With `min_auc`, the new model is saved only if its walk-forward AUC reaches it; otherwise the
    current model is kept. Every run is appended to the training log either way.
    """
    ds = dataset(timeframe, refresh)
    oos = walk_forward(ds, timeframe)
    metrics = {
        "timeframe": timeframe,
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

    with config.atomic(config.MODEL_DIR / f"model_{timeframe}.joblib") as tmp:
        joblib.dump({"model": final, "features": cols}, tmp)
    with config.atomic(config.MODEL_DIR / f"oos_{timeframe}.csv") as tmp:
        oos.to_csv(tmp)
    with config.atomic(config.MODEL_DIR / f"metrics_{timeframe}.json") as tmp:
        tmp.write_text(json.dumps(metrics, indent=2))
    return metrics


def _log(metrics):
    row = pd.DataFrame([metrics])
    row.to_csv(config.TRAINING_LOG, mode="a", header=not config.TRAINING_LOG.exists(), index=False)


def load_log():
    if not config.TRAINING_LOG.exists():
        return pd.DataFrame()
    return pd.read_csv(config.TRAINING_LOG, parse_dates=["trained_at", "last_candle"])


def load(timeframe):
    path = config.MODEL_DIR / f"model_{timeframe}.joblib"
    return joblib.load(path) if path.exists() else None


def load_metrics(timeframe):
    path = config.MODEL_DIR / f"metrics_{timeframe}.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_oos(timeframe):
    path = config.MODEL_DIR / f"oos_{timeframe}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["time"], index_col="time")


def predict_history(raw, timeframe):
    """P(up) per closed candle for each coin in `raw` ({symbol: candles}, must include BTC/USDT).

    Returns a DataFrame with one column per symbol. Past values are in-sample, so use them only for display.
    """
    bundle = load(timeframe)
    if bundle is None:
        return None
    probs = {}
    for sym, X in features.build_all(raw).items():
        X = X[bundle["features"]]
        probs[sym] = pd.Series(bundle["model"].predict_proba(X)[:, 1], index=X.index)
    return pd.DataFrame(probs)
