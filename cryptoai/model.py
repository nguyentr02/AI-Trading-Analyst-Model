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
    frames = []
    for sym in config.SYMBOLS:
        df = data.update(sym, timeframe) if refresh else data.load_cached(sym, timeframe)
        df = data.drop_open_candle(df, timeframe)
        if len(df) < 300:
            continue
        X = features.build(df)
        X["y"] = features.target(df, config.HORIZON[timeframe])
        X["fwd_ret_1"] = df["close"].pct_change().shift(-1)  # next-candle return, for backtests
        X["symbol"] = sym
        frames.append(X.iloc[200:])  # skip warm-up rows where long indicators are undefined
    return pd.concat(frames).sort_index()


def feature_cols(ds):
    return [c for c in ds.columns if c not in ("y", "fwd_ret_1", "symbol")]


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


def train(timeframe, refresh=True):
    ds = dataset(timeframe, refresh)
    oos = walk_forward(ds, timeframe)
    metrics = {
        "timeframe": timeframe,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(ds["y"].notna().sum()),
        "oos_rows": len(oos),
        "oos_accuracy": round(accuracy_score(oos["y"], oos["prob"] > 0.5), 4),
        "oos_auc": round(roc_auc_score(oos["y"], oos["prob"]), 4),
        "baseline_up_rate": round(float(oos["y"].mean()), 4),
    }

    labeled = ds.dropna(subset=["y"])
    cols = feature_cols(ds)
    final = _new_model().fit(labeled[cols], labeled["y"])

    joblib.dump({"model": final, "features": cols}, config.MODEL_DIR / f"model_{timeframe}.joblib")
    oos.to_csv(config.MODEL_DIR / f"oos_{timeframe}.csv")
    (config.MODEL_DIR / f"metrics_{timeframe}.json").write_text(json.dumps(metrics, indent=2))
    return metrics


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


def predict_history(df, timeframe):
    """P(up) for every closed candle in df (in-sample for the past, so use only for display)."""
    bundle = load(timeframe)
    if bundle is None:
        return None
    X = features.build(df)[bundle["features"]]
    return pd.Series(bundle["model"].predict_proba(X)[:, 1], index=X.index)
