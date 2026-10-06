"""Drop warning: the chance that a sharp fall comes before a rise, per coin, from each 4h close.

Label (triple barrier, Lopez de Prado): within the next 3 days (18 candles of 4h), does the price first fall
2 x the coin's recent daily volatility below the close, before rising the same amount? About 22% of 4h closes
are followed by a drop first. Same features and model settings as the "Next 1 day" model.

Tested in experiments/exit_timing.py (2026-10-06): walk-forward AUC 0.57 on 2025-2026 (the direction models
reach about 0.53). Selling everything when P(drop) >= 40% and not buying while it is >= 35% raised the Smart
strategy's Sharpe from 0.36 to 0.42 on 2025-2026 (within noise) and cut its worst drop from -40% to -32%
(-55% to -38% on 2022-2024). Individual warning sales were not better than chance at dodging the next 3 days'
move; the gain comes from avoiding the worst falls and not buying into them.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, data, model

HORIZON = 18  # 4h candles: 3 days
WIDTH = 2.0  # barrier distance in daily volatilities
SELL_AT = 0.40  # "High": the tested exit sells everything at or above this
CAUTION_AT = 0.35  # "Elevated": the tested strategy does not buy at or above this
MODEL_FILE = config.MODEL_DIR / "model_drop.joblib"
METRICS_FILE = config.MODEL_DIR / "metrics_drop.json"
CURRENT_FILE = config.LOG_DIR / "drop_risk.json"


def label(candles):
    """1 if the lower barrier is hit first within HORIZON candles, 0 otherwise, NaN where the future is unknown.

    A candle touching both barriers counts as a drop.
    """
    width = WIDTH * candles["close"].pct_change().rolling(120).std() * np.sqrt(6)  # daily vol from 4h returns
    close = candles["close"].to_numpy()
    lo, hi = close * (1 - width.to_numpy()), close * (1 + width.to_numpy())
    first_dn, first_up = np.full(len(candles), HORIZON + 1), np.full(len(candles), HORIZON + 1)
    for k in range(HORIZON, 0, -1):  # walk backwards so the earliest hit wins
        first_dn = np.where(candles["low"].shift(-k).to_numpy() <= lo, k, first_dn)
        first_up = np.where(candles["high"].shift(-k).to_numpy() >= hi, k, first_up)
    y = pd.Series(((first_dn <= first_up) & (first_dn <= HORIZON)).astype(float), index=candles.index)
    y[np.isnan(width.to_numpy())] = np.nan
    y.iloc[-HORIZON:] = np.nan
    return y


def dataset(refresh=True):
    """The "Next 1 day" model's rows (all coins), with the drop label as the target."""
    ds = model.dataset("4h", refresh)
    candles = data.closed("4h", refresh=False)
    for s in config.SYMBOLS:
        mask = (ds["symbol"] == s).to_numpy()
        ds.loc[mask, "y"] = label(candles[s]).reindex(ds.index[mask]).to_numpy()
    return ds


def train(refresh=True, min_auc=config.MIN_AUC):
    """Retrain on all data, keep the new model only if its walk-forward AUC reaches `min_auc`, save the readings."""
    ds = dataset(refresh)
    oos = model.walk_forward(ds, "4h", gap=HORIZON)
    metrics = {"model": "drop", "last_candle": ds.index.max().isoformat(), "rows": int(ds["y"].notna().sum()),
               "oos_auc": round(roc_auc_score(oos["y"], oos["prob"]), 4),
               "base_rate": round(float(ds["y"].mean()), 4)}
    metrics["accepted"] = metrics["oos_auc"] >= min_auc
    if metrics["accepted"]:
        labeled = ds.dropna(subset=["y"])
        cols = model.feature_cols(ds)
        with config.atomic(MODEL_FILE) as tmp:
            joblib.dump({"model": model._new_model().fit(labeled[cols], labeled["y"]), "features": cols}, tmp)
        with config.atomic(METRICS_FILE) as tmp:
            tmp.write_text(json.dumps(metrics, indent=2))
    save_current(ds)
    return metrics


def level(p):
    return "High" if p >= SELL_AT else "Elevated" if p >= CAUTION_AT else "Normal"


def save_current(ds=None):
    """P(drop) at the last closed 4h candle for every coin, written to logs/drop_risk.json for the dashboard."""
    bundle = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
    if bundle is None:
        return None
    ds = ds if ds is not None else model.dataset("4h", refresh=False)
    rows = {}
    for sym in config.SYMBOLS:
        last = ds[ds["symbol"] == sym].iloc[[-1]]
        p = float(bundle["model"].predict_proba(last.reindex(columns=bundle["features"]))[:, 1][0])
        rows[sym] = {"p_drop": p, "level": level(p), "candle": last.index[0].isoformat()}
    out = {"updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "coins": rows}
    with config.atomic(CURRENT_FILE) as tmp:
        tmp.write_text(json.dumps(out, indent=2))
    return out


def load():
    if not CURRENT_FILE.exists():
        return None
    try:
        return json.loads(CURRENT_FILE.read_text())
    except (OSError, ValueError):
        return None


def load_metrics():
    return json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else None
