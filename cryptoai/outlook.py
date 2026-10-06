"""Potential growth or decline: how much the price really moved after the AI gave a similar reading.

For each crypto model, the out-of-sample P(up) readings of 2022-2026 (monthly retrain, simulate.predictions) are
grouped into bands, and for each band: how often the price rose over the model's window, the average and median
move, and the typical range (25th to 75th percentile). A live reading is then described by its band, e.g. "when the
AI said 55-58%, the price moved +0.4% on average over the next day, typically -2.1% to +2.6%".

Check (fixed 2026-10-06, before the numbers were seen): the bands are only called reliable if readings of 55%+ were
followed by a higher average move than readings of 48% or less, in BOTH 2022-2024 and 2025-2026.

    python -m cryptoai outlook    # recompute (uses the cached predictions in reports/ if present)
"""
import json
import pickle

import numpy as np
import pandas as pd

from . import config, data, simulate

BANDS = [0.0, 0.40, 0.45, 0.48, 0.52, 0.55, 0.58, 0.62, 1.0]
OUT = config.MODEL_DIR / "outlook.json"
CACHE = config.ROOT / "reports" / "predictions_smart_2022_2026.pkl"
START, SPLIT = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")


def _predictions():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    end = pd.Timestamp.now(tz="UTC").normalize()
    return {n: simulate.predictions(n, START, end) for n in config.MODELS}


def _band(p):
    i = int(np.searchsorted(BANDS, p, side="right")) - 1
    return min(max(i, 0), len(BANDS) - 2)


def compute():
    preds = _predictions()
    out = {"computed": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "models": {}}
    for name, spec in config.MODELS.items():
        if name not in preds:
            continue
        tf, h = spec["timeframe"], spec["horizon"]
        candles = data.closed(tf, refresh=False)
        parts = []
        for sym, g in preds[name].groupby("symbol"):
            c = candles[sym]["close"]
            fwd = (c.shift(-h) / c - 1).reindex(g.index)
            parts.append(pd.DataFrame({"p": g["prob"], "fwd": fwd}, index=g.index))
        df = pd.concat(parts).dropna()
        df["band"] = df["p"].map(_band)
        bands = []
        for b, grp in df.groupby("band"):
            r = grp["fwd"]
            bands.append({"from": BANDS[b], "to": BANDS[b + 1], "n": int(len(r)), "up_share": float((r > 0).mean()),
                          "mean": float(r.mean()), "median": float(r.median()),
                          "p25": float(r.quantile(0.25)), "p75": float(r.quantile(0.75))})
        check = {}
        for label, part in (("2022-2024", df[df.index < SPLIT]), ("2025-2026", df[df.index >= SPLIT])):
            hi, lo = part[part["p"] >= 0.55]["fwd"], part[part["p"] <= 0.48]["fwd"]
            check[label] = {"high_mean": float(hi.mean()), "low_mean": float(lo.mean()),
                            "high_n": int(len(hi)), "low_n": int(len(lo))}
        reliable = all(v["high_mean"] > v["low_mean"] for v in check.values())
        out["models"][name] = {"label": spec["label"], "bands": bands, "check": check, "reliable": reliable,
                               "base_mean": float(df["fwd"].mean()), "base_up": float((df["fwd"] > 0).mean())}
    with config.atomic(OUT) as tmp:
        tmp.write_text(json.dumps(out, indent=2))
    return out


def load():
    try:
        return json.loads(OUT.read_text())
    except (OSError, ValueError):
        return None


def describe(name, p, outlook=None):
    """The band a reading falls in, as a dict, or None."""
    o = (outlook or load() or {}).get("models", {}).get(name)
    if not o:
        return None
    b = next((x for x in o["bands"] if x["from"] <= p < x["to"] or (p >= 1 and x["to"] == 1.0)), None)
    return {**b, "reliable": o["reliable"]} if b else None
