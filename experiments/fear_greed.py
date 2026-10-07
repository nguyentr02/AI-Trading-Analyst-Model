"""Does the Crypto Fear & Greed Index improve the crypto AI's predictions?

    .venv\\Scripts\\python experiments\\fear_greed.py

Fixed before any result was seen (2026-10-07). The index (alternative.me, daily since 2018-02) summarises market mood
from volatility, momentum, social media, dominance and search trends. A day's value is treated as known only from
the next day's 00:00 UTC (conservative, no look-ahead). Features added to the live models' inputs:
- fg          the index value (0 = extreme fear, 100 = extreme greed)
- fg_chg7     its change over 7 days
- fg_z90      the value vs its last 90 days (z-score)
Models: "4h" (next 1 day, 4h candles) and "1d" (next 3 days, daily candles), same settings, retrained monthly from
2022 on all earlier rows (out of sample). Adopted only if adding the index raises the AUC by more than 0.003 (the
placebo noise level) in BOTH 2022-2024 and 2025-2026, for that model.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from cryptoai import config, model, news  # noqa: E402
from drop_features import as_of  # noqa: E402

START, SPLIT, END = (pd.Timestamp(x, tz="UTC") for x in ("2022-01-01", "2025-01-01", "2026-10-07"))
NOISE = 0.003
FG = ["fg", "fg_chg7", "fg_z90"]


def fg_series():
    d = news.fear_greed(0)["value"].astype(float)
    f = pd.DataFrame({"fg": d, "fg_chg7": d - d.shift(7), "fg_z90": (d - d.rolling(90).mean()) / d.rolling(90).std()})
    f.index = f.index + pd.Timedelta("1D")  # known from the next day
    return f


def run(name):
    ds = model.dataset(name, refresh=False)
    base = model.feature_cols(ds)
    step = pd.Timedelta(config.MODELS[name]["timeframe"])
    f = fg_series()
    known = ds.index + step  # each candle's close
    for c in FG:
        ds[c] = as_of(f[c], known).to_numpy()
    h = config.MODELS[name]["horizon"]
    months = pd.date_range(START, END, freq="MS")
    preds = {"current": [], "+ fear & greed": []}
    for m0, m1 in zip(months, list(months[1:]) + [END]):
        train = ds[ds.index <= m0 - (h + 1) * step].dropna(subset=["y"])
        test = ds[(ds.index >= m0) & (ds.index < m1)].dropna(subset=["y"])
        if test.empty:
            continue
        for label, cols in (("current", base), ("+ fear & greed", base + FG)):
            mdl = model._new_model().fit(train[cols], train["y"])
            preds[label].append(pd.DataFrame({"y": test["y"], "p": mdl.predict_proba(test[cols])[:, 1]}, index=test.index))
    out = {}
    for label, parts in preds.items():
        p = pd.concat(parts)
        out[label] = {"2022-2024": roc_auc_score(p[p.index < SPLIT]["y"], p[p.index < SPLIT]["p"]),
                      "2025-2026": roc_auc_score(p[p.index >= SPLIT]["y"], p[p.index >= SPLIT]["p"])}
    return out


def main():
    for name in ("4h", "1d"):
        r = run(name)
        gains = {k: r["+ fear & greed"][k] - r["current"][k] for k in r["current"]}
        ok = all(g > NOISE for g in gains.values())
        print(f"\n=== {config.MODELS[name]['label']} ({name}) ===")
        for label, v in r.items():
            print(f"{label:<16} AUC 2022-2024 {v['2022-2024']:.4f}   2025-2026 {v['2025-2026']:.4f}")
        print(f"gain             {gains['2022-2024']:+.4f}              {gains['2025-2026']:+.4f}   -> "
              f"{'ADOPT' if ok else 'not adopted'}")


if __name__ == "__main__":
    main()
