"""How do other AI trading systems train, and do their methods help our models?

    .venv\\Scripts\\python experiments\\training_practices.py

Fixed before any result was seen (2026-10-06). Variants of how the 4h models are trained, all using the
same features, folds and embargo as cryptoai.model.walk_forward:
- baseline        expanding window, all history weighted equally (what we do now)
- window_1y/2y/3y FreqAI-style sliding window: train only on the last 1, 2 or 3 years
- recency_1y      Qlib-style sample reweighting: expanding window, weights halve every year back in time
- ensemble        Qlib DoubleEnsemble / Numerai-style: average of our HistGradientBoosting and a LightGBM

Choice: the best mean walk-forward AUC over the two 4h models ("next 4 hours" and "next 1 day") on test
rows in 2022-2024. Check: 2025-2026, once. Adoption needs a gain over the baseline in BOTH phases larger than
0.003, the size of gain a meaningless moon-phase feature produced (docs/research/technical-analysis-concepts.md).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config, model  # noqa: E402

NOISE = 0.003
VARIANTS = {
    "baseline": {},
    "window_1y": {"window": pd.Timedelta("365D")},
    "window_2y": {"window": pd.Timedelta("730D")},
    "window_3y": {"window": pd.Timedelta("1095D")},
    "recency_1y": {"halflife": pd.Timedelta("365D")},
    "ensemble": {"ensemble": True},
}


def lgbm():
    return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1)


def walk_forward(ds, name, window=None, halflife=None, ensemble=False, n_folds=8):
    """cryptoai.model.walk_forward with optional sliding window, recency weights or a two-model ensemble."""
    cols = model.feature_cols(ds)
    labeled = ds.dropna(subset=["y"])
    times = labeled.index.unique().sort_values()
    edges = np.linspace(len(times) // 3, len(times), n_folds + 1).astype(int)
    gap = config.MODELS[name]["horizon"]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        train_end = times[max(a - gap, 1)]
        train = labeled[labeled.index < train_end]
        if window is not None:
            train = train[train.index >= train_end - window]
        test = labeled[labeled.index.isin(times[a:b])]
        weights = None
        if halflife is not None:
            age = (train_end - train.index).total_seconds() / halflife.total_seconds()
            weights = np.power(0.5, age)
        prob = model._new_model().fit(train[cols], train["y"], sample_weight=weights).predict_proba(test[cols])[:, 1]
        if ensemble:
            prob = (prob + lgbm().fit(train[cols], train["y"]).predict_proba(test[cols])[:, 1]) / 2
        out.append(test[["y"]].assign(prob=prob))
    return pd.concat(out)


def phase_auc(oos, start, end):
    o = oos[(oos.index >= start) & (oos.index < end)]
    return roc_auc_score(o["y"], o["prob"])


def main():
    data = {n: model.dataset(n, refresh=False) for n in ("4h_next", "4h")}
    rows = []
    for v, kw in VARIANTS.items():
        for n, ds in data.items():
            oos = walk_forward(ds, n, **kw)
            rows.append({"variant": v, "model": config.MODELS[n]["label"],
                         "choose 2022-2024": phase_auc(oos, "2022-01-01", "2025-01-01"),
                         "check 2025-2026": phase_auc(oos, "2025-01-01", "2027-01-01")})
            print(f"{v:<11} {config.MODELS[n]['label']:<13} {rows[-1]['choose 2022-2024']:.4f}  {rows[-1]['check 2025-2026']:.4f}",
                  flush=True)
    r = pd.DataFrame(rows)
    mean = r.groupby("variant")[["choose 2022-2024", "check 2025-2026"]].mean()
    gain = mean - mean.loc["baseline"]
    print("\nMean AUC over the two 4h models, and gain over the baseline:")
    print(pd.concat({"AUC": mean, "gain": gain}, axis=1).to_string(float_format="{:+.4f}".format))
    chosen = mean["choose 2022-2024"].idxmax()
    print(f"\nChosen on 2022-2024: {chosen}")
    g = gain.loc[chosen]
    if chosen == "baseline":
        print("Verdict: nothing beats the current training method.")
    elif g["choose 2022-2024"] > NOISE and g["check 2025-2026"] > NOISE:
        print(f"Verdict: ADOPT. Gain {g['choose 2022-2024']:+.4f} when choosing and {g['check 2025-2026']:+.4f} on the check.")
    else:
        print(f"Verdict: not adopted. Gains ({g['choose 2022-2024']:+.4f}, {g['check 2025-2026']:+.4f}) are not both above "
              f"the {NOISE} noise level.")


if __name__ == "__main__":
    main()
