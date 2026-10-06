"""Diagnostic: does the AI know anything the trend doesn't?

    .venv\\Scripts\\python experiments\\ai_vs_trend.py

Fixed before any result was seen (2026-10-06). The information coefficient (IC = Spearman
correlation between the prediction and the realised forward return) over the whole period, with a t-statistic
from a block bootstrap over months (overlapping 1-day returns make plain standard errors too small). A first
version computed one IC per month (Numerai-style); that is biased for slow signals (about -0.25 on a random
walk), so it was replaced before any conclusion was drawn. Predictions: the walk-forward "Next 1 day" P(up)
(monthly retrain, out of sample, BTC/ETH/BNB/SOL, 4h candles, 2022-01 to 2026-10); target: the return over
the next 6 candles (1 day).
- ai              the AI's P(up)
- trend           a plain trend score: the coin's distance above its 300-candle (= 50-day) average
- ai_neutral      the AI's P(up) after removing everything a linear fit on trend features
                  explains (distance from the 20/50/200-candle EMAs, 50/100-candle returns, EMA50 slope,
                  distance from the 100-candle high, BTC's distance from its 50/200 EMAs, and the 50-day score)
Verdict: if ai_neutral's IC t-statistic is below 2, the AI adds no direction signal beyond trend, and should be
used for risk (drop warning) rather than direction. Not a strategy; it does not count as a trial.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config, data, model  # noqa: E402

PRED = ROOT / "reports" / "predictions_smart_2022_2026.pkl"
TREND = ["dist_ema20", "dist_ema50", "dist_ema200", "ret_50", "ret_100", "ema50_slope", "from_high_100",
         "btc_dist_ema50", "btc_dist_ema200", "trend"]


def main():
    pred = pickle.load(open(PRED, "rb"))["4h"]
    ds = model.dataset("4h", refresh=False)
    candles = data.closed("4h", refresh=False)
    parts = []
    for sym in config.SYMBOLS:
        c = candles[sym]["close"]
        f = ds[ds["symbol"] == sym][TREND[:-1]].copy()
        f["trend"] = (c / c.rolling(300).mean() - 1).reindex(f.index)
        f["fwd"] = (c.shift(-6) / c - 1).reindex(f.index)
        f["ai"] = pred[pred["symbol"] == sym]["prob"].reindex(f.index)
        f["symbol"] = sym
        parts.append(f)
    df = pd.concat(parts).dropna()
    df["month"] = df.index.strftime("%Y-%m")

    # Neutralise: remove the part of the AI's P(up) that a linear fit on trend features explains (one fit on all
    # rows; a diagnostic, not a strategy).
    X = np.c_[np.ones(len(df)), df[TREND].to_numpy()]
    beta, *_ = np.linalg.lstsq(X, df["ai"].to_numpy(), rcond=None)
    df["ai_neutral"] = df["ai"].to_numpy() - X @ beta
    r2 = 1 - df["ai_neutral"].var() / df["ai"].var()

    # One IC per signal over the whole period; its uncertainty from resampling whole months (block bootstrap).
    # (An IC computed inside each month is biased for slow signals: on a random walk it averages about -0.25.)
    rng = np.random.default_rng(0)
    months = df["month"].unique()
    groups = {m: g for m, g in df.groupby("month")}
    print(f"{len(months)} months, {len(df):,} predictions\n")
    print(f"{'signal':<12} {'IC':>8} {'t-stat':>7}   2022-24 / 2025-26 IC")
    out = {}
    for col in ("ai", "trend", "ai_neutral"):
        ic = spearmanr(df[col], df["fwd"])[0]
        boot = []
        for _ in range(300):
            sample = pd.concat([groups[m] for m in rng.choice(months, len(months))])
            boot.append(spearmanr(sample[col], sample["fwd"])[0])
        t = ic / np.std(boot)
        early, late = df[df["month"] < "2025-01"], df[df["month"] >= "2025-01"]
        out[col] = t
        print(f"{col:<12} {ic:>+8.4f} {t:>7.2f}   {spearmanr(early[col], early['fwd'])[0]:+.4f} / "
              f"{spearmanr(late[col], late['fwd'])[0]:+.4f}")
    print(f"\nShare of the AI's variation explained by trend features: {r2:.0%}")
    print("-> " + ("The AI has direction signal beyond trend (t >= 2)" if out["ai_neutral"] >= 2 else
                   "No significant direction signal beyond trend (t < 2): use the AI for risk, not direction"))


if __name__ == "__main__":
    main()
