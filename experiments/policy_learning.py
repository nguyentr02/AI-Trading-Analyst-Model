"""Let the AI learn WHAT to buy, WHEN, and HOW MUCH (and the same for selling), to maximise profit after fees.

    .venv\\Scripts\\python experiments\\policy_learning.py

Fixed before any result was seen (2026-10-06). Instead of hand-written rules (buy at P(up) >= 55%, fixed tiers),
1. a return model predicts each coin's expected log return over the next day (6 candles of 4h) from the same
   features as the live models plus the daily 50-day trend; monthly retrain, out of sample from 2022;
2. a decision policy turns it into positions: size = clip(boldness x expected return / expected variance, 0, 1)
   (the Kelly principle: bet more when the expected gain is large relative to the risk); hold nothing unless the
   expected return beats a hurdle of fees; change the position only when the change is at least a band; decide
   every 4h or once a day; either every coin in its own sleeve, or the whole account in the top 1 or 2 coins by
   expected return ("what to buy").
3. the policy's settings are LEARNT by searching every combination on 2022-2024 for the best Sharpe after fees:
   boldness {0.5, 1, 2, 4} x hurdle {0, 1, 2} x round-trip cost x band {0.10, 0.25, 0.50} x cadence {4h, 1d} x
   allocation {per coin, top 1, top 2} = 216 policies.
Costs: 0.1% fee per side on every change (as the Smart backtests). Coins: BTC, ETH, BNB, SOL.
Check: the learnt policy is run once on 2025-01-01 to 2026-10-06. Adopted only if it beats the 50-day rule on
Sharpe in BOTH periods and AI Smart with the drop exit on the check period. With 216 policies searched, the
deflated Sharpe ratio (which corrects for picking the best of many) is reported too.
"""
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from cryptoai import config, data, metrics, model  # noqa: E402
import exit_timing  # noqa: E402

START, END = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2026-10-06", tz="UTC")
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")
STEP, H, FEE = pd.Timedelta("4h"), 6, config.FEE
PRED = ROOT / "reports" / "predictions_return_model.pkl"
GRID = {"boldness": (0.5, 1.0, 2.0, 4.0), "hurdle": (0, 1, 2), "band": (0.10, 0.25, 0.50), "cadence": ("4h", "1d"),
        "alloc": ("per_coin", "top1", "top2")}


def return_model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3, min_samples_leaf=100,
                                         l2_regularization=1.0, early_stopping=False, random_state=42)


def predictions():
    """Out-of-sample expected 1-day log return per coin and 4h candle, monthly retrain (cached)."""
    if PRED.exists():
        return pickle.load(open(PRED, "rb"))
    ds = model.dataset("4h", refresh=False)
    candles = data.closed("4h", refresh=False)
    ds["trend50"] = np.nan
    ds["target"] = np.nan
    for s in config.SYMBOLS:
        mask = (ds["symbol"] == s).to_numpy()
        c = candles[s]["close"]
        d = data.drop_open_candle(data.load_cached(s, "1d"), "1d")["close"]
        trend = (d / d.rolling(50).mean() - 1)
        trend.index = trend.index + pd.Timedelta("1D")  # known at the daily close
        idx = ds.index[mask] + STEP  # each 4h candle's close
        pos = trend.index.searchsorted(idx, side="right") - 1
        ds.loc[mask, "trend50"] = np.where(pos >= 0, trend.to_numpy()[np.clip(pos, 0, None)], np.nan)
        ds.loc[mask, "target"] = np.log(c.shift(-H) / c).reindex(ds.index[mask]).to_numpy()
    cols = [c for c in model.feature_cols(ds) if c != "target"]  # includes trend50, added above
    parts = []
    for m0, m1 in zip(pd.date_range(START, END, freq="MS", tz="UTC"),
                      list(pd.date_range(START, END, freq="MS", tz="UTC")[1:]) + [END]):
        train = ds[ds.index <= m0 - (H + 1) * STEP].dropna(subset=["target"])
        test = ds[(ds.index >= m0) & (ds.index < m1)]
        if test.empty:
            continue
        fitted = return_model().fit(train[cols], train["target"])
        parts.append(pd.DataFrame({"symbol": test["symbol"], "mu": fitted.predict(test[cols])}, index=test.index))
    out = pd.concat(parts)
    pickle.dump(out, open(PRED, "wb"))
    return out


def frames(pred):
    """Per coin: next-4h return, expected 1-day return and variance, and the 50-day trend flag, on the 4h grid."""
    out = {}
    for s in config.SYMBOLS:
        c4 = data.closed("4h", refresh=False, symbols=[s])[s]["close"]
        r = np.log(c4).diff()
        f = pd.DataFrame({"r_next": c4.shift(-1) / c4 - 1, "var": r.rolling(50).var() * H}, index=c4.index)
        d = data.drop_open_candle(data.load_cached(s, "1d"), "1d")["close"]
        flag = (d > d.rolling(50).mean()).astype(float)
        flag.index = flag.index + pd.Timedelta("1D")
        pos = flag.index.searchsorted(f.index + STEP, side="right") - 1
        f["above50"] = np.where(pos >= 0, flag.to_numpy()[np.clip(pos, 0, None)], np.nan)
        f["mu"] = pred[pred["symbol"] == s]["mu"].reindex(f.index)
        out[s] = f[(f.index >= START) & (f.index < END)].dropna()
    common = sorted(set.intersection(*(set(f.index) for f in out.values())))
    return {s: f.loc[common] for s, f in out.items()}


def positions(fr, p):
    """Position per coin (share of the WHOLE account, summing to at most 1) for policy p."""
    syms = list(fr)
    mu = pd.DataFrame({s: fr[s]["mu"] for s in syms})
    var = pd.DataFrame({s: fr[s]["var"] for s in syms})
    hurdle = p["hurdle"] * 2 * FEE
    raw = (p["boldness"] * mu / var).clip(0, 1).where(mu > hurdle, 0.0)
    if p["alloc"] == "per_coin":
        target = raw / len(syms)
    else:
        k = 1 if p["alloc"] == "top1" else 2
        rank = mu.where(mu > hurdle).rank(axis=1, ascending=False)
        target = raw.where(rank <= k, 0.0) / k
    decide = np.ones(len(target), bool) if p["cadence"] == "4h" else (target.index.hour == 20)  # 00:00 UTC close
    band = p["band"] / (len(syms) if p["alloc"] == "per_coin" else (1 if p["alloc"] == "top1" else 2))
    tv, cur, out = target.to_numpy(), np.zeros(len(syms)), np.empty_like(target.to_numpy())
    for i in range(len(tv)):
        if decide[i]:
            for j in range(len(syms)):
                t = tv[i, j]
                if abs(t - cur[j]) >= band or (t == 0 and cur[j] > 0):
                    cur[j] = t
        out[i] = cur
    return pd.DataFrame(out, index=target.index, columns=syms)


def account_returns(fr, pos):
    """Daily returns of the whole account (cash earns nothing), after fees."""
    r_next = pd.DataFrame({s: fr[s]["r_next"] for s in fr})
    change = pos.diff().abs()
    change.iloc[0] = pos.iloc[0]
    r4 = (pos * r_next).sum(axis=1) - change.sum(axis=1) * FEE
    return (1 + r4).groupby(r4.index.floor("1D")).prod() - 1, change.sum(axis=1).sum()


def score(r):
    eq = (1 + r).cumprod()
    return {"return": eq.iloc[-1] - 1, "Sharpe": metrics.sharpe(r, 365), "worst drop": metrics.max_drawdown(eq)}


def main():
    pred = predictions()
    fr = frames(pred)
    choose = lambda r: r[r.index < SPLIT]
    check = lambda r: r[r.index >= SPLIT]
    results = []
    for combo in itertools.product(*GRID.values()):
        p = dict(zip(GRID, combo))
        r, turnover = account_returns(fr, positions(fr, p))
        years = (r.index[-1] - r.index[0]).days / 365
        results.append({**p, "r": r, "Sharpe choose": metrics.sharpe(choose(r), 365), "turnover/yr": turnover / years})
    res = pd.DataFrame(results).sort_values("Sharpe choose", ascending=False)
    print("Top 5 policies on 2022-2024 (the search):")
    print(res.drop(columns="r").head(5).to_string(index=False, float_format="{:.2f}".format))
    best = res.iloc[0]
    learnt = {k: best[k] for k in GRID}

    sma = pd.DataFrame({s: fr[s]["above50"] / len(fr) for s in fr})
    r_sma, _ = account_returns(fr, sma)
    sp = pickle.load(open(exit_timing.SMART_PRED, "rb"))
    dp = pickle.load(open(exit_timing.DROP_PRED, "rb"))
    probs = {**{n: sp[n] for n in config.MODELS}, "drop": dp[["symbol", "prob"]]}
    rows = []
    for phase, (a, b) in (("choose 2022-2024", (START, SPLIT)), ("check 2025-2026", (SPLIT, END))):
        smart = {s: exit_timing.run(s, a, b, probs, exit_timing.VARIANTS["drop_all"])[0] for s in config.SYMBOLS}
        series = {"learnt policy": best["r"], "50-day rule": r_sma,
                  "AI Smart + drop exit": pd.concat(smart, axis=1).fillna(0).mean(axis=1)}
        for name, r in series.items():
            w = r[(r.index >= a) & (r.index < b)]
            rows.append({"phase": phase, "strategy": name, **score(w)})
    out = pd.DataFrame(rows)
    fmt = {"return": "{:+.1%}".format, "Sharpe": "{:+.2f}".format, "worst drop": "{:+.1%}".format}
    print(f"\nLearnt on 2022-2024: {learnt}, turnover {best['turnover/yr']:.0f}x a year")
    for phase in ("choose 2022-2024", "check 2025-2026"):
        print(f"\n=== {phase} ===")
        print(out[out.phase == phase].drop(columns="phase").set_index("strategy").to_string(formatters=fmt))
    s = out.set_index(["phase", "strategy"])["Sharpe"]
    ok = (s[("choose 2022-2024", "learnt policy")] > s[("choose 2022-2024", "50-day rule")]
          and s[("check 2025-2026", "learnt policy")] > s[("check 2025-2026", "50-day rule")]
          and s[("check 2025-2026", "learnt policy")] > s[("check 2025-2026", "AI Smart + drop exit")])
    chk = check(best["r"])
    dsr = metrics.deflated_sharpe(chk, 365, len(res))
    print(f"\nDeflated Sharpe on the check period (216 policies searched): {dsr:.0%} chance the edge is real")
    print("-> " + ("ADOPT" if ok else "not adopted"))
    out.to_csv(ROOT / "reports" / "policy_learning.csv", index=False)
    res.drop(columns="r").to_csv(ROOT / "reports" / "policy_learning_search.csv", index=False)


if __name__ == "__main__":
    main()
