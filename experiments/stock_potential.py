"""Can an AI rank which stocks have the most potential over the next 3 months?

    .venv\\Scripts\\python experiments\\stock_potential.py

Fixed before any result was seen (2026-10-06). Universe: the 15 Binance-listed US stocks (ETFs excluded). At each
month end, for every stock: the features of the stock AI (cryptoai/stockai.py: daily technical features, SPY context,
strength vs SPY, rank among the stocks) and the target = its return over the next 63 trading days (~3 months) minus
SPY's ("beats the market by"). A gradient-boosting regressor (same conservative settings as the other models) is
retrained at the start of each year on all month-ends whose 3-month outcome was known by then; predictions from
2019 (training from 2016).
Checks, on 2019-2022 and 2023-2026:
1. Ranking skill: the Spearman rank correlation (IC) between predicted and actual 3-month excess return across the
   stocks, each month; mean IC and its t-statistic (overlapping 3-month windows: standard error x sqrt(3)).
2. Use: the buy-for-hold list (stocks above their 200-day average, equal weight, monthly; adopted in
   experiments/stock_hold_picks.py) vs the same list tilted to potential: only the half of the list with the
   highest predicted potential, equal weight. 0.1% cost per side.
A "potential" ranking is shown in the app only if the IC is positive with t >= 2 in BOTH periods AND the tilted list
beats the plain list on Sharpe in both periods.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import features, stocks  # noqa: E402

H = 63
PICKS = [t for t in stocks.STOCKS if t not in stocks.ETFS]
PERIODS = {"2019-2022": ("2019-01-01", "2023-01-01"), "2023-2026": ("2023-01-01", "2100-01-01")}
COST = stocks.COST


def monthly_dataset():
    raw = {t: stocks.candles(t) for t in [*PICKS, "SPY"]}
    spy = raw["SPY"]["close"]
    spy_f = features.build(raw["SPY"])[features.MARKET_COLS].add_prefix("spy_")
    closes = pd.DataFrame({t: raw[t]["close"] for t in PICKS})
    rank = closes.pct_change(20).rank(axis=1, pct=True)
    rows = []
    for t in PICKS:
        f = features.build(raw[t]).join(spy_f)
        c = raw[t]["close"]
        f["rel_spy_20"] = c.pct_change(20) - spy.pct_change(20)
        f["rel_spy_120"] = c.pct_change(120) - spy.pct_change(120)
        f["ret_250"] = c.pct_change(250)
        f["from_200d"] = c / c.rolling(200).mean() - 1
        f["xs_rank_20"] = rank[t]
        f["target"] = (c.shift(-H) / c - 1) - (spy.shift(-H) / spy - 1)
        f["above_200d"] = c > c.rolling(200).mean()
        f["ticker"] = t
        month_end = f.groupby(f.index.to_period("M")).tail(1)
        rows.append(month_end.iloc[10:])  # skip warm-up months
    ds = pd.concat(rows).sort_index()
    return ds.drop(columns=["hour"], errors="ignore"), closes


def predictions(ds):
    cols = [c for c in ds.columns if c not in ("target", "above_200d", "ticker")]
    out = []
    for year in range(2019, pd.Timestamp.now().year + 1):
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        known = ds.index <= start - pd.Timedelta(days=int(H * 1.5))  # outcome known before the year starts
        train = ds[known].dropna(subset=["target"])
        test = ds[(ds.index >= start) & (ds.index < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))]
        if len(train) < 300 or test.empty:
            continue
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3, min_samples_leaf=50,
                                          l2_regularization=1.0, random_state=42).fit(train[cols], train["target"])
        out.append(test.assign(pred=m.predict(test[cols])))
    return pd.concat(out)


def ic_table(p):
    ics = p.dropna(subset=["target"]).groupby(p.dropna(subset=["target"]).index.to_period("M")).apply(
        lambda g: spearmanr(g["pred"], g["target"])[0] if len(g) >= 8 else np.nan).dropna()
    rows = {}
    for label, (a, b) in PERIODS.items():
        s = ics[(ics.index >= pd.Period(a[:7], "M")) & (ics.index < pd.Period(b[:7], "M"))]
        se = s.std() / np.sqrt(len(s)) * np.sqrt(3)
        rows[label] = {"mean IC": s.mean(), "t": s.mean() / se if se else np.nan, "months": len(s),
                       "share positive": (s > 0).mean()}
    return pd.DataFrame(rows).T


def strategy_returns(p, closes, tilt):
    rets = closes.pct_change().fillna(0)
    w = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    months = sorted(set(p.index))
    for i, m in enumerate(months):
        g = p[p.index == m]
        g = g[g["above_200d"]]
        if g.empty:
            continue
        if tilt:
            g = g[g["pred"] >= g["pred"].median()]
        nxt = months[i + 1] if i + 1 < len(months) else closes.index[-1] + pd.Timedelta("1D")
        rows = (w.index > m) & (w.index <= nxt)
        w.loc[rows, :] = 0.0
        w.loc[rows, list(g["ticker"])] = 1.0 / len(g)
    held = w.shift(1).fillna(0)
    return (held * rets).sum(axis=1) - w.diff().abs().sum(axis=1).fillna(0) * COST


def stats(r):
    eq = (1 + r).cumprod()
    return {"return": eq.iloc[-1] - 1, "Sharpe": r.mean() / r.std() * np.sqrt(252), "worst fall": (eq / eq.cummax() - 1).min()}


def main():
    ds, closes = monthly_dataset()
    p = predictions(ds)
    ic = ic_table(p)
    print("=== Ranking skill (3-month excess return vs SPY) ===")
    print(ic.to_string(float_format=lambda x: f"{x:+.3f}"))
    first = p.index.min()
    rows = []
    for name, tilt in (("buy-for-hold list", False), ("list tilted to potential", True)):
        r = strategy_returns(p, closes, tilt)
        r = r[r.index > first]
        for label, (a, b) in PERIODS.items():
            w = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]
            rows.append({"strategy": name, "period": label, **stats(w)})
    res = pd.DataFrame(rows)
    fmt = {"return": "{:+.0%}".format, "Sharpe": "{:.2f}".format, "worst fall": "{:+.0%}".format}
    for label in PERIODS:
        print(f"\n=== {label} ===")
        print(res[res.period == label].drop(columns="period").set_index("strategy").to_string(formatters=fmt))
    s = res.set_index(["strategy", "period"])["Sharpe"]
    ok_ic = all(ic.loc[k, "mean IC"] > 0 and ic.loc[k, "t"] >= 2 for k in PERIODS)
    ok_use = all(s[("list tilted to potential", k)] > s[("buy-for-hold list", k)] for k in PERIODS)
    print(f"\nRanking skill in both periods: {ok_ic}; tilt beats the plain list in both: {ok_use} -> "
          f"{'SHOW a potential ranking' if ok_ic and ok_use else 'do NOT show a potential ranking'}")
    latest = p[p.index == p.index.max()].sort_values("pred", ascending=False)
    print("\nLatest month-end predictions (3-month return vs SPY):")
    print(latest[["ticker", "pred", "above_200d"]].to_string(index=False, float_format=lambda x: f"{x:+.1%}"))
    res.to_csv(ROOT / "reports" / "stock_potential.csv", index=False)


if __name__ == "__main__":
    main()
