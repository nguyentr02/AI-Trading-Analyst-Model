"""If the AI must name ONE stock to buy, how should it choose, and how did that do?

    .venv\\Scripts\\python experiments\\stock_top_pick.py

Fixed before any result was seen (2026-10-06). At each month end, among the buy-for-hold list (stocks above their
200-day average; experiments/stock_hold_picks.py), hold ONE stock for the next month, chosen by:
- ai_5d       the highest stock-AI P(up, 5 trading days) (cryptoai/stockai.py walk-forward, out of sample)
- potential   the highest predicted 3-month return vs SPY (experiments/stock_potential.py, out of sample)
- strongest   the furthest above its 200-day average
- steadiest   the lowest past-year volatility
Benchmark: the whole buy-for-hold list, equal weight. 0.1% cost per side. Out-of-sample predictions start in 2021
(potential) / 2019 (ai_5d), so the comparison runs on 2021-2022 (choose) and 2023-2026 (check).
Choice: the best Sharpe on 2021-2022. It is shown as the "AI's top pick" in the app regardless (the user asked for
one pick), together with its record against the list; it is called better than the list only if it beat the list's
Sharpe in both periods.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from cryptoai import stockai, stocks  # noqa: E402
import stock_potential as SP  # noqa: E402

PERIODS = {"2021-2022": ("2021-01-01", "2023-01-01"), "2023-2026": ("2023-01-01", "2100-01-01")}
COST = stocks.COST


def month_end_table():
    ds, closes = SP.monthly_dataset()
    pot = SP.predictions(ds)[["ticker", "pred", "above_200d", "from_200d"]].rename(columns={"pred": "potential"})
    ai = stockai.walk_forward(stockai.dataset())
    ai = ai[ai["ticker"].isin(SP.PICKS)]
    ai_m = pd.DataFrame({"month": ai.index.to_period("M"), "ticker": ai["ticker"].to_numpy(), "ai_5d": ai["p"].to_numpy()})
    ai_m = ai_m.groupby(["month", "ticker"], as_index=False)["ai_5d"].last()
    vol = np.log(closes).diff().rolling(252).std()
    pot.index.name = "time"
    pot = pot.reset_index()
    pot["month"] = pot["time"].dt.tz_localize(None).dt.to_period("M")
    ai_m["month"] = ai_m["month"].astype(str)
    pot["month"] = pot["month"].astype(str)
    pot = pot.merge(ai_m, on=["month", "ticker"], how="left")
    pot["vol"] = [vol.at[t, k] if t in vol.index else np.nan for t, k in zip(pot["time"], pot["ticker"])]
    return pot.set_index("time"), closes


def run(tab, closes, rule):
    rets = closes.pct_change().fillna(0)
    w = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    months = sorted(set(tab.index))
    picks = []
    for i, m in enumerate(months):
        g = tab[(tab.index == m) & tab["above_200d"]]
        if g.empty:
            continue
        if rule == "list":
            chosen = list(g["ticker"])
        else:
            col, asc = {"ai_5d": ("ai_5d", False), "potential": ("potential", False), "strongest": ("from_200d", False),
                        "steadiest": ("vol", True)}[rule]
            g = g.dropna(subset=[col])
            if g.empty:
                continue
            chosen = [g.sort_values(col, ascending=asc).iloc[0]["ticker"]]
            picks.append((m, chosen[0]))
        nxt = months[i + 1] if i + 1 < len(months) else closes.index[-1] + pd.Timedelta("1D")
        rows = (w.index > m) & (w.index <= nxt)
        w.loc[rows, :] = 0.0
        w.loc[rows, chosen] = 1.0 / len(chosen)
    r = (w.shift(1).fillna(0) * rets).sum(axis=1) - w.diff().abs().sum(axis=1).fillna(0) * COST
    return r[r.index > months[0]], picks


def stats(r):
    eq = (1 + r).cumprod()
    return {"return": eq.iloc[-1] - 1, "Sharpe": r.mean() / r.std() * np.sqrt(252), "worst fall": (eq / eq.cummax() - 1).min()}


def main():
    tab, closes = month_end_table()
    rows, last = [], {}
    for rule in ("list", "ai_5d", "potential", "strongest", "steadiest"):
        r, picks = run(tab, closes, rule)
        last[rule] = picks[-3:]
        for label, (a, b) in PERIODS.items():
            w = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]
            rows.append({"rule": rule, "period": label, **stats(w)})
    res = pd.DataFrame(rows)
    fmt = {"return": "{:+.0%}".format, "Sharpe": "{:.2f}".format, "worst fall": "{:+.0%}".format}
    for label in PERIODS:
        print(f"\n=== {label} ===")
        print(res[res.period == label].drop(columns="period").set_index("rule").to_string(formatters=fmt))
    s = res.set_index(["rule", "period"])["Sharpe"]
    chosen = max(("ai_5d", "potential", "strongest", "steadiest"), key=lambda k: s[(k, "2021-2022")])
    better = all(s[(chosen, p)] > s[("list", p)] for p in PERIODS)
    print(f"\nTop-pick rule chosen on 2021-2022: {chosen}; beats the whole list in both periods: {better}")
    print("Recent picks:", {k: [(str(m.date()), t) for m, t in v] for k, v in last.items() if k != "list"})
    res.to_csv(ROOT / "reports" / "stock_top_pick.csv", index=False)


if __name__ == "__main__":
    main()
