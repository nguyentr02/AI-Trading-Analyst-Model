"""Which US stocks to buy for holding, and should you sell when the market crashes?

    .venv\\Scripts\\python experiments\\stock_hold_picks.py

Fixed before any result was seen (2026-10-06). Universe: the 15 Binance-listed US stocks (ETFs excluded), daily
adjusted closes from Yahoo, 0.1% cost per side on every change. Monthly rebalance at month ends unless stated.
- equal          benchmark: all 15 stocks, equal weight
- riskadj_mom5   the 5 with the best past-year return (skipping the last month) divided by past-year volatility
- lowvol5        the 5 with the lowest past-year volatility
- uptrend        equal weight among the stocks whose close is above their 200-day average
- crash_exit     equal weight, but everything in cash while the S&P 500 ETF (SPY) closes below its 200-day
                 average, checked daily (a market-crash exit; Faber 2007)
- crash_10       equal weight, but everything in cash once SPY is 10%+ below its 1-year high, back in when it
                 recovers to within 5% of it, checked daily
Picks: the best Sharpe on 2016-2022 among riskadj_mom5 / lowvol5 / uptrend; it becomes the "buy for hold" list
only if it also beats equal on Sharpe on 2023-2026. Crash exits: reported on both periods (2020 crash and 2022
bear market are in 2016-2022); an exit becomes an automatic SELL alert only if it beats equal on Sharpe in both
periods, otherwise crash alerts are information only.
Survivorship bias: these are today's well-known stocks, so all absolute returns are flattering.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import stocks  # noqa: E402

PICKS = [t for t in stocks.STOCKS if t not in stocks.ETFS]
COST = stocks.COST
PERIODS = {"2016-2022": ("2016-01-01", "2023-01-01"), "2023-2026": ("2023-01-01", "2100-01-01")}


def weights_monthly(closes, rule):
    """Target weights decided at each month end (from data up to that day), held until the next month end."""
    month_ends = closes.resample("ME").last().index
    r = np.log(closes).diff()
    out = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for i, m in enumerate(month_ends[:-1]):
        hist = closes[closes.index <= m]
        if len(hist) < 260:
            continue
        last = hist.iloc[-1]
        avail = last.dropna().index
        if rule == "equal":
            chosen = list(avail)
        elif rule == "uptrend":
            chosen = [t for t in avail if last[t] > hist[t].tail(200).mean()]
        else:
            vol = r[r.index <= m].tail(252)[avail].std()
            if rule == "lowvol5":
                chosen = list(vol.nsmallest(5).index)
            else:  # riskadj_mom5
                mom = hist[avail].iloc[-22] / hist[avail].iloc[-253] - 1
                chosen = list((mom / vol).nlargest(5).index)
        nxt = month_ends[i + 1]
        rows = (out.index > m) & (out.index <= nxt)
        if chosen:
            out.loc[rows, chosen] = 1.0 / len(chosen)
    return out


def crash_overlay(closes, spy, rule):
    """1 = invested, 0 = cash, decided at each daily close (applies from the next day)."""
    if rule == "crash_exit":
        on = (spy > spy.rolling(200).mean()).astype(float)
    else:  # crash_10: out at -10% from the 1-year high, back in within 5% of it
        dd = spy / spy.rolling(252, min_periods=60).max() - 1
        state, vals = 1.0, []
        for d in dd:
            if d <= -0.10:
                state = 0.0
            elif d >= -0.05:
                state = 1.0
            vals.append(state)
        on = pd.Series(vals, index=spy.index)
    return on.reindex(closes.index).ffill().shift(1).fillna(1.0)


def returns(closes, w):
    rets = closes.pct_change().fillna(0)
    held = w.shift(1).fillna(0)
    turnover = w.diff().abs().sum(axis=1).fillna(0)
    return (held * rets).sum(axis=1) - turnover * COST


def stats(r):
    eq = (1 + r).cumprod()
    return {"return": eq.iloc[-1] - 1, "Sharpe": r.mean() / r.std() * np.sqrt(252), "worst fall": (eq / eq.cummax() - 1).min()}


def main():
    closes = pd.DataFrame({t: stocks.history(t) for t in PICKS}).sort_index()
    spy = stocks.history("SPY").reindex(closes.index).ffill()
    eq_w = weights_monthly(closes, "equal")
    series = {"equal": returns(closes, eq_w)}
    for rule in ("riskadj_mom5", "lowvol5", "uptrend"):
        series[rule] = returns(closes, weights_monthly(closes, rule))
    for rule in ("crash_exit", "crash_10"):
        series[rule] = returns(closes, eq_w.mul(crash_overlay(closes, spy, rule), axis=0))
    rows = []
    for name, r in series.items():
        for label, (a, b) in PERIODS.items():
            w = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]
            w = w[w.index >= r[r != 0].index.min()] if (r != 0).any() else w
            rows.append({"strategy": name, "period": label, **stats(w)})
    res = pd.DataFrame(rows)
    fmt = {"return": "{:+.0%}".format, "Sharpe": "{:.2f}".format, "worst fall": "{:+.0%}".format}
    for label in PERIODS:
        print(f"\n=== {label} ===")
        print(res[res.period == label].drop(columns="period").set_index("strategy").to_string(formatters=fmt))
    s = res.set_index(["strategy", "period"])["Sharpe"]
    pick = max(("riskadj_mom5", "lowvol5", "uptrend"), key=lambda k: s[(k, "2016-2022")])
    pick_ok = s[(pick, "2023-2026")] > s[("equal", "2023-2026")]
    print(f"\nPicks chosen on 2016-2022: {pick} -> {'ADOPT as the buy-for-hold list' if pick_ok else 'not adopted (equal weight stays)'}")
    for rule in ("crash_exit", "crash_10"):
        ok = all(s[(rule, p)] > s[("equal", p)] for p in PERIODS)
        print(f"{rule}: {'ADOPT as an automatic sell alert' if ok else 'not adopted as automatic selling; alerts information only'}")
    res.to_csv(ROOT / "reports" / "stock_hold_picks.csv", index=False)


if __name__ == "__main__":
    main()
