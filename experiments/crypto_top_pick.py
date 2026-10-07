"""If the AI must name ONE coin to buy, how should it choose? (crypto version of stock_top_pick.py)

    .venv\\Scripts\\python experiments\\crypto_top_pick.py

Fixed before any result was seen (2026-10-07). Coins: BTC, ETH, BNB, SOL and the 6 altcoins of the altcoin trial
(NEAR, ZEC, XRP, DOGE, AVAX, LINK). The tested crypto core is the 50-day trend rule, so the candidates at each review
are the coins whose daily close is above their 50-day average. Held alone, reviewed monthly (and, as a check of
the user's daily idea, daily). One coin chosen by:
- steadiest   the lowest 90-day volatility
- strongest   the furthest above its 50-day average
- momentum    the best 90-day return
- bitcoin     BTC whenever it is above its 50-day average, else the steadiest candidate
Benchmarks: the 10-coin 50-day rule (each coin a fixed 1/10 share, held while above its 50-day average, cash
otherwise, daily; corrected before any conclusion from a first version that put all the money into whichever
coins were above) and holding BTC.
Daily closes, 0.1% fee + 0.05% slippage per side. Choose on 2022-2024 among the four (monthly), check on 2025-01-01
to now, once. A rule is shown as the crypto top pick; it is called better than the list only if it beats it in
both periods.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import altcoins, config, data, metrics  # noqa: E402

COINS = [*config.SYMBOLS, *altcoins.WATCH]
COST = config.FEE + 0.0005
PERIODS = {"2022-2024": ("2022-01-01", "2025-01-01"), "2025-2026": ("2025-01-01", "2100-01-01")}


def closes():
    return pd.DataFrame({s: data.drop_open_candle(data.load_cached(s, "1d"), "1d")["close"] for s in COINS}).sort_index()


def run(c, rule, every="monthly"):
    ma50 = c.rolling(50).mean()
    vol = np.log(c).diff().rolling(90).std()
    mom = c / c.shift(90) - 1
    above = (c > ma50) & ma50.notna()
    month_end = c.index.to_series().groupby(c.index.to_period("M")).transform("max") == c.index
    w = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    held = None
    for i, t in enumerate(c.index):
        if i < 120:
            continue
        if rule == "list":  # the tested 50-day rule: each coin its own 1/10 share, cash while below its average
            w.loc[t, above.loc[t]] = 1.0 / len(c.columns)
            continue
        if every == "daily" or month_end.loc[t] or held is None:
            cand = above.loc[t][above.loc[t]].index
            if len(cand) == 0:
                held = None
            elif rule == "bitcoin" and "BTC/USDT" in cand:
                held = "BTC/USDT"
            elif rule in ("steadiest", "bitcoin"):
                held = vol.loc[t, cand].idxmin()
            elif rule == "strongest":
                held = (c.loc[t, cand] / ma50.loc[t, cand]).idxmax()
            else:
                held = mom.loc[t, cand].idxmax()
        if held:
            w.loc[t, held] = 1.0
    rets = c.pct_change().fillna(0)
    held_w = w.shift(1).fillna(0)
    return (held_w * rets).sum(axis=1) - w.diff().abs().sum(axis=1).fillna(0) * COST


def stats(r):
    eq = (1 + r).cumprod()
    return {"return": eq.iloc[-1] - 1, "Sharpe": metrics.sharpe(r, 365), "worst fall": metrics.max_drawdown(eq)}


def main():
    c = closes()
    c = c[c.index >= "2020-10-01"]
    series = {"list (benchmark)": run(c, "list"), "hold BTC": c["BTC/USDT"].pct_change().fillna(0)}
    for rule in ("steadiest", "strongest", "momentum", "bitcoin"):
        series[rule] = run(c, rule)
    series["steadiest, daily re-pick"] = run(c, "steadiest", "daily")
    rows = []
    for name, r in series.items():
        for label, (a, b) in PERIODS.items():
            x = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]
            rows.append({"rule": name, "period": label, **stats(x)})
    res = pd.DataFrame(rows)
    fmt = {"return": "{:+.0%}".format, "Sharpe": "{:.2f}".format, "worst fall": "{:+.0%}".format}
    for label in PERIODS:
        print(f"\n=== {label} ===")
        print(res[res.period == label].drop(columns="period").set_index("rule").to_string(formatters=fmt))
    s = res.set_index(["rule", "period"])["Sharpe"]
    chosen = max(("steadiest", "strongest", "momentum", "bitcoin"), key=lambda k: s[(k, "2022-2024")])
    better = all(s[(chosen, p)] > s[("list (benchmark)", p)] for p in PERIODS)
    print(f"\nCrypto top-pick rule chosen on 2022-2024: {chosen}; beats the trend list in both periods: {better}")
    res.to_csv(ROOT / "reports" / "crypto_top_pick.csv", index=False)


if __name__ == "__main__":
    main()
