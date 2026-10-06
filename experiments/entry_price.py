"""At what price should we buy in: the market price now, or a limit order below it?

    .venv\\Scripts\\python experiments\\entry_price.py

Fixed before any result was seen (2026-10-06). Every day is a buy decision at the daily close (crypto: 00:00 UTC
close of BTC/ETH/BNB/SOL; stocks: the US close of the 15 Binance-listed stocks). Plans:
- market            buy at the decision price
- limit k% for N    a buy order k% below the decision price (k in 1, 2, 3), waiting N days (N in 1, 3, 7); it fills
                    if any later low touches it; if not filled after N days, buy at that day's close instead
Measure: the price paid relative to the decision price (negative = cheaper), averaged over all decisions; also the
share of orders filled. Crypto uses hourly lows (finer than daily) from the 1h cache; stocks use daily lows. Limit
orders pay the maker fee and market buys the taker fee; on Binance both are 0.1%, so fees are ignored here.
Choice: the plan with the cheapest average entry on 2022-2024 (crypto) / 2016-2022 (stocks). Check on 2025-2026 /
2023-2026. A limit plan is recommended only if it is cheaper than market in BOTH periods.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config, data, stocks  # noqa: E402

OFFSETS, WAITS = (0.01, 0.02, 0.03), (1, 3, 7)


def plans_on(decisions, lows, closes, step_per_day):
    """decisions: Series of decision prices (index = decision time). lows/closes: finer bars after it."""
    out = []
    lo, cl = lows.to_numpy(), closes.to_numpy()
    idx = lows.index
    for t, px in decisions.items():
        start = idx.searchsorted(t, side="right")
        row = {"time": t}
        for k in OFFSETS:
            limit = px * (1 - k)
            for n in WAITS:
                end = start + n * step_per_day
                if end > len(lo):
                    row[f"{k:.0%} {n}d"] = np.nan
                    continue
                window = lo[start:end]
                hit = np.flatnonzero(window <= limit)
                row[f"{k:.0%} {n}d"] = (limit if len(hit) else cl[end - 1]) / px - 1
                row[f"{k:.0%} {n}d filled"] = float(len(hit) > 0)
        out.append(row)
    return pd.DataFrame(out).set_index("time")


def evaluate(name, frames, choose, check):
    df = pd.concat(frames).sort_index()
    plans = [f"{k:.0%} {n}d" for k in OFFSETS for n in WAITS]
    rows = []
    for label, (a, b) in (("choose", choose), ("check", check)):
        w = df[(df.index >= pd.Timestamp(a, tz="UTC")) & (df.index < pd.Timestamp(b, tz="UTC"))]
        for p in plans:
            rows.append({"period": label, "plan": p, "avg entry vs market": w[p].mean(),
                         "filled": w[f"{p} filled"].mean(), "decisions": int(w[p].notna().sum())})
    res = pd.DataFrame(rows)
    best = res[res.period == "choose"].sort_values("avg entry vs market").iloc[0]["plan"]
    chk = res[(res.period == "check") & (res.plan == best)].iloc[0]
    ok = res[(res.period == "choose") & (res.plan == best)].iloc[0]["avg entry vs market"] < 0 and \
        chk["avg entry vs market"] < 0
    print(f"\n=== {name} ===")
    piv = res.pivot(index="plan", columns="period", values=["avg entry vs market", "filled"])
    print(piv.to_string(float_format=lambda x: f"{x:+.2%}"))
    print(f"Chosen: {best}; check {chk['avg entry vs market']:+.2%} vs market (0) -> "
          f"{'RECOMMEND the limit plan' if ok else 'buy at market'}")
    return res, best, ok


def main():
    crypto = []
    for sym in config.SYMBOLS:
        d = data.drop_open_candle(data.load_cached(sym, "1d"), "1d")["close"]
        d.index = d.index + pd.Timedelta("1D")  # decision at the daily close
        h = data.load_cached(sym, "1h")
        h.index = h.index + pd.Timedelta("1h")  # a bar's low is known at its close
        crypto.append(plans_on(d[d.index >= "2021-12-01"], h["low"], h["close"], 24))
    r1 = evaluate("Crypto (BTC/ETH/BNB/SOL), hourly lows", crypto, ("2022-01-01", "2025-01-01"), ("2025-01-01", "2100-01-01"))
    st = []
    for t in [x for x in stocks.STOCKS if x not in stocks.ETFS]:
        c = stocks.candles(t)
        st.append(plans_on(c["close"], c["low"], c["close"], 1))
    r2 = evaluate("US stocks (15), daily lows", st, ("2016-01-01", "2023-01-01"), ("2023-01-01", "2100-01-01"))
    pd.concat([r1[0].assign(market="crypto"), r2[0].assign(market="stocks")]).to_csv(ROOT / "reports" / "entry_price.csv", index=False)


if __name__ == "__main__":
    main()
