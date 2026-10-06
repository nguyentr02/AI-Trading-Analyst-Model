"""Shock dip-buy with a trigger scaled to each coin's volatility (Binance-style "auto" parameters)?

    .venv\\Scripts\\python experiments\\vol_scaled_shock.py

Fixed before any result was seen (2026-10-06). Binance sizes its "AI" grid parameters from each coin's own
volatility (docs/research/binance-trading-bots.md). Our shock dip-buy (experiments/shock_dip_buy.py, used in both
paper trials) triggers on the same 10% one-hour fall for every coin; that is rare for BTC and routine for DOGE or
ZEC. Coins: the 4 main coins and the 6 altcoins of the altcoin trial. Hold 4 hours, no overlapping trades per coin,
0.1% fee + 0.05% slippage per side (stress test 0.5%), random-entry benchmark as in shock_dip_buy.py.
- fixed10   current rule: 1-hour return <= -10%
- scaled_k  1-hour return <= -k x the coin's 1-hour volatility over the previous 30 days, k in {4, 5, 6}
Choice: the k with the best average net return per trade on 2022-2024 (at least 30 trades). Check on 2025-01 to
2026-10, once. Adopted only if there it (1) beats random entries in at least 95% of draws, (2) has a positive
average net return per trade, and (3) a total net return (sum over trades, each trade the same size) at least as
large as fixed10's.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from cryptoai import altcoins, config, data  # noqa: E402
from shock_dip_buy import net, random_benchmark  # noqa: E402

HOLD = 16  # 15-minute bars = 4 hours
KS = (4, 5, 6)
COINS = [*config.SYMBOLS, *altcoins.WATCH]
PERIODS = {"choose 2022-2024": ("2022-01-01", "2025-01-01"), "check 2025-2026": ("2025-01-01", "2026-10-06")}


def trades(c, threshold, start, end):
    """Non-overlapping entries where the 1-hour return is at or below -threshold (a number or a Series)."""
    r1h = c / c.shift(4) - 1
    fwd = c.shift(-HOLD) / c - 1
    hit = (r1h <= -threshold) & (c.index >= start) & (c.index < end) & fwd.notna()
    taken, free_at = [], -1
    for i in np.flatnonzero(hit.to_numpy()):
        if i >= free_at:
            taken.append(i)
            free_at = i + HOLD
    return fwd.iloc[taken]


def evaluate(closes, thresholds, start, end, slippage=0.0005):
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    rets, n_by, per_coin = [], {}, {}
    for s, c in closes.items():
        g = trades(c, thresholds[s], start, end)
        rets.append(net(g.to_numpy(), slippage))
        n_by[s] = len(g)
        per_coin[s.split("/")[0]] = len(g)
    r = np.concatenate(rets)
    out = {"trades": len(r), "per coin": per_coin}
    if len(r) == 0:
        return out
    bench = random_benchmark(closes, n_by, HOLD, start, end, slippage, draws=500)
    out.update({"avg net": r.mean(), "win rate": (r > 0).mean(), "total net": r.sum(), "worst": r.min(),
                "beats random": float((r.mean() > bench).mean())})
    return out


def main():
    closes = {s: data.load_cached(s, "15m")["close"] for s in COINS}
    vol = {s: (c / c.shift(4) - 1).rolling(2880, min_periods=1000).std().shift(1) for s, c in closes.items()}
    variants = {"fixed10": {s: 0.10 for s in COINS}, **{f"scaled_{k}": {s: k * vol[s] for s in COINS} for k in KS}}
    rows = []
    for phase, (a, b) in PERIODS.items():
        for name, th in variants.items():
            rows.append({"phase": phase, "variant": name, **evaluate(closes, th, a, b)})
    res = pd.DataFrame(rows)
    fmt = {"avg net": "{:+.2%}".format, "win rate": "{:.0%}".format, "total net": "{:+.1%}".format,
           "worst": "{:+.1%}".format, "beats random": "{:.0%}".format}
    for phase in PERIODS:
        print(f"\n=== {phase} ===")
        print(res[res.phase == phase].drop(columns=["phase", "per coin"]).to_string(index=False, formatters=fmt))
    print("\nTrades per coin, 2025-2026:")
    for _, r in res[res.phase == "check 2025-2026"].iterrows():
        print(f"  {r['variant']:<9} {r['per coin']}")
    cho = res[(res.phase == "choose 2022-2024") & res.variant.str.startswith("scaled") & (res.trades >= 30)]
    chosen = cho.sort_values("avg net").iloc[-1]["variant"]
    chk = res[res.phase == "check 2025-2026"].set_index("variant")
    c, f = chk.loc[chosen], chk.loc["fixed10"]
    ok = c["beats random"] >= 0.95 and c["avg net"] > 0 and c["total net"] >= f.get("total net", 0)
    print(f"\nChosen on 2022-2024: {chosen}")
    print(f"Check 2025-2026: avg net {c['avg net']:+.2%}, beats random {c['beats random']:.0%}, total {c['total net']:+.1%} "
          f"vs fixed10 total {f.get('total net', 0):+.1%} ({int(f['trades'])} trades) -> {'ADOPT' if ok else 'not adopted'}")
    if ok:
        th = variants[chosen]
        a, b = PERIODS["check 2025-2026"]
        stress = evaluate(closes, th, a, b, slippage=0.005)
        print(f"Stress test, 0.5% slippage: avg net {stress['avg net']:+.2%}, total {stress['total net']:+.1%}")
    res.drop(columns="per coin").to_csv(ROOT / "reports" / "vol_scaled_shock.csv", index=False)


if __name__ == "__main__":
    main()
