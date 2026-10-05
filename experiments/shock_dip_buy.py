"""Does buying sudden crashes pay? A fixed, pre-registered test of a "buy the shock dip" rule.

    .venv\\Scripts\\python experiments\\shock_dip_buy.py

Fixed before any result was seen (2026-10-06):
- Trigger: a coin's close falls at least X% below its close one hour earlier (checked on 15-minute closes).
- Trade: buy at that 15-minute close, sell after holding H. No overlapping trades per coin.
- Variants: X in {6%, 8%, 10%}, H in {4h, 24h}. The best is chosen on 2022-2024 ONLY, by average net return
  per trade, then checked once on 2025-2026 and on 2019-2021 (neither used for choosing).
- Costs: 0.1% fee + 0.05% slippage per side; stress test at 0.5% slippage (spreads widen in crashes).
- Benchmark: the same number of trades at random 15-minute closes with the same holding time (1,000 draws).
  If shock-buying doesn't beat random timing, it is only catching the market's general drift.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config, data  # noqa: E402

FEE = 0.001
DROPS = (0.06, 0.08, 0.10)
HOLDS = {"4h": 16, "24h": 96}  # in 15-minute bars
PERIODS = {"2019-2021 (check)": ("2019-01-01", "2022-01-01"), "2022-2024 (choose)": ("2022-01-01", "2025-01-01"),
           "2025-2026 (check)": ("2025-01-01", "2027-01-01")}
RNG = np.random.default_rng(7)


def closes():
    return {s: data.load_cached(s, "15m")["close"] for s in config.SYMBOLS}


def shock_trades(c, drop, hold, start, end):
    """Entry bars where the 1-hour return is <= -drop, without overlap; returns gross trade returns."""
    r1h = c / c.shift(4) - 1
    fwd = c.shift(-hold) / c - 1
    idx = np.flatnonzero(((r1h <= -drop) & (c.index >= start) & (c.index < end) & fwd.notna()).to_numpy())
    taken, free_at = [], -1
    for i in idx:
        if i >= free_at:
            taken.append(i)
            free_at = i + hold
    return fwd.iloc[taken], c.index[taken]


def net(gross, slippage):
    """Net return of a round trip: buy and sell each pay the fee, and the price slips against you both ways."""
    return (1 + gross) * (1 - slippage) / (1 + slippage) * (1 - FEE) ** 2 - 1


def random_benchmark(c_all, n_by_coin, hold, start, end, slippage, draws=1000):
    """Average net return per trade of random entries (same count per coin, same holding)."""
    pools = {}
    for s, c in c_all.items():
        fwd = (c.shift(-hold) / c - 1)
        fwd = fwd[(fwd.index >= start) & (fwd.index < end)].dropna().to_numpy()
        pools[s] = fwd
    means = []
    for _ in range(draws):
        picks = np.concatenate([RNG.choice(pools[s], n, replace=False) for s, n in n_by_coin.items() if n and len(pools[s]) >= n])
        means.append(net(picks, slippage).mean())
    return np.array(means)


def evaluate(c_all, drop, hold_name, start, end, slippage=0.0005):
    hold = HOLDS[hold_name]
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    rets, n_by = [], {}
    for s, c in c_all.items():
        g, _ = shock_trades(c, drop, hold, start, end)
        rets.append(net(g.to_numpy(), slippage))
        n_by[s] = len(g)
    r = np.concatenate(rets) if rets else np.array([])
    out = {"drop": f"-{drop:.0%} in 1h", "hold": hold_name, "trades": len(r)}
    if len(r) == 0:
        return out
    bench = random_benchmark(c_all, n_by, hold, start, end, slippage)
    out.update({"avg net per trade": r.mean(), "median": np.median(r), "win rate": (r > 0).mean(),
                "worst trade": r.min(), "random entries avg": bench.mean(),
                "beats random (share of draws)": (r.mean() > bench).mean()})
    return out


def main():
    c_all = closes()
    fmt = {"avg net per trade": "{:+.2%}".format, "median": "{:+.2%}".format, "win rate": "{:.0%}".format,
           "worst trade": "{:+.1%}".format, "random entries avg": "{:+.2%}".format, "beats random (share of draws)": "{:.0%}".format}
    a, b = PERIODS["2022-2024 (choose)"]
    grid = pd.DataFrame([evaluate(c_all, d, h, a, b) for d in DROPS for h in HOLDS])
    print("=== Choosing on 2022-2024 ===")
    print(grid.to_string(index=False, formatters=fmt))
    best = grid.dropna(subset=["avg net per trade"]).sort_values("avg net per trade").iloc[-1]
    drop, hold = float(best["drop"].split("%")[0].lstrip("-")) / 100, best["hold"]
    print(f"\nChosen: drop {drop:.0%} in 1 hour, hold {hold}")
    print("\n=== Checking the chosen rule on periods not used for choosing ===")
    rows = []
    for label in ("2019-2021 (check)", "2025-2026 (check)"):
        a, b = PERIODS[label]
        for slip in (0.0005, 0.005):
            rows.append({"period": label, "slippage": f"{slip:.2%}", **evaluate(c_all, drop, hold, a, b, slip)})
    print(pd.DataFrame(rows).drop(columns=["drop", "hold"]).to_string(index=False, formatters=fmt))


if __name__ == "__main__":
    main()
