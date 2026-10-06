"""Split entries (Binance DCA-style) when the 50-day rule turns long, instead of buying all at once?

    .venv\\Scripts\\python experiments\\split_entries.py

Fixed before any result was seen (2026-10-06). See docs/research/binance-trading-bots.md. The 50-day rule wins
only ~1 trade in 4: most entries are whipsaws that reverse within days. Buying in steps would put less money
into those, at the price of entering real trends later (Vanguard: lump sum beats DCA ~2/3 of the time).
Daily decisions at the close, BTC/ETH/BNB/SOL, equal-weight basket, 0.1% fee + 0.05% slippage per side on
every change in position. Each coin's position while its close is above the 50-day average:
- sma50         benchmark: 100% from the first day (lump sum); exit all at once
- split3        1/3 more each day for 3 days; exit all at once
- split5        1/5 more each day for 5 days; exit all at once
- split10       1/10 more each day for 10 days; exit all at once
- split5_both   in over 5 days; out over 5 days as well (1/5 of the full position a day once below the average)
- buy_hold      benchmark
Choice: best basket Sharpe on 2022-2024 among the four split variants. Check: 2025-01-01 to 2026-10-06, once.
Adopted only if it also beats sma50 on Sharpe on the check period.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config, data, metrics  # noqa: E402

COST = config.FEE + 0.0005
START, END = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2026-10-06", tz="UTC")
CHOOSE, CHECK = (START, pd.Timestamp("2025-01-01", tz="UTC")), (pd.Timestamp("2025-01-01", tz="UTC"), END)


def ramp(flag, n_in, n_out=1):
    """Position stepping up by 1/n_in a day while the flag is on, and down by 1/n_out a day while it is off."""
    out, pos = [], 0.0
    for f in flag:
        pos = min(1.0, pos + 1.0 / n_in) if f else max(0.0, pos - 1.0 / n_out)
        out.append(pos)
    return pd.Series(out, index=flag.index)


def main():
    rets = {}
    for sym in config.SYMBOLS:
        c = data.drop_open_candle(data.load_cached(sym, "1d"), "1d")["close"]
        flag = (c > c.rolling(50).mean()).astype(float)
        r_next = c.shift(-1) / c - 1
        pos = pd.DataFrame({"sma50": flag, "split3": ramp(flag, 3), "split5": ramp(flag, 5), "split10": ramp(flag, 10),
                            "split5_both": ramp(flag, 5, 5), "buy_hold": 1.0})
        pos = pos[(pos.index >= START - pd.Timedelta("1D")) & (pos.index < END - pd.Timedelta("1D"))]
        change = pos.diff().abs()
        change.iloc[0] = pos.iloc[0]
        net = pos.mul(r_next.reindex(pos.index), axis=0) - change * COST
        net.index = net.index + pd.Timedelta("1D")  # earned by the next close
        rets[sym] = net.dropna()
    names = list(next(iter(rets.values())).columns)
    basket = {n: pd.concat({s: r[n] for s, r in rets.items()}, axis=1).mean(axis=1) for n in names}
    rows = []
    for n in names:
        for phase, (a, b) in (("choose 2022-2024", CHOOSE), ("check 2025-2026", CHECK)):
            w = basket[n][(basket[n].index > a) & (basket[n].index <= b)]
            eq = (1 + w).cumprod()
            rows.append({"strategy": n, "phase": phase, "return": eq.iloc[-1] - 1, "Sharpe": metrics.sharpe(w, 365),
                         "Sharpe SE": metrics.sharpe_se(w, 365), "worst drop": metrics.max_drawdown(eq)})
    res = pd.DataFrame(rows)
    fmt = {"return": "{:+.1%}".format, "Sharpe": "{:+.2f}".format, "Sharpe SE": "{:.2f}".format,
           "worst drop": "{:+.1%}".format}
    for phase in ("choose 2022-2024", "check 2025-2026"):
        print(f"\n=== {phase} (basket of 4 coins) ===")
        print(res[res.phase == phase].drop(columns="phase").set_index("strategy").to_string(formatters=fmt))
    cho = res[res.phase == "choose 2022-2024"].set_index("strategy")
    chk = res[res.phase == "check 2025-2026"].set_index("strategy")
    chosen = cho.loc[["split3", "split5", "split10", "split5_both"], "Sharpe"].idxmax()
    better = chk.loc[chosen, "Sharpe"] > chk.loc["sma50", "Sharpe"]
    print(f"\nChosen on 2022-2024: {chosen}")
    print(f"Check 2025-2026: Sharpe {chk.loc[chosen, 'Sharpe']:+.2f} vs sma50 {chk.loc['sma50', 'Sharpe']:+.2f} -> "
          f"{'ADOPT' if better else 'not adopted'}")
    res.to_csv(ROOT / "reports" / "split_entries.csv", index=False)


if __name__ == "__main__":
    main()
