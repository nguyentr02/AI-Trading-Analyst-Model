"""Does the 50-day trend rule do better across more coins?

    .venv\\Scripts\\python experiments\\trend_universe.py

Fixed before any result was seen (2026-10-06). Man AHL (2024) found trend-following portfolio Sharpe in crypto
peaks at about 10-15 coins. Daily decisions, long or cash per coin (spot), equal weight across the chosen coins,
0.1% fee + 0.05% slippage per side on every change in a coin's position.
Survivorship: the pool is 36 coins that were large or heavily traded at some point 2021-2026, including ones
later delisted or renamed (LUNA, FTT, MATIC, EOS). A price series with a gap of more than 3 days is split into
separate listings (Binance reused "LUNA" for the new coin after the May 2022 crash). At each month start the
universe is the N listings with the highest average daily USDT volume over the previous 30 days, among those
with at least 60 days of history; a listing that ends mid-month sits in cash for the rest of the month.
- majors4       benchmark: BTC, ETH, BNB, SOL (the live coins), 50-day rule
- top4/8/12/16  the N most traded each month, 50-day rule
- hold12        benchmark: equal-weight buy & hold of the 12 most traded each month
Choice: best Sharpe on 2022-2024 among top8, top12, top16. Check: 2025-01-01 to 2026-10-06, once.
Adopted (as the reason to widen the paper/live coin list) only if it also beats majors4 on Sharpe on the check.
Simplification: sleeves are treated as rebalanced to equal weight daily without cost.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config, data, metrics  # noqa: E402

POOL = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX", "MATIC", "POL", "LINK", "LTC", "TRX", "UNI",
        "ATOM", "LUNA", "LUNC", "FTT", "SHIB", "NEAR", "ETC", "BCH", "FIL", "SAND", "MANA", "AXS", "ALGO", "XLM", "EOS",
        "APT", "ARB", "OP", "SUI", "PEPE", "TON"]
COST = config.FEE + 0.0005
START, END = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2026-10-06", tz="UTC")
CHOOSE, CHECK = (START, pd.Timestamp("2025-01-01", tz="UTC")), (pd.Timestamp("2025-01-01", tz="UTC"), END)
MAJORS = ["BTC#1", "ETH#1", "BNB#1", "SOL#1"]


def listings():
    """{name: daily candles} with each coin split at gaps of more than 3 days (name#1, name#2, ...)."""
    out = {}
    for b in POOL:
        d = data.drop_open_candle(data.load_cached(f"{b}/USDT", "1d"), "1d")
        seg = (d.index.to_series().diff() > pd.Timedelta("3D")).cumsum()
        for i, part in d.groupby(seg):
            out[f"{b}#{i + 1}"] = part
    return out


def main():
    lst = listings()
    days = pd.date_range(START - pd.Timedelta("400D"), END, freq="D", tz="UTC")
    close = pd.DataFrame({k: v["close"] for k, v in lst.items()}).reindex(days)
    volume = pd.DataFrame({k: v["quote_volume"] for k, v in lst.items()}).reindex(days)
    # A position taken at day t's close earns t+1's return; 0 after a listing ends.
    r_next = (close.shift(-1) / close - 1).fillna(0.0)
    flag = (close > close.rolling(50, min_periods=50).mean()).astype(float)
    age = close.notna().cumsum()

    months = pd.date_range(START, END, freq="MS", tz="UTC")
    members = {}  # (N, month) -> list of listings
    for m0 in months:
        prev = m0 - pd.Timedelta("1D")
        ok = (age.loc[prev] >= 60) & close.loc[prev].notna()
        vol30 = volume.loc[m0 - pd.Timedelta("30D"):prev].mean()[ok].sort_values(ascending=False)
        for n in (4, 8, 12, 16):
            members[(n, m0)] = list(vol30.index[:n])

    def run(name, pick, rule):
        """Daily portfolio returns; pick(month) -> listings, rule: 'trend' or 'hold'."""
        rets, prev_pos = [], pd.Series(0.0, index=close.columns)
        for m0, m1 in zip(months, list(months[1:]) + [END]):
            coins = pick(m0)
            for t in pd.date_range(m0, m1 - pd.Timedelta("1D"), freq="D", tz="UTC"):
                if t > END - pd.Timedelta("2D"):
                    break
                alive = [c for c in coins if not np.isnan(close.at[t, c])]
                pos = pd.Series(0.0, index=close.columns)
                if alive:
                    w = 1.0 / len(coins)  # a listing that ended leaves its sleeve in cash
                    pos[alive] = w * (flag.loc[t, alive] if rule == "trend" else 1.0)
                cost = (pos - prev_pos).abs().sum() * COST
                rets.append((t + pd.Timedelta("1D"), float((pos * r_next.loc[t]).sum() - cost)))
                prev_pos = pos
        return pd.Series(dict(rets)).sort_index()

    strategies = {"majors4": run("majors4", lambda m: MAJORS, "trend")}
    for n in (4, 8, 12, 16):
        strategies[f"top{n}"] = run(f"top{n}", lambda m, n=n: members[(n, m)], "trend")
    strategies["hold12"] = run("hold12", lambda m: members[(12, m)], "hold")

    rows = []
    for name, r in strategies.items():
        for phase, (a, b) in (("choose 2022-2024", CHOOSE), ("check 2025-2026", CHECK)):
            w = r[(r.index > a) & (r.index <= b)]
            eq = (1 + w).cumprod()
            rows.append({"strategy": name, "phase": phase, "return": eq.iloc[-1] - 1, "Sharpe": metrics.sharpe(w, 365),
                         "Sharpe SE": metrics.sharpe_se(w, 365), "worst drop": metrics.max_drawdown(eq)})
    res = pd.DataFrame(rows)
    fmt = {"return": "{:+.1%}".format, "Sharpe": "{:+.2f}".format, "Sharpe SE": "{:.2f}".format,
           "worst drop": "{:+.1%}".format}
    for phase in ("choose 2022-2024", "check 2025-2026"):
        print(f"\n=== {phase} ===")
        print(res[res.phase == phase].drop(columns="phase").set_index("strategy").to_string(formatters=fmt))

    seen = pd.Series([c.split("#")[0] for (n, m), cs in members.items() if n == 12 for c in cs]).value_counts()
    print("\nMonths in the top 12:", ", ".join(f"{k} {v}" for k, v in seen.items()))
    print("Top 12 now:", ", ".join(c.split("#")[0] for c in members[(12, months[-1])]))
    cho = res[res.phase == "choose 2022-2024"].set_index("strategy")
    chk = res[res.phase == "check 2025-2026"].set_index("strategy")
    chosen = cho.loc[["top8", "top12", "top16"], "Sharpe"].idxmax()
    better = chk.loc[chosen, "Sharpe"] > chk.loc["majors4", "Sharpe"]
    print(f"\nChosen on 2022-2024: {chosen}")
    print(f"Check 2025-2026: Sharpe {chk.loc[chosen, 'Sharpe']:+.2f} vs majors4 {chk.loc['majors4', 'Sharpe']:+.2f} -> "
          f"{'ADOPT' if better else 'not adopted'}")
    res.to_csv(ROOT / "reports" / "trend_universe.csv", index=False)


if __name__ == "__main__":
    main()
