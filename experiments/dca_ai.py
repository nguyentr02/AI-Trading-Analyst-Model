"""Regular buying (DCA): can the AI decide how much to buy each week?

    .venv\\Scripts\\python experiments\\dca_ai.py

Fixed before any result was seen (2026-10-06). For someone who adds money every week rather than trading.
$100 arrives in a cash wallet every Monday 00:00 UTC; each strategy decides how much of the wallet to spend, split
equally across BTC/ETH/BNB/SOL at the Sunday daily close, 0.1% fee per buy. Coins are never sold; unspent cash
earns nothing and counts at face value at the end. Score: final value / money put in (and the worst fall of
that ratio along the way).
- dca_plain   benchmark: spend $100 every week
- lump        benchmark: the whole period's money spent in the first week
- dca_dip     no AI: spend 2x when a coin is below its 50-day average (cheaper), 0.5x when above
- dca_trend   no AI: buy a coin only while it is above its 50-day average; save the rest for later
- dca_ai      the AI: weekly amount per coin x clip(1 + 5 x (P(up, next 3 days) - 0.5), 0.25, 2); nothing while
              the drop warning is High (>= 40%); saved cash is spent later when the multiplier allows
All spending is limited by the cash in the wallet. Out-of-sample AI predictions (monthly retrain) as in
exit_timing.py.
Choice: best final value / money in on 2022-2024 among dca_dip, dca_trend, dca_ai. Check: 2025-01 to 2026-10, once.
Adopted (as the AI's DCA advice) only if it beats dca_plain in BOTH periods.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from cryptoai import config, data, droprisk  # noqa: E402
import exit_timing  # noqa: E402

PERIODS = {"choose 2022-2024": ("2022-01-01", "2025-01-01"), "check 2025-2026": ("2025-01-01", "2026-10-06")}
WEEKLY, FEE = 100.0, config.FEE


def inputs():
    sp = pickle.load(open(exit_timing.SMART_PRED, "rb"))
    dp = pickle.load(open(exit_timing.DROP_PRED, "rb"))
    out = {}
    for sym in config.SYMBOLS:
        d = data.drop_open_candle(data.load_cached(sym, "1d"), "1d")["close"]
        f = pd.DataFrame({"close": d, "above50": d > d.rolling(50).mean()})
        f["p3"] = sp["1d"][sp["1d"]["symbol"] == sym]["prob"].reindex(f.index)
        drop = dp[dp["symbol"] == sym]["prob"]
        drop.index = drop.index + pd.Timedelta("4h")  # known at the 4h close
        f["pdrop"] = drop.reindex(f.index + pd.Timedelta("1D")).to_numpy()  # value at the daily close
        out[sym] = f
    return out


def run(strategy, f, start, end):
    """Weekly decisions on Sundays' closes. Returns (final value / money in, worst fall of that ratio)."""
    sundays = [t for t in f[config.SYMBOLS[0]].index if start <= t < end and t.dayofweek == 6]
    cash, invested, coins = 0.0, 0.0, {s: 0.0 for s in config.SYMBOLS}
    each = WEEKLY / len(config.SYMBOLS)
    total = WEEKLY * len(sundays)
    ratios = []
    for i, t in enumerate(sundays):
        cash += WEEKLY if strategy != "lump" else (total if i == 0 else 0.0)
        invested += WEEKLY
        want = {}
        for s in config.SYMBOLS:
            r = f[s].loc[t]
            if strategy in ("dca_plain", "lump"):
                want[s] = each if strategy == "dca_plain" else cash / len(config.SYMBOLS)
            elif strategy == "dca_dip":
                want[s] = each * (0.5 if r["above50"] else 2.0)
            elif strategy == "dca_trend":
                want[s] = cash / len(config.SYMBOLS) if r["above50"] else 0.0
            elif strategy == "dca_ai":
                m = 0.0 if r["pdrop"] >= droprisk.SELL_AT else float(np.clip(1 + 5 * (r["p3"] - 0.5), 0.25, 2.0))
                want[s] = each * m
        scale = min(1.0, cash / sum(want.values())) if sum(want.values()) > 0 else 0.0
        for s, usd in want.items():
            spend = usd * scale
            coins[s] += spend * (1 - FEE) / f[s].loc[t, "close"]
            cash -= spend
        value = cash + sum(coins[s] * f[s].loc[t, "close"] for s in config.SYMBOLS)
        ratios.append(value / (invested if strategy != "lump" else total))
    r = pd.Series(ratios)
    return r.iloc[-1], float((r / r.cummax() - 1).min()), cash / total


def main():
    f = inputs()
    rows = []
    for phase, (a, b) in PERIODS.items():
        a, b = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        for s in ("dca_plain", "lump", "dca_dip", "dca_trend", "dca_ai"):
            final, worst, left = run(s, f, a, b)
            rows.append({"phase": phase, "strategy": s, "value per $1 put in": final, "worst fall": worst,
                         "cash left at end": left})
    res = pd.DataFrame(rows)
    fmt = {"value per $1 put in": "${:.3f}".format, "worst fall": "{:+.1%}".format, "cash left at end": "{:.0%}".format}
    for phase in PERIODS:
        print(f"\n=== {phase}: $100 a week into BTC/ETH/BNB/SOL ===")
        print(res[res.phase == phase].drop(columns="phase").set_index("strategy").to_string(formatters=fmt))
    cho = res[res.phase == "choose 2022-2024"].set_index("strategy")["value per $1 put in"]
    chk = res[res.phase == "check 2025-2026"].set_index("strategy")["value per $1 put in"]
    chosen = cho[["dca_dip", "dca_trend", "dca_ai"]].idxmax()
    ok = cho[chosen] > cho["dca_plain"] and chk[chosen] > chk["dca_plain"]
    print(f"\nChosen on 2022-2024: {chosen}")
    print(f"vs plain DCA: 2022-2024 ${cho[chosen]:.3f} vs ${cho['dca_plain']:.3f}; 2025-2026 ${chk[chosen]:.3f} vs "
          f"${chk['dca_plain']:.3f} -> {'ADOPT' if ok else 'not adopted'}")
    res.to_csv(ROOT / "reports" / "dca_ai.csv", index=False)


if __name__ == "__main__":
    main()
