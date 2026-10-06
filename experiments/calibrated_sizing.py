"""Size by calibrated confidence instead of fixed tiers, and combine the AI with the 50-day rule?

    .venv\\Scripts\\python experiments\\calibrated_sizing.py

Fixed before any result was seen (2026-10-06). experiments/ai_vs_trend.py found the "Next 1 day" P(up) carries
direction information the trend does not (IC +0.049 after removing trend, t 3.3, similar in both periods), while
the trend score itself has ~0 IC at 1 day: the two are complementary. Decisions at every 4h close, BTC/ETH/BNB/SOL,
equal-weight basket, 0.1% fee per side (as the Smart backtests). All variants use the drop-warning brake that was
adopted for Smart (cash while P(drop) >= 40%, allowed again under 35%) and skip position changes under 10 points.
Calibration: isotonic regression of the outcome (price higher 6 candles later) on P(up), refitted each month on
all earlier out-of-sample predictions whose outcome is known (raw P(up) until 3 months of history exist).
- smart_drop     benchmark: the Smart strategy with the drop exit (experiments/exit_timing.py, drop_all)
- sma50          benchmark: the 50-day rule
- lin_raw        position = clip((P(up) - 0.5) x 10, 0, 1): 0 at 50%, all in at 60%
- lin_cal        the same on calibrated P(up)
- lin_cal_smooth the same on the average calibrated P(up) of the last 3 candles (less churn)
- sma50_ai       50-day rule x clip((calibrated P(up) - 0.45) / 0.10, 0, 1): the trend decides whether, the AI how much
Choice: best basket Sharpe on 2022-2024 among the four new variants. Check: 2025-01-01 to 2026-10-06, once.
Adopted only if it beats both benchmarks on Sharpe on the check period.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from cryptoai import config, data, droprisk, metrics  # noqa: E402
import exit_timing  # noqa: E402

START, END = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2026-10-06", tz="UTC")
CHOOSE, CHECK = (START, pd.Timestamp("2025-01-01", tz="UTC")), (pd.Timestamp("2025-01-01", tz="UTC"), END)
STEP, BAND, H = pd.Timedelta("4h"), 0.10, 6


def calibrated(p, y):
    """Monthly-refitted isotonic calibration of P(up), using only predictions whose outcome was known."""
    out = p.copy()
    for m0 in pd.date_range(START, END, freq="MS", tz="UTC"):
        known = (p.index <= m0 - (H + 1) * STEP) & y.notna()
        test = (p.index >= m0) & (p.index < m0 + pd.offsets.MonthBegin(1))
        if (p.index[known] >= START).sum() < 3 * 180:  # about 3 months of 4h rows
            continue
        iso = IsotonicRegression(out_of_bounds="clip").fit(p[known], y[known])
        out[test] = iso.predict(p[test])
    return out


def brake(pdrop):
    out, on = [], 1.0
    for x in pdrop:
        on = 0.0 if x >= droprisk.SELL_AT else 1.0 if x < droprisk.CAUTION_AT else on
        out.append(on)
    return pd.Series(out, index=pdrop.index)


def banded(target):
    out, cur = [], 0.0
    for t in target:
        if abs(t - cur) >= BAND or t in (0.0, 1.0):
            cur = t
        out.append(cur)
    return pd.Series(out, index=target.index)


def coin_frame(sym, p1, pdrop):
    c4 = data.closed("4h", refresh=False, symbols=[sym])[sym]["close"]
    d = data.drop_open_candle(data.load_cached(sym, "1d"), "1d")["close"]
    daily = pd.DataFrame({"above50": (d > d.rolling(50).mean()).astype(float), "known_at": d.index + pd.Timedelta("1D")})
    f = pd.DataFrame({"r_next": c4.shift(-1) / c4 - 1, "y": (c4.shift(-H) > c4).astype(float).where(c4.shift(-H).notna()),
                      "known_at": c4.index + STEP}, index=c4.index)
    f.index.name = "time"
    f = pd.merge_asof(f.reset_index().sort_values("known_at"), daily.reset_index(drop=True).sort_values("known_at"),
                      on="known_at", direction="backward").set_index("time")
    f["p"] = p1[p1["symbol"] == sym]["prob"].reindex(f.index)
    f["pdrop"] = pdrop[pdrop["symbol"] == sym]["prob"].reindex(f.index)
    f = f[(f.index >= START) & (f.index < END)].dropna(subset=["p", "pdrop", "r_next"])
    f["p_cal"] = calibrated(f["p"], f["y"])
    return f


def returns(pos, r_next):
    change = pos.diff().abs()
    change.iloc[0] = pos.iloc[0]
    r4 = pos * r_next - change * config.FEE
    return (1 + r4).groupby(r4.index.floor("1D")).prod() - 1


def main():
    sp = pickle.load(open(exit_timing.SMART_PRED, "rb"))
    dp = pickle.load(open(exit_timing.DROP_PRED, "rb"))
    lin = lambda p: (p - 0.5) * 10
    per = {}
    for sym in config.SYMBOLS:
        f = coin_frame(sym, sp["4h"], dp)
        br = brake(f["pdrop"])
        pos = {
            "sma50": f["above50"],
            "lin_raw": banded(lin(f["p"]).clip(0, 1) * br),
            "lin_cal": banded(lin(f["p_cal"]).clip(0, 1) * br),
            "lin_cal_smooth": banded(lin(f["p_cal"].rolling(3, min_periods=1).mean()).clip(0, 1) * br),
            "sma50_ai": banded(f["above50"] * ((f["p_cal"] - 0.45) / 0.10).clip(0, 1) * br),
        }
        per[sym] = {k: returns(v, f["r_next"]) for k, v in pos.items()}
    probs = {**{n: sp[n] for n in config.MODELS}, "drop": dp[["symbol", "prob"]]}

    rows = []
    for phase, (a, b) in (("choose 2022-2024", CHOOSE), ("check 2025-2026", CHECK)):
        strategies = {}
        for name in ("sma50", "lin_raw", "lin_cal", "lin_cal_smooth", "sma50_ai"):
            coin = {s: per[s][name][(per[s][name].index >= a) & (per[s][name].index < b)] for s in config.SYMBOLS}
            strategies[name] = pd.concat(coin, axis=1).fillna(0).mean(axis=1)
        smart = {s: exit_timing.run(s, a, b, probs, exit_timing.VARIANTS["drop_all"])[0] for s in config.SYMBOLS}
        strategies["smart_drop"] = pd.concat(smart, axis=1).fillna(0).mean(axis=1)
        for name, r in strategies.items():
            eq = (1 + r).cumprod()
            rows.append({"phase": phase, "strategy": name, "return": eq.iloc[-1] - 1, "Sharpe": metrics.sharpe(r, 365),
                         "Sharpe SE": metrics.sharpe_se(r, 365), "worst drop": metrics.max_drawdown(eq)})
    res = pd.DataFrame(rows)
    fmt = {"return": "{:+.1%}".format, "Sharpe": "{:+.2f}".format, "Sharpe SE": "{:.2f}".format,
           "worst drop": "{:+.1%}".format}
    for phase in ("choose 2022-2024", "check 2025-2026"):
        print(f"\n=== {phase} (basket of 4 coins) ===")
        print(res[res.phase == phase].drop(columns="phase").set_index("strategy").to_string(formatters=fmt))
    cho = res[res.phase == "choose 2022-2024"].set_index("strategy")
    chk = res[res.phase == "check 2025-2026"].set_index("strategy")
    chosen = cho.loc[["lin_raw", "lin_cal", "lin_cal_smooth", "sma50_ai"], "Sharpe"].idxmax()
    bar = max(chk.loc["smart_drop", "Sharpe"], chk.loc["sma50", "Sharpe"])
    print(f"\nChosen on 2022-2024: {chosen}")
    print(f"Check 2025-2026: Sharpe {chk.loc[chosen, 'Sharpe']:+.2f} vs best benchmark {bar:+.2f} "
          f"(smart_drop {chk.loc['smart_drop', 'Sharpe']:+.2f}, sma50 {chk.loc['sma50', 'Sharpe']:+.2f}) -> "
          f"{'ADOPT' if chk.loc[chosen, 'Sharpe'] > bar else 'not adopted'}")
    res.to_csv(ROOT / "reports" / "calibrated_sizing.csv", index=False)


if __name__ == "__main__":
    main()
