"""Keep the AI's edge but trade far less.

    .venv\\Scripts\\python experiments\\low_turnover_ai.py

Fixed before any result was seen (2026-10-06). experiments/calibrated_sizing.py found that sizing by the AI's
calibrated "Next 1 day" P(up) has a high Sharpe BEFORE fees (2022-2026 gross Sharpe 1.54-1.58 vs 1.03 for the
50-day rule) but turns the position over 187-340 times a year, so 0.1% fees cost 19-34% a year. Here the same
signal is traded slowly. Calibrated P(up) as in calibrated_sizing.py; "smoothed" = average of the last 6 candles
(1 day). All variants use the drop-warning brake. 0.1% fee per side, decisions at 4h closes unless stated.
- sma50_ai_daily  50-day rule x clip((smoothed P - 0.45) / 0.10, 0, 1), decided once a day at 00:00 UTC, in
                  steps of a quarter (0, 25, 50, 75, 100%)
- ai_hyst         AI alone: all in when smoothed P >= 0.53, all out when <= 0.47, otherwise keep
- sma50_ai_hyst   the 50-day rule must be long AND ai_hyst in
Benchmarks: sma50 and smart_drop (Smart with the drop exit).
Choice: best basket Sharpe on 2022-2024 among the three. Check: 2025-01-01 to 2026-10-06, once.
Adopted only if it beats both benchmarks on Sharpe on the check period. Turnover reported.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from cryptoai import config, metrics  # noqa: E402
import calibrated_sizing as C  # noqa: E402
import exit_timing  # noqa: E402

CHOOSE, CHECK = C.CHOOSE, C.CHECK


def hysteresis(p, on_at=0.53, off_at=0.47):
    out, pos = [], 0.0
    for x in p:
        pos = 1.0 if x >= on_at else 0.0 if x <= off_at else pos
        out.append(pos)
    return pd.Series(out, index=p.index)


def daily_steps(target):
    """Re-decide only at the 00:00 UTC decision (the candle opening 20:00), in quarter steps; hold otherwise."""
    out, cur = [], 0.0
    for t, x in target.items():
        if t.hour == 20:
            cur = np.round(x * 4) / 4
        out.append(cur)
    return pd.Series(out, index=target.index)


def main():
    sp = pickle.load(open(exit_timing.SMART_PRED, "rb"))
    dp = pickle.load(open(exit_timing.DROP_PRED, "rb"))
    per, turnover = {}, {}
    for sym in config.SYMBOLS:
        f = C.coin_frame(sym, sp["4h"], dp)
        br = C.brake(f["pdrop"])
        smooth = f["p_cal"].rolling(6, min_periods=1).mean()
        hyst = hysteresis(smooth)
        pos = {
            "sma50": f["above50"],
            # the brake can still cut the position between daily decisions
            "sma50_ai_daily": daily_steps(f["above50"] * ((smooth - 0.45) / 0.10).clip(0, 1)) * br,
            "ai_hyst": hyst * br,
            "sma50_ai_hyst": f["above50"] * hyst * br,
        }
        years = (f.index[-1] - f.index[0]).days / 365
        per[sym] = {k: C.returns(v, f["r_next"]) for k, v in pos.items()}
        turnover[sym] = {k: v.diff().abs().sum() / years for k, v in pos.items()}
    probs = {**{n: sp[n] for n in config.MODELS}, "drop": dp[["symbol", "prob"]]}
    turn = pd.DataFrame(turnover).mean(axis=1)

    rows = []
    for phase, (a, b) in (("choose 2022-2024", CHOOSE), ("check 2025-2026", CHECK)):
        strategies = {}
        for name in ("sma50", "sma50_ai_daily", "ai_hyst", "sma50_ai_hyst"):
            coin = {s: per[s][name][(per[s][name].index >= a) & (per[s][name].index < b)] for s in config.SYMBOLS}
            strategies[name] = pd.concat(coin, axis=1).fillna(0).mean(axis=1)
        smart = {s: exit_timing.run(s, a, b, probs, exit_timing.VARIANTS["drop_all"])[0] for s in config.SYMBOLS}
        strategies["smart_drop"] = pd.concat(smart, axis=1).fillna(0).mean(axis=1)
        for name, r in strategies.items():
            eq = (1 + r).cumprod()
            rows.append({"phase": phase, "strategy": name, "return": eq.iloc[-1] - 1, "Sharpe": metrics.sharpe(r, 365),
                         "Sharpe SE": metrics.sharpe_se(r, 365), "worst drop": metrics.max_drawdown(eq),
                         "turnover / year": turn.get(name, np.nan)})
    res = pd.DataFrame(rows)
    fmt = {"return": "{:+.1%}".format, "Sharpe": "{:+.2f}".format, "Sharpe SE": "{:.2f}".format,
           "worst drop": "{:+.1%}".format, "turnover / year": "{:.0f}x".format}
    for phase in ("choose 2022-2024", "check 2025-2026"):
        print(f"\n=== {phase} (basket of 4 coins) ===")
        print(res[res.phase == phase].drop(columns="phase").set_index("strategy").to_string(formatters=fmt, na_rep="-"))
    cho = res[res.phase == "choose 2022-2024"].set_index("strategy")
    chk = res[res.phase == "check 2025-2026"].set_index("strategy")
    chosen = cho.loc[["sma50_ai_daily", "ai_hyst", "sma50_ai_hyst"], "Sharpe"].idxmax()
    bar = max(chk.loc["smart_drop", "Sharpe"], chk.loc["sma50", "Sharpe"])
    print(f"\nChosen on 2022-2024: {chosen}")
    print(f"Check 2025-2026: Sharpe {chk.loc[chosen, 'Sharpe']:+.2f} vs best benchmark {bar:+.2f} -> "
          f"{'ADOPT' if chk.loc[chosen, 'Sharpe'] > bar else 'not adopted'}")
    res.to_csv(ROOT / "reports" / "low_turnover_ai.csv", index=False)


if __name__ == "__main__":
    main()
