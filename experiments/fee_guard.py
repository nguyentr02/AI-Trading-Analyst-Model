"""Don't sell when the sale wouldn't even cover the fees, unless a stop-loss is needed?

    .venv\\Scripts\\python experiments\\fee_guard.py

Fixed before any result was seen (2026-10-06), from the user's rule: selling while the profit is smaller than the
buy + sell fees locks in a loss for nothing. Applied to AI Smart with the drop exit (the best AI variant,
experiments/exit_timing.py), whose sell signal fires when the next-1-day P(up) <= 48%. The drop-warning exit and
the +10% take-profit are never blocked (the first is a stop, the second is always above the fees). A sale is
"below fees" when price x (1 - fee) is below the average cost including the buy fee.
- smart_drop        benchmark: as now
- guard_tiny        skip the sell only when the price is above the price paid but not by enough to cover fees
- guard_stop5       skip any sell below fees, unless the loss after fees is 5% or more (a stop)
- guard_stop10      the same with the stop at 10%
Basket of BTC/ETH/BNB/SOL, $1000 per coin, 0.1% fee per side. Choice: best Sharpe on 2022-2024 among the three
guards. Check on 2025-01-01 to 2026-10-06, once. Adopted only if it beats smart_drop on Sharpe in BOTH periods.
Also reported: how many sell signals each guard blocked.
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from cryptoai import config, metrics  # noqa: E402
import exit_timing  # noqa: E402

DROP = exit_timing.VARIANTS["drop_all"]
VARIANTS = {
    "smart_drop": DROP,
    "guard_tiny": {**DROP, "fee_guard": {"mode": "tiny_profit"}},
    "guard_stop5": {**DROP, "fee_guard": {"mode": "no_loss", "stop": 0.05}},
    "guard_stop10": {**DROP, "fee_guard": {"mode": "no_loss", "stop": 0.10}},
}
PERIODS = {"choose 2022-2024": exit_timing.CHOOSE, "check 2025-2026": exit_timing.CHECK}


def main():
    sp = pickle.load(open(exit_timing.SMART_PRED, "rb"))
    dp = pickle.load(open(exit_timing.DROP_PRED, "rb"))
    probs = {**{n: sp[n] for n in config.MODELS}, "drop": dp[["symbol", "prob"]]}
    rows = []
    for phase, (a, b) in PERIODS.items():
        for name, rules in VARIANTS.items():
            coin_r, sells, small = {}, 0, 0
            for s in config.SYMBOLS:
                r, trades = exit_timing.run(s, a, b, probs, rules)
                coin_r[s] = r
                if len(trades):
                    signal_sells = trades[trades["action"].isin(["SELL all", "SELL half"])
                                          & (trades["reason"] != "end of the test period")]
                    sells += len(signal_sells)
                    small += int((signal_sells["profit"].abs() < signal_sells["usdt"] * 0.003).sum())
            basket = pd.concat(coin_r, axis=1).fillna(0).mean(axis=1)
            eq = (1 + basket).cumprod()
            rows.append({"phase": phase, "strategy": name, "return": eq.iloc[-1] - 1,
                         "Sharpe": metrics.sharpe(basket, 365), "worst drop": metrics.max_drawdown(eq),
                         "signal sells": sells, "of which near break-even": small})
    res = pd.DataFrame(rows)
    fmt = {"return": "{:+.1%}".format, "Sharpe": "{:+.2f}".format, "worst drop": "{:+.1%}".format}
    for phase in PERIODS:
        print(f"\n=== {phase} (basket of 4 coins) ===")
        print(res[res.phase == phase].drop(columns="phase").set_index("strategy").to_string(formatters=fmt))
    s = res.set_index(["phase", "strategy"])["Sharpe"]
    chosen = max(("guard_tiny", "guard_stop5", "guard_stop10"), key=lambda v: s[("choose 2022-2024", v)])
    ok = all(s[(p, chosen)] > s[(p, "smart_drop")] for p in PERIODS)
    print(f"\nChosen on 2022-2024: {chosen}")
    print(f"Sharpe vs smart_drop: 2022-2024 {s[('choose 2022-2024', chosen)]:+.2f} vs "
          f"{s[('choose 2022-2024', 'smart_drop')]:+.2f}; 2025-2026 {s[('check 2025-2026', chosen)]:+.2f} vs "
          f"{s[('check 2025-2026', 'smart_drop')]:+.2f} -> {'ADOPT' if ok else 'not adopted'}")
    res.to_csv(ROOT / "reports" / "fee_guard.csv", index=False)


if __name__ == "__main__":
    main()
