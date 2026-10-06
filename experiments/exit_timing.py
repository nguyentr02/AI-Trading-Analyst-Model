"""Can the AI learn WHEN to sell, before a drop, instead of only taking profit at +10%?

    .venv\\Scripts\\python experiments\\exit_timing.py

Fixed before any result was seen (2026-10-06).
Drop-warning model: same 4h features and model settings as the live "Next 1 day" model (15m/1h patterns
included), different question. Label (triple barrier, Lopez de Prado): from each 4h close, within the next
3 days (18 candles), does the price first fall 2 x the coin's recent daily volatility (std of the last 20
daily returns, measured on 4h returns x sqrt(6)) below the close, before rising the same amount? 1 = drop
first (a candle touching both counts as drop), 0 = rise first or neither. Retrained monthly, no look-ahead
(18-candle embargo). Coins: BTC, ETH, BNB, SOL (the live coins).

Exit strategies, all on top of the Smart strategy (simulate.SMART, otherwise unchanged), $1000 per coin, 0.1% fee:
- smart              as now (benchmark)
- trail              non-learned: once up 10% from the average buy price, sell half if price falls 5% from its peak
- drop_half          learned: sell half when P(drop) >= 0.40 (once per position); no new buys while P(drop) >= 0.35
- drop_all           learned: sell all when P(drop) >= 0.40; no new buys while P(drop) >= 0.35
- drop_half_no_tp    drop_half with the fixed +10% take-profit removed (does the learned exit replace it?)
Thresholds were first set at 0.60/0.55; before any strategy was run they were lowered to 0.40/0.35 (about 2x
the base rate) because only ~21% of closes have a drop first, so 0.60 would almost never trigger.
Choice: best basket Sharpe on 2022-2024 among the four new variants. Check: 2025-01-01 to 2026-10-06, once.
Adopted only if the chosen variant also beats smart on Sharpe on the check period.
Also reported: drop-model AUC, and for every exit sale, whether the price 3 days later was lower
(the sale dodged a fall) or higher (sold too early).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config, data, metrics, model, simulate  # noqa: E402
from cryptoai.droprisk import HORIZON as H, label as drop_label  # noqa: E402

START, END = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2026-10-06", tz="UTC")
CHOOSE, CHECK = (START, pd.Timestamp("2025-01-01", tz="UTC")), (pd.Timestamp("2025-01-01", tz="UTC"), END)
CASH, Q = 1000.0, 365
SMART_PRED = ROOT / "reports" / "predictions_smart_2022_2026.pkl"
DROP_PRED = ROOT / "reports" / "predictions_drop_2022_2026.pkl"
VARIANTS = {
    "smart": {},
    "trail": {"trail": {"after": 0.10, "pct": 0.05, "share": 0.5}},
    "drop_half": {"drop_sell_half": 0.40, "drop_block_buy": 0.35},
    "drop_all": {"drop_sell_all": 0.40, "drop_block_buy": 0.35},
    "drop_half_no_tp": {"drop_sell_half": 0.40, "drop_block_buy": 0.35, "take_profit": 10.0},
}


def smart_predictions():
    if SMART_PRED.exists():
        return pickle.load(open(SMART_PRED, "rb"))
    out = {n: simulate.predictions(n, START, END) for n in config.MODELS}
    pickle.dump(out, open(SMART_PRED, "wb"))
    return out


def drop_predictions():
    if DROP_PRED.exists():
        return pickle.load(open(DROP_PRED, "rb"))
    ds = model.dataset("4h", refresh=False)
    candles = data.closed("4h", refresh=False)
    for s in config.SYMBOLS:
        mask = (ds["symbol"] == s).to_numpy()
        ds.loc[mask, "y"] = drop_label(candles[s]).reindex(ds.index[mask]).to_numpy()
    cols = model.feature_cols(ds)
    step = pd.Timedelta("4h")
    months = pd.date_range(START, END, freq="MS", tz="UTC")
    parts = []
    for m0, m1 in zip(months, list(months[1:]) + [END]):
        train = ds[ds.index <= m0 - (H + 1) * step].dropna(subset=["y"])
        test = ds[(ds.index >= m0) & (ds.index < m1)]
        if test.empty:
            continue
        fitted = model._new_model().fit(train[cols], train["y"])
        parts.append(pd.DataFrame({"symbol": test["symbol"], "y": test["y"],
                                   "prob": fitted.predict_proba(test[cols])[:, 1]}, index=test.index))
    out = pd.concat(parts)
    pickle.dump(out, open(DROP_PRED, "wb"))
    return out


def run(sym, a, b, probs, rules):
    p = simulate._smart_inputs(probs, sym, a, b)
    p = p[p.index <= b - 2 * pd.Timedelta("4h")]
    candles = data.closed("4h", refresh=False, symbols=[sym])[sym]
    res = simulate._smart_trade(p, candles, CASH, config.FEE, {**simulate.SMART, **rules})
    return res["equity"].resample("1D").last().ffill().pct_change().dropna(), res["trades"]


def summarise(r):
    eq = CASH * (1 + r).cumprod()
    return {"final $": eq.iloc[-1], "return": eq.iloc[-1] / CASH - 1, "Sharpe": metrics.sharpe(r, Q),
            "Sharpe SE": metrics.sharpe_se(r, Q), "worst drop": metrics.max_drawdown(eq)}


def exit_quality(trades, sym):
    """For every exit sale (not end-of-period), the price change over the next 3 days."""
    if trades is None or not len(trades):
        return []
    c = data.closed("4h", refresh=False, symbols=[sym])[sym]["close"]
    close_at = pd.Series(c.values, index=c.index + pd.Timedelta("4h"))  # indexed by close time
    out = []
    for t in trades[trades["action"].str.startswith("SELL") & (trades["reason"] != "end of the test period")].itertuples():
        later = close_at[close_at.index >= t.time + pd.Timedelta("3D")]
        if len(later):
            out.append({"symbol": sym, "kind": t.action, "after_3d": later.iloc[0] / t.price - 1})
    return out


def main():
    pd.set_option("display.width", 200)
    sp = smart_predictions()
    dp = drop_predictions()
    lab = dp.dropna(subset=["y"])
    print(f"Drop label: {lab['y'].mean():.0%} of 4h closes are followed by a sharp drop first")
    for phase, (a, b) in (("2022-2024", CHOOSE), ("2025-2026", CHECK)):
        w = lab[(lab.index >= a) & (lab.index < b)]
        per = {s.split("/")[0]: roc_auc_score(g["y"], g["prob"]) for s, g in w.groupby("symbol")}
        hit = w[w["prob"] >= 0.40]
        print(f"Drop-model AUC {phase}: all {roc_auc_score(w['y'], w['prob']):.3f}  "
              + "  ".join(f"{k} {v:.3f}" for k, v in per.items())
              + f"  | P(drop)>=40%: {len(hit) / len(w):.1%} of candles, drop followed {hit['y'].mean():.0%} "
                f"(vs {w['y'].mean():.0%} overall)")

    probs = {**{n: sp[n] for n in config.MODELS}, "drop": dp[["symbol", "prob"]]}
    rows, quality = [], []
    for phase, (a, b) in (("choose 2022-2024", CHOOSE), ("check 2025-2026", CHECK)):
        for name, rules in VARIANTS.items():
            coin_r = {}
            for sym in config.SYMBOLS:
                r, trades = run(sym, a, b, probs, rules)
                coin_r[sym] = r
                rows.append({"phase": phase, "strategy": name, "coin": sym.split("/")[0], **summarise(r)})
                quality += [{"phase": phase, "strategy": name, **q} for q in exit_quality(trades, sym)]
            basket = pd.concat(coin_r, axis=1).fillna(0).mean(axis=1)
            rows.append({"phase": phase, "strategy": name, "coin": "basket", **summarise(basket)})
    res = pd.DataFrame(rows)
    fmt = {"final $": "${:,.0f}".format, "return": "{:+.1%}".format, "Sharpe": "{:+.2f}".format,
           "Sharpe SE": "{:.2f}".format, "worst drop": "{:+.1%}".format}
    for phase in ("choose 2022-2024", "check 2025-2026"):
        print(f"\n=== {phase}: basket of 4 coins, $1000 each ===")
        print(res[(res.phase == phase) & (res.coin == "basket")].drop(columns=["phase", "coin"])
              .to_string(index=False, formatters=fmt))

    q = pd.DataFrame(quality)
    if len(q):
        q["dodged a fall"] = q["after_3d"] < 0
        print("\n=== Exit sales: price 3 days later (negative = the sale dodged a fall) ===")
        print(q.groupby(["phase", "strategy", "kind"]).agg(sales=("after_3d", "size"),
                                                           dodged=("dodged a fall", "mean"),
                                                           avg_3d_after=("after_3d", "mean"))
              .to_string(formatters={"dodged": "{:.0%}".format, "avg_3d_after": "{:+.1%}".format}))

    bk = res[res.coin == "basket"].set_index(["phase", "strategy"])
    new = [v for v in VARIANTS if v != "smart"]
    chosen = max(new, key=lambda s: bk.loc[("choose 2022-2024", s), "Sharpe"])
    s_new, s_base = bk.loc[("check 2025-2026", chosen), "Sharpe"], bk.loc[("check 2025-2026", "smart"), "Sharpe"]
    print(f"\nChosen on 2022-2024: {chosen}")
    print(f"Check 2025-2026: Sharpe {s_new:+.2f} vs smart {s_base:+.2f} -> {'ADOPT' if s_new > s_base else 'not adopted'}")
    res.to_csv(ROOT / "reports" / "exit_timing.csv", index=False)
    pickle.dump({"results": res, "exit_quality": q, "chosen": chosen}, open(ROOT / "reports" / "exit_timing_full.pkl", "wb"))


if __name__ == "__main__":
    main()
