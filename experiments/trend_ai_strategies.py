"""Trend rules, volatility targeting, and an AI that learns from them: which strategy should the AI trade?

    .venv\\Scripts\\python experiments\\trend_ai_strategies.py

Fixed before any result was seen (2026-10-06). Daily decisions per coin, long or partly long, equal-weight
basket of BTC, ETH, BNB, SOL. Costs: 0.1% fee + 0.05% slippage per side on every change in position.
- sma50            benchmark: hold when the daily close is above its 50-day average
- trend_ens        average of 4 trend flags (close above its 20/50/100/200-day average): 0, 25, 50, 75 or 100%
- trend_ens_vt40   trend_ens scaled by min(1, 40% / 30-day realised volatility, annualised)
- trend_ens_vt60   same with a 60% volatility target
- trend_ens_ai     trend_ens x AI sizer, sizer = clip(0.5 + 2 x (P(up, next 3 days) - 0.5), 0, 1)
- learned          trend_ens x clip((P - 0.4) / 0.2, 0, 1), where P is a model's P(next 3 days up) trained each
                   year on all earlier years. Its inputs: the trend flags, distances from the averages,
                   volatility, 20/60-day momentum and the AI's own P(up, next 3 days). It learns WHEN the trend
                   signals work, rather than copying them.
- learned_vt40     learned, with the 40% volatility target
Choice: the best basket Sharpe on 2022-2024 among the non-benchmark strategies. Check: 2025-2026, once.
Adopted only if it also beats sma50 on the check period.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config, data, metrics, model, simulate  # noqa: E402

COST = 0.001 + 0.0005
Q = 365
CHOOSE, CHECK = ("2022-01-01", "2025-01-01"), ("2025-01-01", "2026-10-01")
PRED_FILE = ROOT / "reports" / "predictions_1d_2020_2026.pkl"
LEARN_COLS = ["above20", "above50", "above100", "above200", "dist20", "dist50", "dist100", "dist200", "vol30",
              "mom20", "mom60", "p3d"]


def ai_predictions():
    """Walk-forward P(up, next 3 days) from the daily model, retrained monthly, 2020 onwards (cached)."""
    if PRED_FILE.exists():
        return pickle.load(open(PRED_FILE, "rb"))
    pred = simulate.predictions("1d", pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2026-10-06", tz="UTC"))
    pickle.dump(pred, open(PRED_FILE, "wb"))
    return pred


def frame(sym, pred):
    c = data.drop_open_candle(data.load_cached(sym, "1d"), "1d")["close"]
    f = pd.DataFrame(index=c.index)
    for n in (20, 50, 100, 200):
        ma = c.rolling(n).mean()
        f[f"above{n}"] = (c > ma).astype(float)
        f[f"dist{n}"] = c / ma - 1
    r = np.log(c).diff()
    f["vol30"] = r.rolling(30).std() * np.sqrt(Q)
    f["mom20"], f["mom60"] = c.pct_change(20), c.pct_change(60)
    p = pred[pred["symbol"] == sym]["prob"]
    f["p3d"] = p.reindex(f.index)
    f["r_next"] = c.shift(-1) / c - 1          # what a position taken at today's close earns
    f["y3"] = (c.shift(-3) > c).astype(float).where(c.shift(-3).notna())
    f["symbol"] = sym
    return f.dropna(subset=["above200", "vol30", "p3d"])


def learned_probs(df):
    """P(next 3 days up) from a model retrained at each year start on all earlier rows (3-day embargo)."""
    out = []
    for year in range(2021, 2027):
        start, end = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        train = df[df.index < start - pd.Timedelta("3D")].dropna(subset=["y3"])
        test = df[(df.index >= start) & (df.index < end)]
        if len(train) < 300 or test.empty:
            continue
        m = model._new_model().fit(train[LEARN_COLS], train["y3"])
        out.append(pd.Series(m.predict_proba(test[LEARN_COLS])[:, 1], index=test.index))
    return pd.concat(out)


def positions(f):
    ens = f[["above20", "above50", "above100", "above200"]].mean(axis=1)
    vt = lambda target: (target / f["vol30"]).clip(upper=1.0)
    ai = (0.5 + 2 * (f["p3d"] - 0.5)).clip(0, 1)
    learned = ens * ((f["learned_p"] - 0.4) / 0.2).clip(0, 1)
    return pd.DataFrame({
        "buy_hold": 1.0, "sma50": f["above50"], "trend_ens": ens, "trend_ens_vt40": ens * vt(0.40),
        "trend_ens_vt60": ens * vt(0.60), "trend_ens_ai": ens * ai, "learned": learned,
        "learned_vt40": learned * vt(0.40),
    }, index=f.index)


def net_returns(pos, r_next):
    change = pos.diff().abs()
    change.iloc[0] = pos.iloc[0]
    return pos * r_next - change * COST


def summarise(rets, start, end):
    w = rets[(rets.index >= start) & (rets.index < end)]
    eq = (1 + w).cumprod()
    return {"return": eq.iloc[-1] - 1, "Sharpe": metrics.sharpe(w, Q), "Sharpe SE": metrics.sharpe_se(w, Q),
            "worst drop": metrics.max_drawdown(eq)}


def main():
    pred = ai_predictions()
    coin_rets = {}
    for sym in config.SYMBOLS:
        f = frame(sym, pred)
        f["learned_p"] = learned_probs(f).reindex(f.index)
        f = f.dropna(subset=["learned_p"])
        pos = positions(f)
        coin_rets[sym] = pos.apply(lambda col: net_returns(col, f["r_next"])).dropna()
    names = list(next(iter(coin_rets.values())).columns)
    basket = {n: pd.concat({s: r[n] for s, r in coin_rets.items()}, axis=1).mean(axis=1) for n in names}
    rows = []
    for n in names:
        for phase, (a, b) in (("choose 2022-2024", CHOOSE), ("check 2025-2026", CHECK)):
            rows.append({"strategy": n, "phase": phase, **summarise(basket[n], pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"))})
    r = pd.DataFrame(rows)
    fmt = {"return": "{:+.1%}".format, "Sharpe": "{:+.2f}".format, "Sharpe SE": "{:.2f}".format, "worst drop": "{:+.1%}".format}
    for phase in ("choose 2022-2024", "check 2025-2026"):
        print(f"\n=== {phase} (basket of 4 coins) ===")
        print(r[r.phase == phase].drop(columns="phase").set_index("strategy").to_string(formatters=fmt))
    cand = r[(r.phase == "choose 2022-2024") & ~r.strategy.isin(["buy_hold", "sma50"])]
    chosen = cand.sort_values("Sharpe").iloc[-1]["strategy"]
    chk = r[r.phase == "check 2025-2026"].set_index("strategy")
    print(f"\nChosen on 2022-2024: {chosen}")
    better = chk.loc[chosen, "Sharpe"] > chk.loc["sma50", "Sharpe"]
    print(f"Check 2025-2026: {chosen} Sharpe {chk.loc[chosen, 'Sharpe']:+.2f} vs sma50 {chk.loc['sma50', 'Sharpe']:+.2f} "
          f"-> {'ADOPT' if better else 'not adopted'}")
    pd.DataFrame(r).to_csv(ROOT / "reports" / "trend_ai_strategies.csv", index=False)


if __name__ == "__main__":
    main()
