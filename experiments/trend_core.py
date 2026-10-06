"""The 50-day trend rule as the core, sized by forecast volatility and braked by the drop warning.

    .venv\\Scripts\\python experiments\\trend_core.py

Fixed before any result was seen (2026-10-06). Research (Man AHL 2024, Zarattini et al. 2025, Harvey et al. 2018)
says trend following is the best-supported crypto strategy and volatility sizing mostly cuts tail losses. An
earlier test (trend_ai_strategies.py) applied volatility targeting only to the weaker trend ensemble; here the
core is the 50-day rule itself. Decisions at every 4h close, BTC/ETH/BNB/SOL, equal-weight basket, 0.1% fee +
0.05% slippage per side on every change in position, long or cash only (spot).
- sma50              benchmark: hold while the last daily close is above its 50-day average
- buy_hold           benchmark
- sma50_vt           sma50 x min(1, 60% / forecast annual volatility); position changes under 10 points skipped
- sma50_brake        sma50, but cash while the drop warning is High (P >= 40%), back in once it is under 35%
- sma50_vt_brake     both
- ens_vt_brake       average of the 20/50/100/200-day trend flags, volatility-sized and braked
Volatility forecast: HAR-RV (Corsi 2009) per coin on daily realised variance from 15m returns (log form: today,
last 5 days, last 22 days), refitted at each month start on earlier days only. Drop warning: the walk-forward
predictions from exit_timing.py (monthly retrain, out of sample from 2022).
Choice: best basket Sharpe on 2022-2024 among the four new variants. Check: 2025-01-01 to 2026-10-06, once.
Adopted only if it also beats sma50 on Sharpe on the check period. Worst fall reported alongside.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config, data, metrics  # noqa: E402

COST = config.FEE + 0.0005
START, END = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2026-10-06", tz="UTC")
CHOOSE, CHECK = (START, pd.Timestamp("2025-01-01", tz="UTC")), (pd.Timestamp("2025-01-01", tz="UTC"), END)
TARGET_VOL, BAND, SELL_AT, BACK_AT = 0.60, 0.10, 0.40, 0.35
DROP_PRED = ROOT / "reports" / "predictions_drop_2022_2026.pkl"
STEP = pd.Timedelta("4h")


def har_forecast(sym):
    """Forecast annualised volatility for the next day, known at each daily close (index = day the forecast is for)."""
    m15 = data.load_cached(sym, "15m")["close"]
    r = np.log(m15).diff().dropna()
    rv = (r ** 2).groupby(r.index.floor("1D")).sum()
    rv = rv[rv > 0]
    x = pd.DataFrame({"d": np.log(rv), "w": np.log(rv.rolling(5).mean()), "m": np.log(rv.rolling(22).mean())})
    x["y"] = x["d"].shift(-1)  # tomorrow's log RV
    out = {}
    months = pd.date_range(x.index[0] + pd.Timedelta("120D"), x.index[-1], freq="MS")
    for m0, m1 in zip(months, list(months[1:]) + [x.index[-1] + pd.Timedelta("1D")]):
        train = x[x.index < m0 - pd.Timedelta("1D")].dropna()
        b, *_ = np.linalg.lstsq(np.c_[np.ones(len(train)), train[["d", "w", "m"]]], train["y"], rcond=None)
        resid_var = np.var(train["y"] - np.c_[np.ones(len(train)), train[["d", "w", "m"]]] @ b)
        test = x[(x.index >= m0) & (x.index < m1)].dropna(subset=["d", "w", "m"])
        pred_log = np.c_[np.ones(len(test)), test[["d", "w", "m"]]] @ b
        out.update(dict(zip(test.index + pd.Timedelta("1D"), np.sqrt(np.exp(pred_log + resid_var / 2) * 365))))
    return pd.Series(out).sort_index()


def frame(sym, pdrop):
    c4 = data.closed("4h", refresh=False, symbols=[sym])[sym]["close"]
    d = data.drop_open_candle(data.load_cached(sym, "1d"), "1d")["close"]
    daily = pd.DataFrame({f"above{n}": (d > d.rolling(n).mean()).astype(float) for n in (20, 50, 100, 200)})
    daily["vol"] = har_forecast(sym).reindex(daily.index + pd.Timedelta("1D")).to_numpy()
    daily["known_at"] = daily.index + pd.Timedelta("1D")
    f = pd.DataFrame({"r_next": c4.shift(-1) / c4 - 1, "known_at": c4.index + STEP}, index=c4.index)
    f = pd.merge_asof(f.reset_index().sort_values("known_at"), daily.reset_index(drop=True).sort_values("known_at"),
                      on="known_at", direction="backward").set_index("time")
    f["pdrop"] = pdrop[pdrop["symbol"] == sym]["prob"].reindex(f.index)
    return f[(f.index >= START) & (f.index < END)].dropna(subset=["above200", "vol", "pdrop", "r_next"])


def brake(p):
    """1 = allowed in, 0 = braked: off at P >= SELL_AT, back on once P < BACK_AT."""
    out, on = [], 1.0
    for x in p:
        if x >= SELL_AT:
            on = 0.0
        elif x < BACK_AT:
            on = 1.0
        out.append(on)
    return pd.Series(out, index=p.index)


def banded(target):
    """Skip position changes smaller than BAND (except going fully in or out)."""
    out, cur = [], 0.0
    for t in target:
        if abs(t - cur) >= BAND or t in (0.0, 1.0):
            cur = t
        out.append(cur)
    return pd.Series(out, index=target.index)


def positions(f):
    vt = (TARGET_VOL / f["vol"]).clip(upper=1.0)
    br = brake(f["pdrop"])
    ens = f[["above20", "above50", "above100", "above200"]].mean(axis=1)
    return pd.DataFrame({
        "buy_hold": 1.0, "sma50": f["above50"],
        "sma50_vt": banded(f["above50"] * vt), "sma50_brake": f["above50"] * br,
        "sma50_vt_brake": banded(f["above50"] * vt * br), "ens_vt_brake": banded(ens * vt * br),
    }, index=f.index)


def net_returns(pos, r_next):
    change = pos.diff().abs()
    change.iloc[0] = pos.iloc[0]
    return pos * r_next - change * COST


def summarise(r4):
    r = (1 + r4).groupby(r4.index.floor("1D")).prod() - 1  # daily returns
    eq = (1 + r).cumprod()
    return {"return": eq.iloc[-1] - 1, "Sharpe": metrics.sharpe(r, 365), "Sharpe SE": metrics.sharpe_se(r, 365),
            "worst drop": metrics.max_drawdown(eq)}


def main():
    pdrop = pickle.load(open(DROP_PRED, "rb"))
    coin_rets, turnover, invested = {}, {}, {}
    for sym in config.SYMBOLS:
        f = frame(sym, pdrop)
        pos = positions(f)
        coin_rets[sym] = pos.apply(lambda col: net_returns(col, f["r_next"]))
        turnover[sym] = pos.diff().abs().sum() / ((f.index[-1] - f.index[0]).days / 365)
        invested[sym] = pos.mean()
    names = list(next(iter(coin_rets.values())).columns)
    basket = {n: pd.concat({s: r[n] for s, r in coin_rets.items()}, axis=1).mean(axis=1) for n in names}
    rows = []
    for n in names:
        for phase, (a, b) in (("choose 2022-2024", CHOOSE), ("check 2025-2026", CHECK)):
            w = basket[n][(basket[n].index >= a) & (basket[n].index < b)]
            rows.append({"strategy": n, "phase": phase, **summarise(w)})
    r = pd.DataFrame(rows)
    extra = pd.DataFrame({"turnover / year": pd.DataFrame(turnover).mean(axis=1),
                          "time invested": pd.DataFrame(invested).mean(axis=1)})
    fmt = {"return": "{:+.1%}".format, "Sharpe": "{:+.2f}".format, "Sharpe SE": "{:.2f}".format,
           "worst drop": "{:+.1%}".format, "turnover / year": "{:.1f}x".format, "time invested": "{:.0%}".format}
    for phase in ("choose 2022-2024", "check 2025-2026"):
        print(f"\n=== {phase} (basket of 4 coins) ===")
        print(r[r.phase == phase].drop(columns="phase").set_index("strategy").join(extra).to_string(formatters=fmt))
    new = [n for n in names if n not in ("buy_hold", "sma50")]
    chk = r[r.phase == "check 2025-2026"].set_index("strategy")
    cho = r[r.phase == "choose 2022-2024"].set_index("strategy")
    chosen = cho.loc[new, "Sharpe"].idxmax()
    better = chk.loc[chosen, "Sharpe"] > chk.loc["sma50", "Sharpe"]
    print(f"\nChosen on 2022-2024: {chosen}")
    print(f"Check 2025-2026: Sharpe {chk.loc[chosen, 'Sharpe']:+.2f} vs sma50 {chk.loc['sma50', 'Sharpe']:+.2f}; worst drop "
          f"{chk.loc[chosen, 'worst drop']:+.1%} vs {chk.loc['sma50', 'worst drop']:+.1%} -> {'ADOPT' if better else 'not adopted'}")
    r.to_csv(ROOT / "reports" / "trend_core.csv", index=False)


if __name__ == "__main__":
    main()
