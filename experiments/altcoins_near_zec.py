"""Can the AI learn NEAR and ZEC, two high-volatility altcoins? Should they join the live AI?

    .venv\\Scripts\\python experiments\\altcoins_near_zec.py

Fixed before any result was seen (2026-10-06). Same three models and features as the live AI (config.MODELS),
retrained at the start of every month on all rows whose outcome was already known (no look-ahead).
Features are built with all six coins together (BTC context and the 24h cross-coin rank cover all six).
- pool "majors"  trained on BTC, ETH, BNB, SOL only, then asked about NEAR and ZEC (does what it knows transfer?)
- pool "all6"    NEAR and ZEC added to the training rows
Trading on NEAR and ZEC from $1000 each, 0.1% fee per side, compared on a basket (average daily return of both):
- smart_majors / smart_all6   the Smart strategy (simulate.SMART, unchanged) on that pool's predictions
- sma50                       hold while the daily close is above its 50-day average (benchmark)
- buy_hold                    benchmark
Choice: between the two pools, the better Smart basket Sharpe on 2022-2024. Check: 2025-01-01 to 2026-10-06, once.
NEAR and ZEC are recommended for the live AI only if, on the check period, the chosen Smart beats sma50 on
Sharpe AND the next-1-day model's AUC on NEAR/ZEC is at least config.MIN_AUC. Adding them to the training pool
is also reported for its effect on the four majors' AUC (noise level ~0.003, from the moon-phase placebo).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config, data, features, metrics, model, simulate  # noqa: E402

MAJORS = list(config.SYMBOLS)
ALTS = ["NEAR/USDT", "ZEC/USDT"]
ALL = MAJORS + ALTS
POOLS = {"majors": MAJORS, "all6": ALL}
START, END = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2026-10-06", tz="UTC")
CHOOSE, CHECK = (START, pd.Timestamp("2025-01-01", tz="UTC")), (pd.Timestamp("2025-01-01", tz="UTC"), END)
CASH, Q = 1000.0, 365
PRED_FILE = ROOT / "reports" / "predictions_near_zec.pkl"


def dataset(name):
    """model.dataset for all six coins (cache only; download first)."""
    sp = config.MODELS[name]
    raw = data.closed(sp["timeframe"], refresh=False, symbols=ALL)
    bars = None
    if sp["intraday"]:
        since = min(df.index[0] for df in raw.values()) - pd.Timedelta("2D")
        bars = data.intraday(False, since=since, symbols=ALL)
    feats = features.build_all(raw, bars)
    frames = []
    for sym, df in raw.items():
        X = feats[sym]
        X["y"] = features.target(df, sp["horizon"])
        X["fwd_ret_1"] = df["close"].pct_change().shift(-1)
        X["symbol"] = sym
        frames.append(X.iloc[200:])
    return pd.concat(frames).sort_index()


def predictions():
    """{pool: {model: DataFrame(symbol, prob)}}: monthly-retrained out-of-sample P(up) for all six coins."""
    if PRED_FILE.exists():
        return pickle.load(open(PRED_FILE, "rb"))
    out = {p: {} for p in POOLS}
    months = pd.date_range(START, END, freq="MS", tz="UTC")
    for name, sp in config.MODELS.items():
        ds = dataset(name)
        cols = model.feature_cols(ds)
        step = pd.Timedelta(sp["timeframe"])
        for pool, syms in POOLS.items():
            parts = []
            for m0, m1 in zip(months, list(months[1:]) + [END]):
                train = ds[(ds.index <= m0 - (sp["horizon"] + 1) * step) & ds["symbol"].isin(syms)].dropna(subset=["y"])
                test = ds[(ds.index >= m0) & (ds.index < m1)]
                if test.empty:
                    continue
                fitted = model._new_model().fit(train[cols], train["y"])
                parts.append(pd.DataFrame({"symbol": test["symbol"], "y": test["y"],
                                           "prob": fitted.predict_proba(test[cols])[:, 1]}, index=test.index))
            out[pool][name] = pd.concat(parts)
            print(f"  predictions: {name} / {pool} done", flush=True)
    pickle.dump(out, open(PRED_FILE, "wb"))
    return out


def auc_table(pred):
    rows = []
    for pool in POOLS:
        for name in config.MODELS:
            p = pred[pool][name].dropna(subset=["y"])
            for phase, (a, b) in (("2022-2024", CHOOSE), ("2025-2026", CHECK)):
                w = p[(p.index >= a) & (p.index < b)]
                row = {"pool": pool, "model": name, "period": phase}
                for sym in ALL:
                    s = w[w["symbol"] == sym]
                    row[sym.split("/")[0]] = roc_auc_score(s["y"], s["prob"])
                rows.append(row)
    return pd.DataFrame(rows)


def daily_close(sym):
    return data.drop_open_candle(data.load_cached(sym, "1d"), "1d")["close"]


def rule_returns(sym, a, b, rule):
    """Daily net returns of buy & hold or the SMA50 rule, deciding at each daily close, 0.1% fee per side."""
    c = daily_close(sym)
    pos = pd.Series(1.0, index=c.index) if rule == "buy_hold" else (c > c.rolling(50).mean()).astype(float)
    r_next = c.shift(-1) / c - 1
    pos, r_next = pos[(pos.index >= a) & (pos.index < b)], r_next[(r_next.index >= a) & (r_next.index < b)]
    change = pos.diff().abs()
    change.iloc[0] = pos.iloc[0]
    net = pos * r_next - change * config.FEE
    net.iloc[-1] -= pos.iloc[-1] * config.FEE  # sell at the end, as the Smart runs do
    net.index = net.index + pd.Timedelta("1D")  # the return is earned by the next close
    return net.dropna()


def smart_returns(sym, a, b, probs):
    p = simulate._smart_inputs(probs, sym, a, b)
    p = p[p.index <= b - 2 * pd.Timedelta("4h")]
    candles = data.closed("4h", refresh=False, symbols=[sym])[sym]
    res = simulate._smart_trade(p, candles, CASH, config.FEE, simulate.SMART)
    eq = res["equity"].resample("1D").last().ffill()
    return eq.pct_change().dropna(), res


def summarise(r):
    eq = CASH * (1 + r).cumprod()
    return {"final $": eq.iloc[-1], "return": eq.iloc[-1] / CASH - 1, "Sharpe": metrics.sharpe(r, Q),
            "Sharpe SE": metrics.sharpe_se(r, Q), "worst drop": metrics.max_drawdown(eq)}


def main():
    pred = predictions()
    auc = auc_table(pred)
    pd.set_option("display.width", 200)
    print("\n=== Walk-forward AUC (0.50 = coin flip) ===")
    print(auc.to_string(index=False, float_format="{:.3f}".format))

    rows, trades = [], {}
    for phase, (a, b) in (("choose 2022-2024", CHOOSE), ("check 2025-2026", CHECK)):
        per = {}
        for sym in ALTS:
            for pool in POOLS:
                probs = {n: pred[pool][n][["symbol", "prob"]] for n in config.MODELS}
                r, res = smart_returns(sym, a, b, probs)
                per[(f"smart_{pool}", sym)] = r
                trades[(phase, pool, sym)] = res["trades"]
            for rule in ("sma50", "buy_hold"):
                per[(rule, sym)] = rule_returns(sym, a, b, rule)
        for strat in ("smart_majors", "smart_all6", "sma50", "buy_hold"):
            coin_r = {sym: per[(strat, sym)] for sym in ALTS}
            for sym, r in coin_r.items():
                rows.append({"phase": phase, "strategy": strat, "coin": sym.split("/")[0], **summarise(r)})
            basket = pd.concat(coin_r, axis=1).fillna(0).mean(axis=1)
            rows.append({"phase": phase, "strategy": strat, "coin": "basket", **summarise(basket)})
    res = pd.DataFrame(rows)
    fmt = {"final $": "${:,.0f}".format, "return": "{:+.1%}".format, "Sharpe": "{:+.2f}".format,
           "Sharpe SE": "{:.2f}".format, "worst drop": "{:+.1%}".format}
    for phase in ("choose 2022-2024", "check 2025-2026"):
        print(f"\n=== {phase}: $1000 per coin ===")
        print(res[res.phase == phase].drop(columns="phase").to_string(index=False, formatters=fmt))

    bk = res[res.coin == "basket"].set_index(["phase", "strategy"])
    chosen = max(("smart_majors", "smart_all6"), key=lambda s: bk.loc[("choose 2022-2024", s), "Sharpe"])
    s_ai, s_trend = bk.loc[("check 2025-2026", chosen), "Sharpe"], bk.loc[("check 2025-2026", "sma50"), "Sharpe"]
    pool = chosen.removeprefix("smart_")
    a = auc[(auc.pool == pool) & (auc.model == "4h") & (auc.period == "2025-2026")][["NEAR", "ZEC"]].mean(axis=1).iloc[0]
    print(f"\nChosen on 2022-2024: {chosen}")
    print(f"Check 2025-2026: Sharpe {s_ai:+.2f} vs sma50 {s_trend:+.2f}; next-1-day AUC on NEAR/ZEC {a:.3f}")
    print("-> " + ("RECOMMEND adding NEAR and ZEC to the live AI" if s_ai > s_trend and a >= config.MIN_AUC
                   else "do NOT add them to the live AI"))
    maj = auc[auc.period == "2025-2026"].set_index(["pool", "model"])[[s.split("/")[0] for s in MAJORS]].mean(axis=1)
    for name in config.MODELS:
        print(f"Majors' AUC 2025-2026, {name}: trained on majors {maj[('majors', name)]:.3f}, "
              f"with NEAR/ZEC added {maj[('all6', name)]:.3f} ({maj[('all6', name)] - maj[('majors', name)]:+.3f})")
    res.to_csv(ROOT / "reports" / "altcoins_near_zec.csv", index=False)
    auc.to_csv(ROOT / "reports" / "altcoins_near_zec_auc.csv", index=False)
    pickle.dump({"results": res, "auc": auc, "trades": trades, "chosen": chosen}, open(ROOT / "reports" / "altcoins_near_zec_full.pkl", "wb"))


if __name__ == "__main__":
    main()
