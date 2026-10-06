"""The stock AI: chance each US stock is higher 5 trading days from now, and whether trading on it beats holding.

Same model and features as the crypto models (features.build on daily candles), pooled across the stocks in
stocks.STOCKS, with the S&P 500 ETF (SPY) as market context the way Bitcoin is for coins: SPY's own moves, each
stock's strength vs SPY, and its rank among the stocks. Data: 10 years of Yahoo daily candles (Binance's stock
history is only months long).

    python -m cryptoai stock-ai    # retrain, test, save signals and the test results

Test, fixed before any result was seen (2026-10-06): walk-forward predictions retrained at each month start from
2019 on all earlier rows (5-day embargo). Strategies on an equal-weight basket of all the stocks, decisions at each
Monday's close, 0.1% cost per side on every change:
- hold      every stock held all the time
- ai_gate   hold a stock unless its P(up, 5 days) < 45% (the AI as a risk filter, as it works best on crypto)
- ai_tilt   position = clip(0.5 + 4 x (P - 0.5), 0, 1): fully in at 62.5%, half at 50%, out at 37.5%
Choice: the better of ai_gate / ai_tilt by Sharpe on 2019-2022; check on 2023-01-01 to now, once. Trading on the AI
is recommended only if it beats hold on Sharpe in BOTH periods. Results saved to models/stock_ai.json.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, features, model, stocks

HORIZON = 5  # trading days
MARKET = "SPY"
START, SPLIT = pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")
COST = stocks.COST
MODEL_FILE = config.MODEL_DIR / "model_stocks.joblib"
RESULTS = config.MODEL_DIR / "stock_ai.json"
UNUSED = {"hour"}  # daily candles: always 0


def dataset():
    raw = {t: stocks.candles(t) for t in stocks.STOCKS}
    own = {t: features.build(df) for t, df in raw.items()}
    mkt = own[MARKET][features.MARKET_COLS].add_prefix("spy_")
    rank = pd.DataFrame({t: df["close"].pct_change(20) for t, df in raw.items()}).rank(axis=1, pct=True)
    frames = []
    for t, f in own.items():
        f = f.join(mkt)
        f["rel_spy_20"] = f["ret_24"] - f["spy_ret_24"]
        f["rel_spy_5"] = f["ret_6"] - f["spy_ret_6"]
        f["xs_rank_20"] = rank[t]
        c = raw[t]["close"]
        fwd = c.shift(-HORIZON) / c - 1
        f["y"] = (fwd > 0).astype(float).where(fwd.notna())
        f["fwd_ret"] = fwd
        f["r_next"] = c.shift(-1) / c - 1
        f["ticker"] = t
        frames.append(f.iloc[200:])
    return pd.concat(frames).sort_index()


def feature_cols(ds):
    return [c for c in ds.columns if c not in ("y", "fwd_ret", "r_next", "ticker", *UNUSED)]


def walk_forward(ds):
    cols = feature_cols(ds)
    parts = []
    months = pd.date_range(START, ds.index.max() + pd.Timedelta("1D"), freq="MS", tz="UTC")
    for m0, m1 in zip(months, list(months[1:]) + [ds.index.max() + pd.Timedelta("1D")]):
        train = ds[ds.index < m0 - pd.Timedelta(f"{HORIZON + 3}D")].dropna(subset=["y"])
        test = ds[(ds.index >= m0) & (ds.index < m1)]
        if test.empty or len(train) < 2000:
            continue
        m = model._new_model().fit(train[cols], train["y"])
        parts.append(pd.DataFrame({"ticker": test["ticker"], "p": m.predict_proba(test[cols])[:, 1],
                                   "y": test["y"], "r_next": test["r_next"]}, index=test.index))
    return pd.concat(parts)


def _basket(pred, rule):
    """Daily net returns of the equal-weight basket for a position rule, deciding at Monday closes."""
    out = {}
    for t, g in pred.groupby("ticker"):
        g = g.sort_index()
        if rule == "hold":
            pos = pd.Series(1.0, index=g.index)
        else:
            target = (g["p"] >= 0.45).astype(float) if rule == "ai_gate" else (0.5 + 4 * (g["p"] - 0.5)).clip(0, 1)
            pos = target.where(g.index.dayofweek == 0).ffill().fillna(1.0)  # change only on Mondays
        change = pos.diff().abs()
        change.iloc[0] = pos.iloc[0]
        out[t] = pos * g["r_next"] - change * COST
    return pd.DataFrame(out).mean(axis=1).dropna()


def _sharpe(r):
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() else float("nan")


def _summary(r):
    eq = (1 + r).cumprod()
    return {"sharpe": _sharpe(r), "return": float(eq.iloc[-1] - 1), "worst": float((eq / eq.cummax() - 1).min())}


def train_and_test():
    ds = dataset()
    pred = walk_forward(ds)
    lab = pred.dropna(subset=["y"])
    result = {"computed": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "horizon_days": HORIZON,
              "auc": {}, "strategies": {}}
    for label, part in (("2019-2022", lab[lab.index < SPLIT]), ("2023-now", lab[lab.index >= SPLIT])):
        result["auc"][label] = float(roc_auc_score(part["y"], part["p"]))
    for rule in ("hold", "ai_gate", "ai_tilt"):
        r = _basket(pred.dropna(subset=["r_next"]), rule)
        result["strategies"][rule] = {"2019-2022": _summary(r[r.index < SPLIT]), "2023-now": _summary(r[r.index >= SPLIT])}
    s = result["strategies"]
    chosen = max(("ai_gate", "ai_tilt"), key=lambda k: s[k]["2019-2022"]["sharpe"])
    result["chosen"] = chosen
    result["trade_on_ai"] = all(s[chosen][p]["sharpe"] > s["hold"][p]["sharpe"] for p in ("2019-2022", "2023-now"))

    # Final model on all labelled rows, and today's signal per stock.
    cols = feature_cols(ds)
    labelled = ds.dropna(subset=["y"])
    final = model._new_model().fit(labelled[cols], labelled["y"])
    with config.atomic(MODEL_FILE) as tmp:
        joblib.dump({"model": final, "features": cols}, tmp)
    signals = {}
    for t, g in ds.groupby("ticker"):
        last = g.iloc[[-1]]
        signals[t] = {"date": str(last.index[0].date()), "p_up_5d": float(final.predict_proba(last[cols])[:, 1][0])}
    result["signals"] = signals
    with config.atomic(RESULTS) as tmp:
        tmp.write_text(json.dumps(result, indent=2))
    return result


def load():
    try:
        return json.loads(RESULTS.read_text())
    except (OSError, ValueError):
        return None
