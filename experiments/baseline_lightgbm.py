"""Baseline test: can a simple LightGBM on price and volume beat buy & hold and trivial rules after costs?

    .venv\\Scripts\\python experiments\\baseline_lightgbm.py            # development + validation only
    .venv\\Scripts\\python experiments\\baseline_lightgbm.py --final    # ALSO the final holdout, once

Everything below was fixed before any result was seen:
- Daily Binance candles for BTC, ETH, BNB, SOL. One model pooled across the four coins.
- Inputs: price and volume only (returns, volatility, RSI, distance from moving averages, range, volume).
- Target: is tomorrow's close higher than today's? Long-or-cash, decided at each daily close.
- Costs: 0.1% fee + 0.05% slippage per side on every position change.
- Trivial rules: buy & hold; trend (hold when close > 50-day average); momentum (hold when 20-day return > 0).
- Development: walk-forward, retrained each year on all earlier data, tested on 2021, 2022, 2023.
- Validation: 2024, used only to pick the buy threshold from {0.50, 0.52, 0.55} by the basket's Sharpe.
- Final holdout: 2025-01-01 to the last closed day, retrained once at each year start, run ONCE.
  The first --final run writes docs/backTestResult/baseline_holdout_used.json; later runs refuse.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config, data, features, metrics  # noqa: E402

FEE, SLIPPAGE = 0.001, 0.0005
COST = FEE + SLIPPAGE  # per side
Q = 365  # periods per year, daily crypto
THRESHOLDS = (0.50, 0.52, 0.55)
HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")
MARKER = ROOT / "docs" / "backTestResult" / "baseline_holdout_used.json"
STRATEGIES = ["LightGBM", "Trend (close > SMA50)", "Momentum (20-day return > 0)", "Buy & hold"]


def make_model():
    return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=42, verbose=-1)


def build(df):
    """Price and volume features at each daily close, plus the target and the trivial rules' positions."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 5, 10, 20):
        f[f"ret_{n}"] = c.pct_change(n)
    f["vol_10"], f["vol_30"] = r.rolling(10).std(), r.rolling(30).std()
    f["vol_ratio"] = f["vol_10"] / f["vol_30"]
    f["rsi_14"] = features._rsi(c, 14)
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    f["range_pos_20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min())
    f["day_range"] = (h - l) / c
    f["close_location"] = (c - l) / (h - l).replace(0, np.nan)
    f["volume_z_20"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    f["volume_ratio_5_20"] = v.rolling(5).mean() / v.rolling(20).mean()
    f = f.replace([np.inf, -np.inf], np.nan)
    f["fwd_ret"] = c.shift(-1) / c - 1  # tomorrow's return: what a position taken at today's close earns
    f["y"] = (f["fwd_ret"] > 0).astype(float).where(f["fwd_ret"].notna())
    f["pos_trend"] = (c > c.rolling(50).mean()).astype(float)
    f["pos_momentum"] = (c.pct_change(20) > 0).astype(float)
    return f.iloc[200:]  # warm-up for the 200-day average


FEATURES = None


def dataset():
    global FEATURES
    raw = data.closed("1d", refresh=False)
    ds = pd.concat([build(df[["open", "high", "low", "close", "volume"]]).assign(symbol=s) for s, df in raw.items()])
    FEATURES = [c for c in ds.columns if c not in ("fwd_ret", "y", "pos_trend", "pos_momentum", "symbol")]
    return ds.sort_index()


def fit_predict(ds, train_end, test_start, test_end):
    """Train on rows whose label was known before `train_end`, predict P(up) for [test_start, test_end)."""
    train = ds[(ds.index < train_end - pd.Timedelta("1D"))].dropna(subset=["y"])
    test = ds[(ds.index >= test_start) & (ds.index < test_end)].dropna(subset=["fwd_ret"])
    model = make_model().fit(train[FEATURES], train["y"])
    return test.assign(prob=model.predict_proba(test[FEATURES])[:, 1])


def walk_forward(ds, start, end):
    """Retrain at each year start on everything before it; predict that year."""
    years = pd.date_range(start, end, freq="YS", tz="UTC")
    years = years[years < end]
    if len(years) == 0 or years[0] > start:
        years = years.insert(0, start)
    parts = [fit_predict(ds, a, a, b) for a, b in zip(years, list(years[1:]) + [end])]
    return pd.concat(parts)


def strategy_returns(pred, threshold):
    """Daily net returns per coin for every strategy (positions decided at the close, costs on changes)."""
    out = {}
    for sym, g in pred.groupby("symbol"):
        g = g.sort_index()
        positions = {"LightGBM": (g["prob"] > threshold).astype(float), "Trend (close > SMA50)": g["pos_trend"],
                     "Momentum (20-day return > 0)": g["pos_momentum"],
                     "Buy & hold": pd.Series(1.0, index=g.index)}
        for name, pos in positions.items():
            changes = pos.diff().abs().fillna(pos.iloc[0])  # entering on day one costs too
            net = pos * g["fwd_ret"] - changes * COST
            if name == "Buy & hold":
                net.iloc[-1] -= COST  # sell at the end
            out[(name, sym)] = net
    return pd.DataFrame(out)


def summarise(rets):
    """Per strategy: basket (equal weight, rebalanced daily) and per-coin results."""
    rows = []
    for name in STRATEGIES:
        cols = rets[name]
        basket = cols.mean(axis=1, skipna=True)
        for label, r in [("All four (basket)", basket)] + [(s.split("/")[0], cols[s].dropna()) for s in cols]:
            eq = (1 + r).cumprod()
            yrs = len(r) / Q
            rows.append({"strategy": name, "coin": label, "total return": eq.iloc[-1] - 1,
                         "CAGR": eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan,
                         "Sharpe": metrics.sharpe(r, Q), "Sharpe SE": metrics.sharpe_se(r, Q),
                         "chance Sharpe > 0": metrics.psr(r, Q), "worst drop": metrics.max_drawdown(eq)})
    return pd.DataFrame(rows)


def trades_per_year(pred, threshold):
    out = {}
    for sym, g in pred.groupby("symbol"):
        g = g.sort_index()
        for name, pos in (("LightGBM", (g["prob"] > threshold).astype(float)), ("Trend (close > SMA50)", g["pos_trend"]),
                          ("Momentum (20-day return > 0)", g["pos_momentum"])):
            out.setdefault(name, []).append((pos.diff().abs() > 0).sum() / 2 / (len(g) / Q))
    return {k: float(np.mean(v)) for k, v in out.items()}


def show(title, pred, threshold):
    print(f"\n=== {title}: {pred.index.min():%Y-%m-%d} to {pred.index.max():%Y-%m-%d}, threshold {threshold} ===")
    print(f"LightGBM AUC: {roc_auc_score(pred['y'], pred['prob']):.4f}   accuracy: "
          f"{((pred['prob'] > 0.5) == pred['y']).mean():.1%}   up-day rate: {pred['y'].mean():.1%}")
    s = summarise(strategy_returns(pred, threshold))
    b = s[s.coin == "All four (basket)"].set_index("strategy")
    print(b[["total return", "CAGR", "Sharpe", "Sharpe SE", "chance Sharpe > 0", "worst drop"]].to_string(
        formatters={"total return": "{:+.1%}".format, "CAGR": "{:+.1%}".format, "Sharpe": "{:+.2f}".format,
                    "Sharpe SE": "{:.2f}".format, "chance Sharpe > 0": "{:.0%}".format, "worst drop": "{:+.1%}".format}))
    print("round trips per coin per year:", {k: round(v, 1) for k, v in trades_per_year(pred, threshold).items()})
    return s


def main():
    global COST
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="also run the final holdout (only once)")
    args = ap.parse_args()
    ds = dataset()
    last = ds.dropna(subset=["fwd_ret"]).index.max() + pd.Timedelta("1D")
    print(f"{len(ds):,} coin-days, {len(FEATURES)} features: {', '.join(FEATURES)}")
    print(f"costs: {FEE:.2%} fee + {SLIPPAGE:.2%} slippage per side")

    dev = walk_forward(ds, pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
    show("Development (walk-forward 2021-2023)", dev, 0.50)

    val = fit_predict(ds, pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"), HOLDOUT_START)
    sharpe = {t: summarise(strategy_returns(val, t)).query("strategy == 'LightGBM' and coin == 'All four (basket)'")
              ["Sharpe"].iloc[0] for t in THRESHOLDS}
    threshold = max(sharpe, key=sharpe.get)
    print(f"\nValidation 2024, basket Sharpe by threshold: { {t: round(v, 2) for t, v in sharpe.items()} } "
          f"-> chosen {threshold}")
    show("Validation (2024)", val, threshold)

    if not args.final:
        print("\nFinal holdout not run. Use --final once everything above is settled.")
        return
    if MARKER.exists():
        print(f"\nRefusing: the final holdout was already used ({json.loads(MARKER.read_text())['used_at']}). "
              "Re-running it after seeing the result would turn it into another validation set.")
        return
    MARKER.write_text(json.dumps({"used_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "threshold": threshold, "costs_per_side": COST,
                                  "holdout": [str(HOLDOUT_START.date()), str((last - pd.Timedelta('1D')).date())]}, indent=2))
    hold = walk_forward(ds, HOLDOUT_START, last)
    s = show("FINAL HOLDOUT (touched once)", hold, threshold)
    s.to_csv(ROOT / "reports" / "baseline_holdout_results.csv", index=False)
    print("\nPer coin, holdout:")
    print(s.pivot_table(index="coin", columns="strategy", values="total return")[STRATEGIES].to_string(float_format="{:+.1%}".format))
    print("\nFee sensitivity, holdout basket (cost per side -> LightGBM / Trend / Buy & hold total return):")
    for cost in (0.0, 0.00075, 0.0015, 0.003):
        COST = cost
        b = summarise(strategy_returns(hold, threshold)).query("coin == 'All four (basket)'").set_index("strategy")["total return"]
        print(f"  {cost:.3%}: {b['LightGBM']:+.1%} / {b['Trend (close > SMA50)']:+.1%} / {b['Buy & hold']:+.1%}")


if __name__ == "__main__":
    main()
