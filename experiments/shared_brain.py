"""One shared AI for crypto and stocks: does learning from both markets predict better than separate AIs?

    .venv\\Scripts\\python experiments\\shared_brain.py

Fixed before any result was seen (2026-10-06). Daily candles: BTC/ETH/BNB/SOL (Binance, from 2019) and the 15
Binance-listed US stocks (Yahoo, adjusted, 10 years). Target: price higher 5 days later (5 trading days for stocks,
5 days for coins). Features: features.build on each daily chart, plus its market's context with shared column names
(mkt_*: Bitcoin for coins, the S&P 500 ETF for stocks), strength vs that market over 5 and 20 days, rank within its
own market, and an is_crypto flag. Same model settings as the live models (model._new_model).
- separate  one model per market, trained only on that market's rows
- shared    one model trained on both markets' rows
Walk-forward: retrained at each month start on all rows whose 5-day outcome was known (7-day embargo).
Adopted only if shared beats separate in AUC by more than 0.003 (the placebo noise level) for BOTH markets in BOTH
2020-2022 and 2023-2026.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config, data, features, model, stocks  # noqa: E402

H = 5
STOCK_LIST = [t for t in stocks.STOCKS if t not in stocks.ETFS]
PERIODS = {"2020-2022": ("2020-01-01", "2023-01-01"), "2023-2026": ("2023-01-01", "2100-01-01")}
NOISE = 0.003


def market_rows(raw, market_key, is_crypto):
    """Feature rows for every asset in `raw` ({name: daily candles}) with `market_key` as market context."""
    own = {k: features.build(v) for k, v in raw.items()}
    mkt = own[market_key][features.MARKET_COLS].add_prefix("mkt_")
    rank = pd.DataFrame({k: v["close"].pct_change(20) for k, v in raw.items()}).rank(axis=1, pct=True)
    out = []
    for k, f in own.items():
        if k == market_key and not is_crypto:
            continue  # SPY is context only, not a stock to predict
        f = f.join(mkt)
        f["rel_mkt_20"] = f["ret_24"] - f["mkt_ret_24"]
        f["rel_mkt_5"] = f["ret_6"] - f["mkt_ret_6"]
        f["xs_rank_20"] = rank[k]
        c = raw[k]["close"]
        fwd = c.shift(-H) / c - 1
        f["y"] = (fwd > 0).astype(float).where(fwd.notna())
        f["is_crypto"] = float(is_crypto)
        f["asset"] = k
        out.append(f.iloc[200:])
    return pd.concat(out)


def dataset():
    crypto = {s: data.drop_open_candle(data.load_cached(s, "1d"), "1d")[["open", "high", "low", "close", "volume"]]
              for s in config.SYMBOLS}
    stock = {t: stocks.candles(t) for t in [*STOCK_LIST, "SPY"]}
    ds = pd.concat([market_rows(crypto, "BTC/USDT", True), market_rows(stock, "SPY", False)]).sort_index()
    return ds.drop(columns=["hour"], errors="ignore")


def walk(ds, train_mask_fn, cols):
    parts = []
    months = pd.date_range(pd.Timestamp("2020-01-01", tz="UTC"), ds.index.max() + pd.Timedelta("1D"), freq="MS")
    for m0, m1 in zip(months, list(months[1:]) + [ds.index.max() + pd.Timedelta("1D")]):
        train = ds[(ds.index < m0 - pd.Timedelta("7D")) & train_mask_fn(ds)].dropna(subset=["y"])
        test = ds[(ds.index >= m0) & (ds.index < m1)].dropna(subset=["y"])
        if test.empty or len(train) < 1000:
            continue
        m = model._new_model().fit(train[cols], train["y"])
        parts.append(test[["asset", "is_crypto", "y"]].assign(p=m.predict_proba(test[cols])[:, 1]))
    return pd.concat(parts)


def main():
    ds = dataset()
    cols = [c for c in ds.columns if c not in ("y", "asset")]
    shared = walk(ds, lambda d: np.ones(len(d), bool), cols)
    sep = pd.concat([walk(ds, lambda d: (d["is_crypto"] == 1.0).to_numpy(), cols).query("is_crypto == 1.0"),
                     walk(ds, lambda d: (d["is_crypto"] == 0.0).to_numpy(), cols).query("is_crypto == 0.0")])
    rows, ok = [], True
    for market, flag in (("crypto", 1.0), ("stocks", 0.0)):
        for label, (a, b) in PERIODS.items():
            a, b = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
            s = shared[(shared["is_crypto"] == flag) & (shared.index >= a) & (shared.index < b)]
            p = sep[(sep["is_crypto"] == flag) & (sep.index >= a) & (sep.index < b)]
            auc_s, auc_p = roc_auc_score(s["y"], s["p"]), roc_auc_score(p["y"], p["p"])
            gain = auc_s - auc_p
            ok &= gain > NOISE
            rows.append({"market": market, "period": label, "separate AUC": auc_p, "shared AUC": auc_s,
                         "gain": gain, "rows": len(s)})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nShared brain beats separate AIs by more than {NOISE} in both markets and periods: {ok} -> "
          f"{'ADOPT' if ok else 'not adopted'}")
    res.to_csv(ROOT / "reports" / "shared_brain.csv", index=False)


if __name__ == "__main__":
    main()
