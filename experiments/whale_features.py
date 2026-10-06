"""Does whale and leverage positioning help the AI predict direction or sharp drops?

    .venv\\Scripts\\python experiments\\whale_features.py

Fixed before any result was seen (2026-10-06). Data: Binance's public USDT-M futures metrics (5-minute open interest
and long/short ratios, data.binance.vision; BTC from 2020-09, ETH/BNB/SOL from 2021-12) and our spot candles.
"Whale" bundle, per coin, as known at each 4h close:
- oi_chg_4h, oi_chg_24h   change in open interest value (log), i.e. leverage being added or removed
- oi_z30d                 open interest vs its last 30 days (z-score of the log)
- top_pos_ratio           top traders' long/short ratio by position size (log), and its 24h change
- top_acc_ratio           top traders' long/short ratio by number of accounts (log)
- crowd_ratio             all accounts' long/short ratio (log)
- top_vs_crowd            top traders' position ratio minus the crowd's ratio (smart money vs retail)
- avg_trade_z             spot average trade size (quote volume / trades) vs its last 30 days: big players active
Taker buy/sell ratios are left out (tested and rejected on 2026-10-05).
Models: the "Next 1 day" direction model (target: price higher 6 candles later) and the drop warning
(cryptoai/droprisk.py), each with and without the bundle. Both versions train only on rows from 2021-12-01, when
all four coins have the data, and are retrained monthly; evaluation on 2022-07 to 2024-12 and 2025-01 to 2026-10.
Adopted only if the AUC gain is positive in BOTH periods, at least 0.003 on 2025-2026, and positive in at least
95% of block-bootstrap resamples of months on 2025-2026.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from cryptoai import config, data, droprisk, model  # noqa: E402
from drop_features import as_of  # noqa: E402

TRAIN_FROM = pd.Timestamp("2021-12-01", tz="UTC")
START, SPLIT, END = (pd.Timestamp(x, tz="UTC") for x in ("2022-07-01", "2025-01-01", "2026-10-06"))
STEP = pd.Timedelta("4h")
OUT = ROOT / "reports" / "predictions_whale_features.pkl"
WHALE = ["oi_chg_4h", "oi_chg_24h", "oi_z30d", "top_pos_ratio", "top_pos_chg24h", "top_acc_ratio", "crowd_ratio",
         "top_vs_crowd", "avg_trade_z"]


def whale_features(sym, close_times):
    m = pd.read_csv(ROOT / "data" / f"futures_metrics_{sym.replace('/', '')}.csv", parse_dates=["create_time"])
    m = m.set_index(pd.to_datetime(m["create_time"], utc=True)).sort_index()
    oi = np.log(m["sum_open_interest_value"].where(m["sum_open_interest_value"] > 0))
    top_pos = np.log(m["sum_toptrader_long_short_ratio"].where(m["sum_toptrader_long_short_ratio"] > 0))
    top_acc = np.log(m["count_toptrader_long_short_ratio"].where(m["count_toptrader_long_short_ratio"] > 0))
    crowd = np.log(m["count_long_short_ratio"].where(m["count_long_short_ratio"] > 0))
    day = pd.Timedelta("1D")
    now_oi, now_top = as_of(oi, close_times), as_of(top_pos, close_times)
    f = pd.DataFrame({
        "oi_chg_4h": now_oi - as_of(oi, close_times - STEP).to_numpy(),
        "oi_chg_24h": now_oi - as_of(oi, close_times - day).to_numpy(),
        "oi_z30d": as_of((oi - oi.rolling("30D").mean()) / oi.rolling("30D").std(), close_times),
        "top_pos_ratio": now_top,
        "top_pos_chg24h": now_top - as_of(top_pos, close_times - day).to_numpy(),
        "top_acc_ratio": as_of(top_acc, close_times),
        "crowd_ratio": as_of(crowd, close_times),
        "top_vs_crowd": as_of(top_pos - crowd, close_times),
    })
    c = data.closed("4h", refresh=False, symbols=[sym])[sym]
    avg = np.log(c["quote_volume"] / c["trades"].where(c["trades"] > 0))
    z = (avg - avg.rolling(180).mean()) / avg.rolling(180).std()
    z.index = z.index + STEP  # known at the candle's close
    f["avg_trade_z"] = as_of(z, close_times)
    return f


def with_whales(ds):
    ds = ds.copy()
    for col in WHALE:
        ds[col] = np.nan
    for sym in config.SYMBOLS:
        mask = (ds["symbol"] == sym).to_numpy()
        f = whale_features(sym, ds.index[mask] + STEP)
        for col in WHALE:
            ds.loc[mask, col] = f[col].to_numpy()
    return ds


def walk(ds, cols, embargo):
    parts = []
    months = pd.date_range(START, END, freq="MS", tz="UTC")
    for m0, m1 in zip(months, list(months[1:]) + [END]):
        train = ds[(ds.index >= TRAIN_FROM) & (ds.index <= m0 - (embargo + 1) * STEP)].dropna(subset=["y"])
        test = ds[(ds.index >= m0) & (ds.index < m1)].dropna(subset=["y"])
        if test.empty:
            continue
        fitted = model._new_model().fit(train[cols], train["y"])
        parts.append(pd.DataFrame({"y": test["y"], "prob": fitted.predict_proba(test[cols])[:, 1]}, index=test.index))
    return pd.concat(parts)


def predictions():
    if OUT.exists():
        return pickle.load(open(OUT, "rb"))
    out = {}
    for target, ds, embargo in (("direction (next 1 day)", model.dataset("4h", refresh=False), 6),
                                ("drop warning", droprisk.dataset(refresh=False), droprisk.HORIZON)):
        base = model.feature_cols(ds)
        ds = with_whales(ds)
        out[target] = {"current": walk(ds, base, embargo), "+ whales": walk(ds, base + WHALE, embargo)}
        print(f"  {target}: done", flush=True)
    pickle.dump(out, open(OUT, "wb"))
    return out


def main():
    pred = predictions()
    rng = np.random.default_rng(0)
    for target, p in pred.items():
        a, b = p["current"], p["+ whales"].reindex(p["current"].index)
        gains = {}
        print(f"\n=== {target} ===")
        for phase, mask in (("2022-07 to 2024", a.index < SPLIT), ("2025-2026", a.index >= SPLIT)):
            auc_a, auc_b = roc_auc_score(a["y"][mask], a["prob"][mask]), roc_auc_score(a["y"][mask], b["prob"][mask])
            gains[phase] = auc_b - auc_a
            print(f"{phase:<16} AUC current {auc_a:.4f}   + whales {auc_b:.4f}   gain {auc_b - auc_a:+.4f}")
        late = a.index >= SPLIT
        df = pd.DataFrame({"y": a["y"][late], "a": a["prob"][late], "b": b["prob"][late], "m": a.index[late].strftime("%Y-%m")})
        groups = {m: g for m, g in df.groupby("m")}
        boot = []
        for _ in range(500):
            s = pd.concat([groups[m] for m in rng.choice(list(groups), len(groups))])
            boot.append(roc_auc_score(s["y"], s["b"]) - roc_auc_score(s["y"], s["a"]))
        share = float(np.mean(np.array(boot) > 0))
        ok = all(g > 0 for g in gains.values()) and gains["2025-2026"] >= 0.003 and share >= 0.95
        print(f"2025-2026 gain positive in {share:.0%} of resamples -> {'ADOPT' if ok else 'not adopted'}")


if __name__ == "__main__":
    main()
