"""Can volatility information make the drop warning sharper?

    .venv\\Scripts\\python experiments\\drop_features.py

Fixed before any result was seen (2026-10-06). The research summary of 2026-10-06 ranked volatility forecasting
(HAR, Corsi 2009) and option-implied volatility (Deribit DVOL) as the most promising inputs for a risk model.
Drop warning as in cryptoai/droprisk.py (triple-barrier label, 4h features of the "Next 1 day" model), retrained
monthly, out of sample from 2022. Feature bundles added on top of the current features:
- har    realised variance from 15m returns over the last 1, 5 and 22 days (log), and 1-day / 22-day ratio
- dvol   Deribit DVOL for BTC and ETH (30-day implied volatility, hourly, from 2021-03): level, 1-day change,
         z-score vs its last 30 days, and BTC implied minus BTC realised 22-day volatility (variance risk premium)
- both   har + dvol
Rows before DVOL exists are left blank (the model handles missing values).
Choice: the bundle with the largest AUC gain over the current model on 2022-2024. Check on 2025-01-01 to
2026-10-06, once: adopted only if its AUC gain there is at least 0.005 and positive in at least 95% of block-bootstrap
resamples of months.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config, data, droprisk, model  # noqa: E402

START, END = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2026-10-06", tz="UTC")
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")
OUT = ROOT / "reports" / "predictions_drop_features.pkl"
STEP = pd.Timedelta("4h")


def as_of(series, times):
    """Last value of `series` (indexed by the time it became known) at or before each time."""
    s = series[~series.index.duplicated()].sort_index().dropna()
    pos = s.index.searchsorted(times, side="right") - 1
    vals = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=times)


def har_features(sym, close_times):
    m = data.load_cached(sym, "15m")["close"]
    r2 = np.log(m).diff() ** 2
    known = r2.index + pd.Timedelta("15min")  # a 15m bar's return is known at its close
    rv = pd.Series(r2.to_numpy(), index=known)
    f = pd.DataFrame({f"rv_{n}d": as_of(np.log(rv.rolling(96 * n).sum()), close_times) for n in (1, 5, 22)})
    f["rv_ratio"] = f["rv_1d"] - f["rv_22d"]  # log ratio
    return f


def dvol_features(close_times):
    out = {}
    for cur in ("BTC", "ETH"):
        d = pd.read_csv(ROOT / "data" / f"dvol_{cur}_1h.csv", index_col="time", parse_dates=["time"])["close"]
        d.index = d.index + pd.Timedelta("1h")  # an hourly bar's close is known at the end of the hour
        lvl = np.log(d)
        out[f"dvol_{cur.lower()}"] = as_of(lvl, close_times)
        out[f"dvol_{cur.lower()}_chg24h"] = as_of(lvl.diff(24), close_times)
        out[f"dvol_{cur.lower()}_z30d"] = as_of((lvl - lvl.rolling(720).mean()) / lvl.rolling(720).std(), close_times)
    f = pd.DataFrame(out)
    m = data.load_cached("BTC/USDT", "15m")["close"]
    rv = pd.Series((np.log(m).diff() ** 2).to_numpy(), index=m.index + pd.Timedelta("15min"))
    btc_rv22 = as_of(np.sqrt(rv.rolling(96 * 22).sum() / 22 * 365), close_times)
    f["vrp_btc"] = np.exp(f["dvol_btc"]) / 100 - btc_rv22
    return f


def build():
    ds = droprisk.dataset(refresh=False)
    base = model.feature_cols(ds)
    extra = {}
    for sym in config.SYMBOLS:
        mask = (ds["symbol"] == sym).to_numpy()
        times = ds.index[mask] + STEP  # features as known at each candle's close
        h = har_features(sym, times)
        v = dvol_features(times)
        part = pd.concat([h, v], axis=1)
        part.index = ds.index[mask]
        extra[sym] = part
    for col in extra[config.SYMBOLS[0]].columns:
        ds[col] = np.nan
        for sym in config.SYMBOLS:
            mask = (ds["symbol"] == sym).to_numpy()
            ds.loc[mask, col] = extra[sym][col].to_numpy()
    har = ["rv_1d", "rv_5d", "rv_22d", "rv_ratio"]
    dvol = [c for c in extra[config.SYMBOLS[0]].columns if c not in har]
    return ds, {"current": base, "har": base + har, "dvol": base + dvol, "both": base + har + dvol}


def predictions():
    if OUT.exists():
        return pickle.load(open(OUT, "rb"))
    ds, bundles = build()
    months = pd.date_range(START, END, freq="MS", tz="UTC")
    out = {}
    for name, cols in bundles.items():
        parts = []
        for m0, m1 in zip(months, list(months[1:]) + [END]):
            train = ds[ds.index <= m0 - (droprisk.HORIZON + 1) * STEP].dropna(subset=["y"])
            test = ds[(ds.index >= m0) & (ds.index < m1)].dropna(subset=["y"])
            if test.empty:
                continue
            fitted = model._new_model().fit(train[cols], train["y"])
            parts.append(pd.DataFrame({"symbol": test["symbol"], "y": test["y"],
                                       "prob": fitted.predict_proba(test[cols])[:, 1]}, index=test.index))
        out[name] = pd.concat(parts)
        print(f"  {name}: done", flush=True)
    pickle.dump(out, open(OUT, "wb"))
    return out


def main():
    pred = predictions()
    rows = []
    for name, p in pred.items():
        for phase, w in (("2022-2024", p[p.index < SPLIT]), ("2025-2026", p[p.index >= SPLIT])):
            rows.append({"bundle": name, "period": phase, "AUC": roc_auc_score(w["y"], w["prob"])})
    auc = pd.DataFrame(rows).pivot(index="bundle", columns="period", values="AUC")
    auc["gain 2022-2024"] = auc["2022-2024"] - auc.loc["current", "2022-2024"]
    auc["gain 2025-2026"] = auc["2025-2026"] - auc.loc["current", "2025-2026"]
    print("\n=== Drop-warning AUC ===")
    print(auc.to_string(float_format="{:+.4f}".format))

    chosen = auc.drop(index="current")["gain 2022-2024"].idxmax()
    cur, new = pred["current"], pred[chosen]
    late = cur.index >= SPLIT
    both = pd.DataFrame({"y": cur["y"][late], "a": cur["prob"][late], "b": new["prob"].reindex(cur.index)[late],
                         "m": cur.index[late].strftime("%Y-%m")}).dropna()
    groups = {m: g for m, g in both.groupby("m")}
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(500):
        s = pd.concat([groups[m] for m in rng.choice(list(groups), len(groups))])
        gains.append(roc_auc_score(s["y"], s["b"]) - roc_auc_score(s["y"], s["a"]))
    gain, share = auc.loc[chosen, "gain 2025-2026"], float(np.mean(np.array(gains) > 0))
    print(f"\nChosen on 2022-2024: {chosen}")
    print(f"Check 2025-2026: AUC gain {gain:+.4f}, positive in {share:.0%} of resamples -> "
          f"{'ADOPT' if gain >= 0.005 and share >= 0.95 else 'not adopted'}")
    auc.to_csv(ROOT / "reports" / "drop_features.csv")


if __name__ == "__main__":
    main()
