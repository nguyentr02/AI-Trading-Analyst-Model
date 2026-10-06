"""Market-stress filters on top of the stock buy-for-hold list: hold less when fear gauges flash red?

    .venv\\Scripts\\python experiments\\stock_regime.py

Fixed before any result was seen (2026-10-06), from the stock research summary of that day (VIX term structure and
credit spreads as risk filters, Goyal-Welch-Zafirov 2024: use them for exposure, not return forecasts).
Base: the adopted buy-for-hold list (equal weight across the 15 stocks above their 200-day average, monthly;
experiments/stock_hold_picks.py). Overlays decided at each daily close from Yahoo data, applied from the next day,
0.1% cost per side on every change:
- vix_ts         exposure 50% while VIX / VIX3M > 1 (short-term fear above 3-month fear: stress), else 100%
- credit         exposure 50% while the high-yield vs Treasury bond ratio (HYG / IEF) fell more than 2% over 21 days
- vix_credit     0% when both are on, 50% when one is, 100% otherwise
Choice: best Sharpe on 2016-2022 among the three. Check on 2023-2026, once. Adopted only if it beats the plain list
on Sharpe in BOTH periods.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from cryptoai import stocks  # noqa: E402
import stock_hold_picks as H  # noqa: E402


def yahoo_close(ticker):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}", params={"range": "10y", "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]
    s = pd.Series(j["indicators"]["quote"][0]["close"], index=pd.to_datetime(j["timestamp"], unit="s", utc=True).normalize())
    return s[~s.index.duplicated(keep="last")].dropna()


def main():
    closes = pd.DataFrame({t: stocks.history(t) for t in H.PICKS}).sort_index()
    idx = closes.index
    vix, vix3m = yahoo_close("^VIX").reindex(idx).ffill(), yahoo_close("^VIX3M").reindex(idx).ffill()
    credit = (yahoo_close("HYG") / yahoo_close("IEF")).reindex(idx).ffill()
    stress_vix = (vix / vix3m > 1).astype(float)
    stress_credit = (credit / credit.shift(21) - 1 < -0.02).astype(float)
    overlays = {
        "vix_ts": 1 - 0.5 * stress_vix,
        "credit": 1 - 0.5 * stress_credit,
        "vix_credit": 1 - 0.5 * stress_vix - 0.5 * stress_credit,
    }
    base_w = H.weights_monthly(closes, "uptrend")
    series = {"buy list": H.returns(closes, base_w)}
    for name, exposure in overlays.items():
        series[name] = H.returns(closes, base_w.mul(exposure.shift(0).fillna(1.0), axis=0))
    rows = []
    for name, r in series.items():
        for label, (a, b) in H.PERIODS.items():
            w = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]
            w = w[w.index >= r[r != 0].index.min()]
            rows.append({"strategy": name, "period": label, **H.stats(w)})
    res = pd.DataFrame(rows)
    fmt = {"return": "{:+.0%}".format, "Sharpe": "{:.2f}".format, "worst fall": "{:+.0%}".format}
    for label in H.PERIODS:
        print(f"\n=== {label} ===")
        print(res[res.period == label].drop(columns="period").set_index("strategy").to_string(formatters=fmt))
    print(f"\nShare of days in stress: VIX {stress_vix.mean():.0%}, credit {stress_credit.mean():.0%}")
    s = res.set_index(["strategy", "period"])["Sharpe"]
    chosen = max(overlays, key=lambda k: s[(k, "2016-2022")])
    ok = all(s[(chosen, p)] > s[("buy list", p)] for p in H.PERIODS)
    print(f"Chosen on 2016-2022: {chosen} -> {'ADOPT' if ok else 'not adopted'}")
    res.to_csv(ROOT / "reports" / "stock_regime.csv", index=False)


if __name__ == "__main__":
    main()
