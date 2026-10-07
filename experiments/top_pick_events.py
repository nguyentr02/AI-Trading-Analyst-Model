"""Should the AI's top stock pick change when the chart turns against it, instead of only at the monthly review?

    .venv\\Scripts\\python experiments\\top_pick_events.py

Fixed before any result was seen (2026-10-07). The top pick (experiments/stock_top_pick.py) is the steadiest stock
(lowest past-year volatility) among the 15 stocks above their 200-day average, held alone. Variants:
- monthly     re-pick at each month end (current rule)
- weekly      re-pick at each week's last close
- daily       re-pick at every close (added 2026-10-07 at the user's request, before its result was seen)
- break50     monthly, but if the pick closes below its 50-day average, switch at once to the next steadiest stock
              that is above its 200-day AND 50-day averages
- drop8       monthly, but if the pick closes 8%+ below the price it was bought at, switch at once to the next
              steadiest stock above its 200-day average
Daily closes (Yahoo, adjusted), decisions at the close, applied from the next day, 0.1% cost per side.
Adopted only if a variant beats monthly on Sharpe in BOTH 2016-2022 and 2023-2026.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from cryptoai import stocks  # noqa: E402
import stock_hold_picks as H  # noqa: E402


def run(closes, variant):
    r = np.log(closes).diff()
    vol = r.rolling(252).std()
    ma200, ma50 = closes.rolling(200).mean(), closes.rolling(50).mean()
    month_end = closes.index.to_series().groupby(closes.index.to_period("M")).transform("max") == closes.index
    week_end = closes.index.to_series().groupby(closes.index.to_period("W")).transform("max") == closes.index
    held, entry, out = None, None, []
    for i, t in enumerate(closes.index):
        if i < 260:
            out.append(None)
            continue
        px, v = closes.loc[t], vol.loc[t]
        up = (px > ma200.loc[t]) & v.notna()

        def pick(extra=None, exclude=None):
            ok = up if extra is None else up & extra
            cands = v[ok.fillna(False)]
            if exclude is not None:
                cands = cands.drop(exclude, errors="ignore")
            return cands.idxmin() if len(cands) else None

        review = True if variant == "daily" else (month_end.loc[t] if variant != "weekly" else week_end.loc[t])
        new = held
        if review or held is None:
            new = pick()
        elif variant == "break50" and px[held] < ma50.loc[t, held]:
            new = pick(extra=px > ma50.loc[t], exclude=held)
        elif variant == "drop8" and px[held] <= entry * 0.92:
            new = pick(exclude=held)
        if new != held:
            held, entry = new, (px[new] if new else None)
        out.append(held)
    picks = pd.Series(out, index=closes.index)
    w = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for t, k in picks.items():
        if k:
            w.loc[t, k] = 1.0
    real = picks.dropna()
    return H.returns(closes, w), int((real != real.shift()).sum() - 1)


def main():
    closes = pd.DataFrame({t: stocks.history(t) for t in H.PICKS}).sort_index()
    rows = []
    for variant in ("monthly", "daily", "weekly", "break50", "drop8"):
        r, switches = run(closes, variant)
        r = r[r.index >= closes.index[261]]
        for label, (a, b) in H.PERIODS.items():
            x = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]
            rows.append({"variant": variant, "period": label, "switches (all years)": int(switches), **H.stats(x)})
    res = pd.DataFrame(rows)
    fmt = {"return": "{:+.0%}".format, "Sharpe": "{:.2f}".format, "worst fall": "{:+.0%}".format}
    for label in H.PERIODS:
        print(f"\n=== {label} ===")
        print(res[res.period == label].drop(columns="period").set_index("variant").to_string(formatters=fmt))
    s = res.set_index(["variant", "period"])["Sharpe"]
    for v in ("daily", "weekly", "break50", "drop8"):
        ok = all(s[(v, p)] > s[("monthly", p)] for p in H.PERIODS)
        print(f"{v}: {'ADOPT' if ok else 'not adopted'}")
    res.to_csv(ROOT / "reports" / "top_pick_events.csv", index=False)


if __name__ == "__main__":
    main()
