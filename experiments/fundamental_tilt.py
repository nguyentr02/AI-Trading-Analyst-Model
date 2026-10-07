"""Slow fundamentals (net issuance, buybacks, profitability) learnt on the S&P 500, applied to our stock buy list.

    .venv\\Scripts\\python experiments\\fundamental_tilt.py     (after experiments/fundamentals_download.py)

Fixed before any result was seen (2026-10-06), from docs/research/stock-signals.md (statistical power needs a wide
universe; net issuance and gross profitability are among the few anomalies that hold in large caps).
Universe: S&P 500 members as of each month end (fja05680/sp500 start/end dates), 2015-2026, with Yahoo prices and SEC
EDGAR facts (current tickers only: delisted names without EDGAR/Yahoo data drop out, a remaining survivorship gap).
Point in time: at each month end d, only facts FILED before d are used; for each (concept, period) the first-filed
value is kept (later restatements ignored).
Features (all from the latest filing available at d):
- iss_12m       log(diluted weighted shares now / a year earlier), both from the same filing (split-consistent);
                negative = shrinking share count
- buyback       (repurchases - share issuance proceeds) / total assets, last annual report
- gpa           gross profit (or revenue - cost of revenue) / total assets, last annual report
- roe           net income / stockholders' equity, last annual report
Target: return over the next 63 trading days minus the universe's average (cross-sectional).
1. Each feature's direction is learnt from its mean monthly rank IC on 2016-2022 only. Composite score = average of
   the direction-adjusted cross-sectional percentile ranks of the four features (missing features ignored).
2. Composite IC per month on the universe; mean and t (overlapping 3-month windows: standard error x sqrt(3)).
3. Use on our 15 stocks: the buy-for-hold list (above the 200-day average, equal weight, monthly) vs the list cut
   to its top half by composite score.
Adopted only if the composite IC is positive with t >= 2 in BOTH 2016-2022 and 2023-2026 AND the tilted list beats
the plain list on Sharpe in both periods.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from cryptoai import stocks  # noqa: E402
import stock_hold_picks as H  # noqa: E402

FUND = ROOT / "data" / "fundamentals"
PERIODS = {"2016-2022": ("2016-01-01", "2023-01-01"), "2023-2026": ("2023-01-01", "2100-01-01")}
FEATURES = ["iss_12m", "buyback", "gpa", "roe"]
H63 = 63


def load_facts(t):
    p = FUND / f"facts_{t}.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    rows = []
    for concept, units in raw.items():
        for unit, vals in units.items():
            for v in vals:
                if "filed" not in v or "end" not in v:
                    continue
                rows.append({"concept": concept.split(":")[1], "start": v.get("start"), "end": v["end"], "val": v["val"],
                             "form": v.get("form", ""), "filed": v["filed"], "accn": v.get("accn")})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["days"] = (df["end"] - df["start"]).dt.days
    return df


def first_filed(df):
    """For each (concept, start, end) keep the value from the earliest filing."""
    return df.sort_values("filed").drop_duplicates(["concept", "start", "end"], keep="first")


def annual(ff, concept, d):
    """Latest annual (about 1-year duration) value of `concept` filed before d, or NaN."""
    x = ff[(ff["concept"] == concept) & (ff["filed"] < d) & ff["days"].between(350, 380)]
    return float(x.sort_values("end").iloc[-1]["val"]) if len(x) else np.nan


def instant(ff, concept, d):
    x = ff[(ff["concept"] == concept) & (ff["filed"] < d) & ff["start"].isna()]
    return float(x.sort_values("end").iloc[-1]["val"]) if len(x) else np.nan


def issuance(df, d):
    """log(shares now / a year earlier) from the latest filing before d that reports both periods."""
    x = df[(df["concept"] == "WeightedAverageNumberOfDilutedSharesOutstanding") & (df["filed"] < d)]
    if x.empty:
        return np.nan
    for accn, g in x.sort_values("filed", ascending=False).groupby("accn", sort=False):
        g = g.dropna(subset=["start"])
        if g.empty:
            continue
        cur = g.sort_values("end").iloc[-1]
        prior = g[(abs((g["end"] - (cur["end"] - pd.Timedelta(days=365))).dt.days) <= 20)
                  & (abs(g["days"] - cur["days"]) <= 20)]
        if len(prior) and cur["val"] > 0 and prior.iloc[0]["val"] > 0:
            return float(np.log(cur["val"] / prior.iloc[0]["val"]))
    return np.nan


def features_at(df, ff, d):
    assets = instant(ff, "Assets", d)
    gp = annual(ff, "GrossProfit", d)
    if np.isnan(gp):
        rev = annual(ff, "Revenues", d)
        rev = annual(ff, "RevenueFromContractWithCustomerExcludingAssessedTax", d) if np.isnan(rev) else rev
        cost = annual(ff, "CostOfRevenue", d)
        cost = annual(ff, "CostOfGoodsAndServicesSold", d) if np.isnan(cost) else cost
        gp = rev - cost
    rep = annual(ff, "PaymentsForRepurchaseOfCommonStock", d)
    iss = annual(ff, "ProceedsFromIssuanceOfCommonStock", d)
    eq = instant(ff, "StockholdersEquity", d)
    ni = annual(ff, "NetIncomeLoss", d)
    return {"iss_12m": issuance(df, d),
            "buyback": (np.nan_to_num(rep) - np.nan_to_num(iss)) / assets if assets and assets > 0 else np.nan,
            "gpa": gp / assets if assets and assets > 0 else np.nan,
            "roe": ni / eq if eq and eq > 0 else np.nan}


def build_panel():
    members = pd.read_csv(FUND / "sp500_ticker_start_end.csv", parse_dates=["start_date", "end_date"])
    tickers = sorted(members[(members["end_date"].isna()) | (members["end_date"] >= "2015-01-01")]["ticker"].unique())
    closes = {}
    for t in tickers:
        p = ROOT / "data" / f"stock_{t.replace('.', '-')}_1d.csv"
        if p.exists():
            closes[t] = pd.read_csv(p, index_col=0, parse_dates=True)["close"]
    px = pd.DataFrame(closes).sort_index()
    month_ends = px.resample("ME").last().index
    month_ends = month_ends[(month_ends >= "2015-06-30")]
    fwd = px.shift(-H63) / px - 1
    rows = []
    for t in tickers:
        df = load_facts(t)
        if df is None or t not in px:
            continue
        ff = first_filed(df)
        spans = members[members["ticker"] == t]
        # Features only change when a new filing arrives: compute them once per filing date (as known the day
        # after it was filed), then each month end takes the latest filing made before it.
        filed = sorted(ff["filed"].dropna().unique())
        known = {pd.Timestamp(f): features_at(df, ff, pd.Timestamp(f) + pd.Timedelta("1D")) for f in filed}
        keys = list(known)
        for m in month_ends:
            day = px.index[px.index <= m][-1]
            d = day.tz_localize(None)
            inside = ((spans["start_date"] <= d) & (spans["end_date"].isna() | (spans["end_date"] > d))).any()
            if not inside or np.isnan(px.at[day, t]):
                continue
            pos = np.searchsorted(keys, d, side="left") - 1  # filed strictly before the month end
            if pos < 0:
                continue
            rows.append({"date": day, "ticker": t, "fwd": fwd.at[day, t], **known[keys[pos]]})
    panel = pd.DataFrame(rows)
    panel["target"] = panel["fwd"] - panel.groupby("date")["fwd"].transform("mean")
    return panel


def monthly_ic(panel, col):
    def one(g):
        g = g.dropna(subset=[col, "target"])
        return spearmanr(g[col], g["target"])[0] if len(g) >= 30 else np.nan
    return panel.groupby("date").apply(one).dropna()


def summary(ic):
    out = {}
    for label, (a, b) in PERIODS.items():
        s = ic[(ic.index >= pd.Timestamp(a, tz="UTC")) & (ic.index < pd.Timestamp(b, tz="UTC"))]
        se = s.std() / np.sqrt(len(s)) * np.sqrt(3) if len(s) > 2 else np.nan
        out[label] = (s.mean(), s.mean() / se if se else np.nan, len(s))
    return out


def main():
    cache = FUND / "panel.pkl"
    panel = pd.read_pickle(cache) if cache.exists() else build_panel()
    panel.to_pickle(cache)
    print(f"panel: {len(panel):,} stock-months, {panel['ticker'].nunique()} stocks, "
          f"{panel['date'].nunique()} month ends; feature coverage "
          + ", ".join(f"{f} {panel[f].notna().mean():.0%}" for f in FEATURES))
    signs = {}
    print("\n=== Rank IC with the next 3 months' relative return (mean, t, months) ===")
    for f in FEATURES:
        s = summary(monthly_ic(panel, f))
        signs[f] = np.sign(s["2016-2022"][0]) or 1.0
        print(f"{f:<9} " + "  ".join(f"{k}: {v[0]:+.3f} (t {v[1]:+.1f}, {v[2]})" for k, v in s.items()))
    ranks = pd.DataFrame({f: panel.groupby("date")[f].rank(pct=True) * signs[f] for f in FEATURES})
    panel["composite"] = ranks.mean(axis=1, skipna=True)
    comp = summary(monthly_ic(panel, "composite"))
    print("composite " + "  ".join(f"{k}: {v[0]:+.3f} (t {v[1]:+.1f})" for k, v in comp.items()))
    print("directions learnt on 2016-2022:", {f: ("higher is better" if s > 0 else "lower is better") for f, s in signs.items()})

    # Use on our 15 stocks: buy list vs the buy list cut to its top half by composite (composite ranked within the 15)
    ours = panel[panel["ticker"].isin(H.PICKS)]
    closes = pd.DataFrame({t: stocks.history(t) for t in H.PICKS}).sort_index()
    base = H.weights_monthly(closes, "uptrend")
    tilt = base.copy()
    for m, g in ours.groupby("date"):
        rows = (tilt.index > m) & (tilt.index <= m + pd.offsets.MonthEnd(1))
        held = [t for t in base.columns if base.loc[rows].iloc[:1][t].sum() > 0] if rows.any() else []
        sc = g.set_index("ticker")["composite"].reindex(held).dropna()
        if len(sc) >= 2:
            keep = list(sc[sc >= sc.median()].index)
            tilt.loc[rows, :] = 0.0
            tilt.loc[rows, keep] = 1.0 / len(keep)
    res = []
    for name, w in (("buy list", base), ("tilted to fundamentals", tilt)):
        r = H.returns(closes, w)
        for label, (a, b) in PERIODS.items():
            x = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]
            x = x[x.index >= r[r != 0].index.min()]
            res.append({"strategy": name, "period": label, **H.stats(x)})
    res = pd.DataFrame(res)
    print("\n=== Our 15 stocks ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    s = res.set_index(["strategy", "period"])["Sharpe"]
    ok_ic = all(comp[p][0] > 0 and comp[p][1] >= 2 for p in PERIODS)
    ok_use = all(s[("tilted to fundamentals", p)] > s[("buy list", p)] for p in PERIODS)
    print(f"\nComposite IC positive with t>=2 in both periods: {ok_ic}; tilt beats the list in both: {ok_use} -> "
          f"{'ADOPT' if ok_ic and ok_use else 'not adopted'}")
    latest = panel[panel["date"] == panel["date"].max()]
    print("\nLatest composite scores for our stocks (higher = stronger fundamentals):")
    print(latest[latest["ticker"].isin(H.PICKS)].sort_values("composite", ascending=False)[
        ["ticker", *FEATURES, "composite"]].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
