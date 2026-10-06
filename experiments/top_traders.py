"""How do the biggest winners and losers on Hyperliquid actually trade?

    .venv\\Scripts\\python experiments\\top_traders.py

Hyperliquid is an on-chain perpetuals exchange: its leaderboard and every account's fills are public.
- Directional traders only: |all-time PnL| / volume >= 0.2% (market makers earn ~0.01% and tell us nothing
  about direction), all-time volume >= $1M.
- Winners: the 100 with the highest all-time PnL. Losers: the 100 with the lowest (largest losses).
- For each, the most recent fills (the API keeps the last 10,000). Big traders split orders into many fills and
  rarely go flat, so most measures use fills (sized by notional) and "close events" (a coin's closing fills in
  the same hour); holding times use the complete round trips (flat -> position -> flat) where there are any.
Measured per trader, then compared between groups (median): trades per day, share of trades in BTC/ETH/SOL,
short share, win rate, average win / average loss, hours held for winners vs losers (disposition effect),
adding to a losing position (averaging down), trading with the daily trend (price vs its 50-day average at
entry), current leverage, and fees as a share of the gross result.
Raw data is cached in reports/hl_traders/ (gitignored). Results are reported per group, not per address.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "reports" / "hl_traders"
CACHE.mkdir(parents=True, exist_ok=True)
INFO = "https://api.hyperliquid.xyz/info"
N, MIN_EDGE, MIN_VOLUME = 100, 0.002, 1e6
MAJORS = {"BTC", "ETH", "SOL"}


def post(body, retries=5):
    for i in range(retries):
        r = requests.post(INFO, json=body, timeout=60)
        if r.status_code == 429:
            time.sleep(5 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("rate limited")


def leaderboard():
    path = CACHE / "leaderboard.json"
    if not path.exists():
        path.write_text(requests.get("https://stats-data.hyperliquid.xyz/Mainnet/leaderboard", timeout=120).text, encoding="utf-8")
    rows = json.loads(path.read_text(encoding="utf-8"))["leaderboardRows"]
    out = []
    for r in rows:
        w = dict(r["windowPerformances"])["allTime"]
        out.append({"address": r["ethAddress"], "account": float(r["accountValue"]), "pnl": float(w["pnl"]),
                    "volume": float(w["vlm"]), "roi": float(w["roi"])})
    df = pd.DataFrame(out)
    df["edge"] = df["pnl"] / df["volume"].where(df["volume"] > 0)
    return df


def fills(address):
    path = CACHE / f"fills_{address}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    got, start = [], 0
    while True:  # oldest first, pages of up to 2000; the API keeps only the last 10,000
        page = post({"type": "userFillsByTime", "user": address, "startTime": start})
        if not page:
            break
        got += page
        if len(page) < 2000 or len(got) >= 10000:
            break
        start = max(f["time"] for f in page) + 1
        time.sleep(1)
    state = post({"type": "clearinghouseState", "user": address})
    data = {"fills": got, "state": state}
    path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(1)
    return data


def daily_closes(coin, start_ms):
    path = CACHE / f"candles_{coin}.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        rows = post({"type": "candleSnapshot", "req": {"coin": coin, "interval": "1d", "startTime": start_ms - 80 * 86400000,
                                                        "endTime": int(time.time() * 1000)}})
        path.write_text(json.dumps(rows), encoding="utf-8")
        time.sleep(0.5)
    if not rows:
        return None
    c = pd.Series([float(r["c"]) for r in rows], index=pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True))
    return c


def round_trips(f):
    """Rebuild positions from fills: one row per round trip (flat -> position -> flat, or a flip)."""
    trips = []
    for coin, g in f.sort_values("time").groupby("coin"):
        known = False  # False until the first time we see this coin flat (earlier positions began before the data)
        avg = pnl = fees = 0.0
        opened, side, adds, adds_losing = None, 0, 0, 0
        for r in g.itertuples():
            pos = float(r.startPosition)  # the exchange's own position before this fill
            q = r.sz if r.side == "B" else -r.sz
            new = pos + q
            tol = 1e-9 * max(abs(pos), abs(q), 1e-9)
            if abs(pos) <= tol:
                known = True
                opened, pnl, fees, side, adds, adds_losing, avg = r.time, 0.0, 0.0, np.sign(q), 0, 0, r.px
            elif not known:
                if abs(new) <= tol or np.sign(new) != np.sign(pos):
                    known = True  # this closes a position opened before the data; a flip starts a fresh one
                    if abs(new) > tol:
                        opened, pnl, fees, side, adds, adds_losing, avg = r.time, 0.0, 0.0, np.sign(new), 0, 0, r.px
                continue
            elif np.sign(q) == np.sign(pos):  # adding to the position
                adds += 1
                adds_losing += (r.px - avg) * np.sign(pos) < 0
                avg = (avg * abs(pos) + r.px * abs(q)) / (abs(pos) + abs(q))
            pnl += r.closedPnl
            fees += r.fee
            if abs(pos) > tol and (abs(new) <= tol or np.sign(new) != np.sign(pos)):
                trips.append({"coin": coin, "side": side, "opened": opened, "closed": r.time, "pnl": pnl, "fees": fees,
                              "entry": avg, "adds": adds, "adds_losing": adds_losing})
                if abs(new) > tol:  # flipped: the rest opens a new position
                    opened, pnl, fees, side, adds, adds_losing, avg = r.time, 0.0, 0.0, np.sign(new), 0, 0, r.px
    return pd.DataFrame(trips)


def profile(address, raw, trend):
    f = pd.DataFrame(raw["fills"])
    if f.empty:
        return None
    for c in ("px", "sz", "closedPnl", "fee"):
        f[c] = f[c].astype(float)
    f = f[~f["coin"].str.startswith("@") & ~f["coin"].str.contains("/")]  # spot pairs (@index or A/B names): keep perpetuals
    if len(f) < 50:
        return None
    f = f.sort_values("time")
    f["notional"] = f["px"] * f["sz"]
    f["start"] = f["startPosition"].astype(float)
    opens = f[f["dir"].str.startswith("Open")]
    closes = f[f["closedPnl"] != 0].copy()
    # One close event = all closing fills of a coin in the same hour (big traders split orders into many fills).
    closes["hour"] = closes["time"] // 3_600_000
    ev = closes.groupby(["coin", "hour"])["closedPnl"].sum()
    wins, losses = ev[ev > 0], ev[ev < 0]
    days = max((f["time"].max() - f["time"].min()) / 8.64e7, 1)

    # Adding to a losing position: an opening fill on an existing position at a worse price than the
    # average price of the opening fills since that position was last flat.
    adds = adds_losing = 0
    for coin, g in f.groupby("coin"):
        vwap_n = vwap_q = 0.0
        for r in g.itertuples():
            if abs(r.start) < 1e-12:
                vwap_n = vwap_q = 0.0
            if r.dir.startswith("Open"):
                if abs(r.start) > 1e-12 and vwap_q > 0:
                    adds += 1
                    long = r.dir == "Open Long"
                    adds_losing += (r.px < vwap_n / vwap_q) if long else (r.px > vwap_n / vwap_q)
                vwap_n += r.px * r.sz
                vwap_q += r.sz
    # Trend and chasing at entry, weighted by size: the coin's position vs its 50-day average, and its
    # previous day's move, as known at the start of the day of the fill.
    aligned = chased = weight = 0.0
    for r in opens.itertuples():
        c = trend.get(r.coin)
        if c is None:
            continue
        day = pd.Timestamp(r.time, unit="ms", tz="UTC").normalize() - pd.Timedelta("1D")
        if day not in c.index or np.isnan(c.loc[day, "trend"]):
            continue
        sign = 1 if r.dir == "Open Long" else -1
        aligned += r.notional * (sign == c.loc[day, "trend"])
        chased += r.notional * (sign * c.loc[day, "ret1"] > 0.03)
        weight += r.notional
    st = raw["state"] or {}
    lev_n = lev_w = 0.0
    for ap in st.get("assetPositions", []):
        pos = ap.get("position", {})
        v = abs(float(pos.get("positionValue", 0) or 0))
        lev = (pos.get("leverage") or {}).get("value")
        if v and lev:
            lev_n += v * float(lev)
            lev_w += v
    t = round_trips(f)
    hold_w = hold_l = np.nan
    if len(t):
        t["hours"] = (t["closed"] - t["opened"]) / 3.6e6
        hold_w = t.loc[t["pnl"] > 0, "hours"].median()
        hold_l = t.loc[t["pnl"] < 0, "hours"].median()
    gross = ev.sum()
    return {
        "days of data": days, "fills per day": len(f) / days, "close events": len(ev),
        "BTC/ETH/SOL share (size)": f.loc[f["coin"].isin(MAJORS), "notional"].sum() / f["notional"].sum(),
        "coins traded": f["coin"].nunique(),
        "short share (opening size)": opens.loc[opens["dir"] == "Open Short", "notional"].sum() / opens["notional"].sum()
        if len(opens) else np.nan,
        "win rate (close events)": (ev > 0).mean() if len(ev) else np.nan,
        "avg win / avg loss": wins.mean() / -losses.mean() if len(wins) and len(losses) else np.nan,
        "biggest loss / avg win": -losses.min() / wins.mean() if len(wins) and len(losses) else np.nan,
        "hours held (wins)": hold_w, "hours held (losses)": hold_l, "complete round trips": len(t),
        "adds that were at a loss": adds_losing / adds if adds else np.nan,
        "opened with the 50-day trend": aligned / weight if weight else np.nan,
        "opened after a 3%+ day the same way": chased / weight if weight else np.nan,
        "leverage on open positions": lev_n / lev_w if lev_w else np.nan,
        "fees / gross closed PnL": f["fee"].sum() / abs(gross) if gross else np.nan,
        "closed PnL in window": gross,
    }


def main():
    lb = leaderboard()
    pool = lb[(lb["volume"] >= MIN_VOLUME) & (lb["edge"].abs() >= MIN_EDGE)]
    print(f"{len(lb):,} traders on the leaderboard; {len(pool):,} directional (|PnL|/volume >= {MIN_EDGE:.1%}, "
          f"volume >= ${MIN_VOLUME / 1e6:.0f}M)")
    groups = {"winners": pool.nlargest(N, "pnl"), "losers": pool.nsmallest(N, "pnl")}
    for g, d in groups.items():
        print(f"{g}: all-time PnL ${d['pnl'].min() / 1e6:,.1f}M to ${d['pnl'].max() / 1e6:,.1f}M, "
              f"median volume ${d['volume'].median() / 1e6:,.0f}M, median PnL/volume {d['edge'].median():+.2%}")

    raw = {}
    for g, d in groups.items():
        for i, a in enumerate(d["address"]):
            try:
                raw[a] = fills(a)
            except Exception as e:
                print(f"  skipped one {g[:-1]} ({type(e).__name__})")
            if i % 10 == 9:
                print(f"  {g}: {i + 1}/{N} fetched", flush=True)

    # 50-day trend per coin: +1 when the daily close is above its 50-day average, -1 below.
    first = min((min(x["time"] for x in r["fills"]) for r in raw.values() if r["fills"]), default=0)
    counts = pd.Series([f["coin"] for r in raw.values() for f in r["fills"]]).value_counts()
    trend = {}
    for coin in [c for c in counts.index if not c.startswith("@") and "/" not in c][:40]:
        c = daily_closes(coin, first)
        if c is not None and len(c) > 50:
            trend[coin] = pd.DataFrame({"trend": np.sign(c - c.rolling(50).mean()), "ret1": c.pct_change()})
    print(f"trend data for the {len(trend)} most traded coins")

    rows = []
    for g, d in groups.items():
        for a in d["address"]:
            if a in raw:
                p = profile(a, raw[a], trend)
                if p:
                    rows.append({"group": g, **p})
    prof = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    table = prof.groupby("group").median(numeric_only=True).T[["winners", "losers"]]
    table.loc["traders analysed"] = prof["group"].value_counts()[["winners", "losers"]].values
    hold = {g: d[["hours held (wins)", "hours held (losses)"]].dropna() for g, d in prof.groupby("group")}
    fmt = lambda k, v: (f"{v:.0%}" if any(s in k for s in ("share", "rate", "trend", "fees /", "adds", "after")) else
                        f"${v:,.0f}" if "PnL" in k else f"{v:,.2f}")
    print("\n=== Median per trader ===")
    for k, r in table.iterrows():
        print(f"{k:<32} {fmt(k, r['winners']):>14} {fmt(k, r['losers']):>14}")
    for g, h in hold.items():
        print(f"{g}: holding time measured for {len(h)} traders with complete round trips")
    prof.to_csv(ROOT / "reports" / "top_traders_profiles.csv", index=False)


if __name__ == "__main__":
    main()
