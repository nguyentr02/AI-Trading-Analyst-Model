"""US stocks on Binance: live prices and charts from Binance, long history from Yahoo for testing.

Binance lists US stocks two ways (2026): tokenized stocks on spot ("NVDAB/USDT", from mid-2026) and TradFi
perpetual futures ("NVDA/USDT:USDT", from early 2026, far more volume, trades 24/7). Live prices and charts come
from the perpetuals. Their history is only months long, too short to test anything, so the hold-vs-trade tests use
10 years of daily closes from Yahoo Finance (the same stock).

Survivorship bias: this list is today's well-known stocks that Binance chose to list, so their past returns look
better than a stock picked in advance would have. Hold-vs-trade on the SAME stock is a fair comparison; the size
of the hold returns is not.
"""
import time

import numpy as np
import pandas as pd
import requests

from . import config, data

STOCKS = {
    "NVDA": "Nvidia", "MSFT": "Microsoft", "AAPL": "Apple", "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta",
    "AMD": "AMD", "TSLA": "Tesla", "NFLX": "Netflix", "PLTR": "Palantir", "INTC": "Intel", "UBER": "Uber",
    "COIN": "Coinbase", "HOOD": "Robinhood", "MSTR": "Strategy (MicroStrategy)", "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF",
}
ETFS = {"SPY", "QQQ"}
COST = 0.001  # per side
RULES = (50, 100, 200)  # moving-average trend rules tested
CHOOSE, CHECK = ("2016-01-01", "2023-01-01"), ("2023-01-01", "2100-01-01")


def perp(ticker):
    return f"{ticker}/USDT:USDT"


def spot_token(ticker):
    return f"{ticker}B/USDT"


def candles(ticker, max_age_hours=12):
    """Daily open/high/low/close/volume for up to 10 years, adjusted for splits and dividends (all prices scaled by
    adjusted close / close), cached in data/stock_<TICKER>_1d.csv."""
    path = config.DATA_DIR / f"stock_{ticker}_1d.csv"
    cached = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
    if cached is not None and "open" in cached and time.time() - path.stat().st_mtime < max_age_hours * 3600:
        return cached
    j = None
    for attempt in range(4):  # Yahoo drops connections when asked too quickly
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                             params={"range": "10y", "interval": "1d"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            r.raise_for_status()
            j = r.json()["chart"]["result"][0]
            break
        except (requests.RequestException, ValueError, KeyError, TypeError):
            time.sleep(2 * (attempt + 1))
    if j is None:
        if cached is not None and "open" in cached:
            return cached  # stale but usable
        raise RuntimeError(f"could not download {ticker} history from Yahoo")
    q = j["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(j["timestamp"], unit="s", utc=True).normalize())
    adj = pd.Series(j["indicators"]["adjclose"][0]["adjclose"], index=df.index)
    factor = adj / df["close"]
    for k in ("open", "high", "low", "close"):
        df[k] = df[k] * factor
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "time"
    df.to_csv(path)
    time.sleep(0.5)
    return df


def history(ticker, max_age_hours=12):
    """Daily adjusted closes for up to 10 years."""
    return candles(ticker, max_age_hours)["close"]


def live_prices(tickers=STOCKS):
    """Last price and 24h change of each stock's Binance perpetual, as {ticker: (price, change, quote volume)}."""
    tk = data.exchange().fetch_tickers([perp(t) for t in tickers])
    return {t: (tk[perp(t)]["last"], (tk[perp(t)].get("percentage") or 0) / 100, tk[perp(t)].get("quoteVolume") or 0)
            for t in tickers if perp(t) in tk}


def _rule_returns(c, n):
    """Daily net returns of holding while the close is above its n-day average (decided at the close)."""
    pos = (c > c.rolling(n).mean()).astype(float).shift(1)
    return (pos * c.pct_change() - pos.diff().abs() * COST).dropna()


def _stats(r):
    if len(r) < 60:
        return {"return": np.nan, "sharpe": np.nan, "worst": np.nan}
    eq = (1 + r).cumprod()
    return {"return": eq.iloc[-1] - 1, "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan,
            "worst": (eq / eq.cummax() - 1).min()}


def _window(r, period):
    a, b = (pd.Timestamp(x, tz="UTC") for x in period)
    return r[(r.index >= a) & (r.index < b)]


def analyse(ticker):
    """Hold view (trend, momentum, risk) and trade test (trend rules vs holding) for one stock."""
    c = history(ticker)
    last = c.iloc[-1]
    ret = lambda days: last / c.iloc[-1 - days] - 1 if len(c) > days else np.nan
    r = np.log(c).diff().dropna()
    out = {"ticker": ticker, "name": STOCKS[ticker], "close": float(last), "date": c.index[-1],
           "1m": ret(21), "3m": ret(63), "1y": ret(252), "3y": ret(756),
           "momentum_12_1": c.iloc[-22] / c.iloc[-253] - 1 if len(c) > 253 else np.nan,  # skip the last month
           "vol": float(r.tail(252).std() * np.sqrt(252)), "from_high": float(last / c.tail(252).max() - 1),
           "worst_10y": float((c / c.cummax() - 1).min()),
           "above_50d": bool(last > c.tail(50).mean()), "above_200d": bool(last > c.tail(200).mean()),
           "from_200d": float(last / c.tail(200).mean() - 1), "years": len(c) / 252}
    hold = c.pct_change().dropna()
    hold.iloc[0] -= COST
    tests = {"hold": hold, **{f"sma{n}": _rule_returns(c, n) for n in RULES}}
    for name, series in tests.items():
        for label, period in (("choose", CHOOSE), ("check", CHECK)):
            s = _stats(_window(series, period))
            out[f"{name}_{label}_sharpe"] = s["sharpe"]
            out[f"{name}_{label}_return"] = s["return"]
            out[f"{name}_{label}_worst"] = s["worst"]
    rules = [f"sma{n}" for n in RULES if not np.isnan(out[f"sma{n}_choose_sharpe"])]
    if rules:
        best = max(rules, key=lambda k: out[f"{k}_choose_sharpe"])
        out["rule"] = best
        out["rule_beats_hold"] = bool(out[f"{best}_check_sharpe"] > out["hold_check_sharpe"])
        out["rule_in_now"] = bool(last > c.tail(int(best[3:])).mean())
    else:
        out.update(rule=None, rule_beats_hold=None, rule_in_now=None)
    return out


def analyse_all():
    rows = []
    for t in STOCKS:
        try:
            rows.append(analyse(t))
        except Exception as e:
            rows.append({"ticker": t, "name": STOCKS[t], "error": str(e)})
        time.sleep(0.2)
    return pd.DataFrame(rows)


PORTFOLIO = config.ROOT / "stock_portfolio.json"  # your stock holdings and spare cash; kept off GitHub
MAX_WEIGHT = 0.25  # above this share of the stock portfolio, one stock is flagged as concentrated


def load_portfolio():
    import json
    try:
        p = json.loads(PORTFOLIO.read_text())
        return p.get("holdings", []), float(p.get("cash", 0.0))
    except (OSError, ValueError):
        return [], 0.0


def save_portfolio(holdings, cash):
    import json
    with config.atomic(PORTFOLIO) as tmp:
        tmp.write_text(json.dumps({"holdings": holdings, "cash": cash}, indent=2))


def advise(holdings, cash, analysis, prices):
    """Hold-first advice per holding, plus a plan for spare cash. Testing found holding beat trend-rule trading on
    every stock here, so the advice never says sell on a trend signal; it flags concentration and risk instead."""
    a = analysis.set_index("ticker") if len(analysis) else analysis
    rows, total = [], cash
    for h in holdings:
        t = h["ticker"].upper()
        px = prices.get(t) or (a.loc[t, "close"] if t in a.index else float(h["avg_cost"]))
        rows.append({"ticker": t, "shares": float(h["shares"]), "avg_cost": float(h["avg_cost"]), "price": px,
                     "value": float(h["shares"]) * px})
        total += rows[-1]["value"]
    for r in rows:
        r["weight"] = r["value"] / total if total else 0.0
        r["pnl"] = r["value"] - r["shares"] * r["avg_cost"]
        r["pnl_pct"] = r["price"] / r["avg_cost"] - 1 if r["avg_cost"] else float("nan")
        info = a.loc[r["ticker"]] if r["ticker"] in a.index else None
        if r["weight"] > MAX_WEIGHT:
            r["action"] = "TRIM"
            r["reason"] = (f"{r['weight']:.0%} of your stock money is in this one stock. Consider trimming to about "
                           f"{MAX_WEIGHT:.0%} or less, so one bad fall can't hurt too much.")
        elif info is None:
            r["action"], r["reason"] = "HOLD", "Not tracked here; no trend or risk data."
        elif not info["above_200d"]:
            r["action"] = "HOLD"
            r["reason"] = (f"Below its 200-day average ({info['from_200d']:+.0%}). In testing, selling on that did not "
                           "beat holding; check that the reason you own it still holds.")
        else:
            r["action"] = "HOLD"
            r["reason"] = f"Uptrend ({info['from_200d']:+.0%} vs its 200-day average). Holding beat trading in testing."
        if info is not None:
            r["reason"] += f" Worst fall in 10 years: {info['worst_10y']:.0%}."
    plan = None
    if cash >= config.MIN_TRADE_USDT * 4:
        plan = (f"Invest the ${cash:,.0f} in 4 weekly steps of ${cash / 4:,.0f} rather than all at once (buying in steps "
                f"beat a lump sum when prices fell). Putting part in an index ETF (SPY or QQQ) spreads the risk.")
    return pd.DataFrame(rows), total, plan


ALERT_STATE = config.LOG_DIR / "stock_alerts.json"
BIG_MOVE = 0.08  # a daily close this far from the previous one is alerted


def alerts(notify_fn):
    """After a US close: alert when a stock crosses its 200-day average or moves BIG_MOVE+ in a day (once each)."""
    import json
    try:
        state = json.loads(ALERT_STATE.read_text())
    except (OSError, ValueError):
        state = {}
    sent = []
    for t, name in STOCKS.items():
        c = history(t, max_age_hours=1)
        if len(c) < 202:
            continue
        day = str(c.index[-1].date())
        ma = c.rolling(200).mean()
        above_now, above_before = c.iloc[-1] > ma.iloc[-1], c.iloc[-2] > ma.iloc[-2]
        move = c.iloc[-1] / c.iloc[-2] - 1
        msgs = []
        if above_now != above_before:
            msgs.append(f"crossed {'above' if above_now else 'below'} its 200-day average (${ma.iloc[-1]:,.2f}); "
                        f"close ${c.iloc[-1]:,.2f}")
        if abs(move) >= BIG_MOVE:
            msgs.append(f"moved {move:+.1%} in one day, to ${c.iloc[-1]:,.2f}")
        for m in msgs:
            key = f"{t}|{day}|{m[:12]}"
            if key not in state:
                state[key] = True
                sent.append(f"{t} ({name}) {m}")
    if sent:
        notify_fn("US stocks: " + ", ".join(s.split(" ")[0] for s in sent), "\n".join(sent) +
                  "\n(Information only. In testing, trading on trend crossings did not beat holding.)")
    with config.atomic(ALERT_STATE) as tmp:
        tmp.write_text(json.dumps(dict(list(state.items())[-500:]), indent=1))
    return sent


def momentum_test(top=5):
    """Cross-sectional momentum on these stocks (ETFs excluded): each month hold the `top` with the best
    12-1 month return, equal weight, vs holding all equally. Choose period / check period as above."""
    closes = pd.DataFrame({t: history(t) for t in STOCKS if t not in ETFS}).sort_index()
    monthly = closes.resample("ME").last()
    mom = monthly.shift(1) / monthly.shift(12) - 1  # 12-1: past year, skipping the last month
    nxt = monthly.pct_change().shift(-1)
    rows = []
    for t in monthly.index[12:-1]:
        avail = mom.loc[t].dropna()
        if len(avail) < top + 3:
            continue
        pick = avail.nlargest(top).index
        turnover_cost = 2 * COST  # rough: monthly rebalance
        rows.append({"time": t, "momentum": nxt.loc[t, pick].mean() - turnover_cost,
                     "equal": nxt.loc[t, avail.index].mean()})
    df = pd.DataFrame(rows).set_index("time")
    out = {}
    for label, period in (("choose", CHOOSE), ("check", CHECK)):
        w = _window(df, period)
        for col in ("momentum", "equal"):
            r = w[col]
            out[f"{col}_{label}_sharpe"] = r.mean() / r.std() * np.sqrt(12) if len(r) > 6 and r.std() else np.nan
            out[f"{col}_{label}_return"] = (1 + r).prod() - 1
    latest = mom.iloc[-1].dropna().sort_values(ascending=False)
    out["top_now"] = list(latest.index[:top])
    return out
