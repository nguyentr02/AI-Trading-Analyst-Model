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


def history(ticker, max_age_hours=12):
    """Daily closes (split- and dividend-adjusted) for up to 10 years, cached in data/stock_<TICKER>_1d.csv."""
    path = config.DATA_DIR / f"stock_{ticker}_1d.csv"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
        return pd.read_csv(path, index_col=0, parse_dates=True)["close"]
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}", params={"range": "10y", "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]
    close = pd.Series(j["indicators"]["adjclose"][0]["adjclose"],
                      index=pd.to_datetime(j["timestamp"], unit="s", utc=True).normalize(), name="close").dropna()
    close.to_frame().to_csv(path)
    return close


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
