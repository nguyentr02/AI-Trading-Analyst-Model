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
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from dateutil.easter import easter

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
    """Advice per holding, plus a plan for spare cash, following the tested monthly rule (buy_list below): hold the
    stocks above their 200-day average in equal shares and sell one at the monthly review if it is below it (daily
    in-and-out trading on one stock did NOT beat holding; the monthly rule across the stocks did). Also flags
    concentration. ETFs (SPY, QQQ) are held."""
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
        if r["weight"] > MAX_WEIGHT and r["ticker"] not in ETFS:  # an index ETF is already diversified
            r["action"] = "TRIM"
            r["reason"] = (f"{r['weight']:.0%} of your stock money is in this one stock. Consider trimming to about "
                           f"{MAX_WEIGHT:.0%} or less, so one bad fall can't hurt too much.")
        elif info is None:
            r["action"], r["reason"] = "HOLD", "Not tracked here; no trend or risk data."
        elif r["ticker"] in ETFS:
            r["action"], r["reason"] = "HOLD", "Index ETF: a diversified core to hold."
        else:
            c = history(r["ticker"])
            avg200 = c.tail(200).mean()
            if r["price"] > avg200:
                r["action"] = "HOLD"
                r["reason"] = (f"In the buy-for-hold list: {r['price'] / avg200 - 1:+.0%} vs its 200-day average. "
                               f"Next review {next_review():%b %d}.")
            else:
                r["action"] = "SELL at review"
                r["reason"] = (f"Below its 200-day average ({r['price'] / avg200 - 1:+.0%}). The tested rule sells it at "
                               f"the monthly review ({next_review():%b %d}, US close) if it is still below then.")
        if info is not None:
            r["reason"] += f" Worst fall in 10 years: {info['worst_10y']:.0%}."
    plan = None
    if cash >= config.MIN_TRADE_USDT * 4:
        picks = [t for t in PICKS if (h := history(t)).iloc[-1] > h.tail(200).mean()]
        plan = (f"Invest the ${cash:,.0f} in 4 weekly steps of ${cash / 4:,.0f} (buying in steps beat a lump sum when "
                f"prices fell), spread equally over the buy-for-hold list ({', '.join(picks)}), or partly in an "
                "index ETF (SPY or QQQ).")
    return pd.DataFrame(rows), total, plan


# ---------- buy-for-hold list (experiments/stock_hold_picks.py, adopted 2026-10-06) ----------
# Equal weight across the stocks whose close is above their 200-day average, reviewed once a month: Sharpe 0.83 vs
# 0.61 for holding all equally on 2016-2022 and 2.05 vs 1.91 on 2023-2026; worst fall -46% vs -59% on 2016-2022.
PICKS = [t for t in STOCKS if t not in ETFS]


def buy_list(analysis, live=None):
    """The adopted buy-for-hold list now: each stock in or out, using the live Binance price as today's close."""
    live = live or {}
    rows = []
    for _, r in analysis.set_index("ticker").loc[[t for t in PICKS if t in set(analysis["ticker"])]].iterrows():
        c = history(r.name)
        avg200 = c.tail(200).mean()
        px = live.get(r.name, (r["close"],))[0]
        rows.append({"ticker": r.name, "name": r["name"], "price": px, "avg200": avg200, "in": px > avg200,
                     "from_200d": px / avg200 - 1, "vol": r["vol"], "worst_10y": r["worst_10y"], "1y": r["1y"]})
    df = pd.DataFrame(rows)
    n = int(df["in"].sum())
    df["weight"] = np.where(df["in"], 1 / n if n else 0.0, 0.0)
    return df.sort_values(["in", "from_200d"], ascending=[False, False])


def top_pick(buy):
    """The single stock to buy if you only buy one: the steadiest (lowest past-year volatility) stock in the buy
    list. Chosen in experiments/stock_top_pick.py among four one-stock rules (best on 2021-2022: +5% vs -16% for the
    whole list, worst fall -28%), but it lagged the whole list in 2023-2026 (+172% vs +921%): no one-stock rule beat
    holding the list."""
    inside = buy[buy["in"]]
    return None if inside.empty else inside.sort_values("vol").iloc[0]


def runner_ups(buy, held, n=2):
    """The next steadiest stocks in the buy list after the held top pick: the second and third choice."""
    inside = buy[buy["in"] & (buy["ticker"] != held)]
    return inside.sort_values("vol").head(n)


# The 3 steadiest in equal shares (experiments/stock_top_pick.py, added 2026-10-07, not used to choose the rule).
TOP3_RECORD = {"top3_2122": -0.08, "top3_2326": 2.92, "top3_worst": -0.39}

TOP_PICK_STATE = config.LOG_DIR / "top_pick.json"


def _month_review(month_start):
    """The review of the month starting at `month_start` (New York): its last US trading day, at the close."""
    end = (month_start + pd.offsets.MonthEnd(1)).normalize()
    while not us_market_open(end.date()):
        end -= pd.Timedelta("1D")
    return end + pd.Timedelta(hours=us_close_hour(end.date()))


def last_review_day(now=None):
    """The most recent monthly review that has happened: this month's last US close if it has passed, else the
    previous month's (New York dates)."""
    now = now or pd.Timestamp.now(tz="America/New_York")
    this_month = now.normalize().replace(day=1)
    review = _month_review(this_month)
    if now >= review:
        return review.normalize()
    return _month_review(this_month - pd.offsets.MonthBegin(1)).normalize()


def held_top_pick(buy):
    """(held pick, today's best option). The pick is chosen at a monthly review and held until the next one
    (switching daily or on chart breaks did worse: experiments/top_pick_events.py); the AI still checks every day
    which stock would be best today, and reports it. Saved in logs/top_pick.json."""
    import json
    best = top_pick(buy)
    try:
        state = json.loads(TOP_PICK_STATE.read_text())
    except (OSError, ValueError):
        state = {}
    review = str(last_review_day().date())
    if best is not None and (state.get("review") != review or state.get("ticker") not in set(buy["ticker"])):
        state = {"ticker": best["ticker"], "review": review, "chosen": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                 "price": float(best["price"])}
        with config.atomic(TOP_PICK_STATE) as tmp:
            tmp.write_text(json.dumps(state, indent=1))
    held = buy[buy["ticker"] == state.get("ticker")]
    return (held.iloc[0] if len(held) else best), best, state


NEWS_WARNING, NEWS_MIN_HEADLINES = -0.30, 5


TOP_PICK_RECORD = {"pick_2122": 0.05, "list_2122": -0.16, "pick_2326": 1.72, "list_2326": 9.21,
                   "pick_worst": -0.32, "list_worst": -0.44}


def next_review(now=None):
    """The next monthly review: the last US trading day's close of this month, or of next month once this month's
    has passed (weekends and NYSE holidays skipped)."""
    now = now or pd.Timestamp.now(tz="America/New_York")
    this_month = now.normalize().replace(day=1)
    review = _month_review(this_month)
    return review if now < review else _month_review(this_month + pd.offsets.MonthBegin(1))


# ---------- US market calendar (NYSE) ----------
def _observed(d):
    """A fixed-date holiday on a weekend is taken on the Friday before (Saturday) or the Monday after (Sunday)."""
    return d - timedelta(days=1) if d.weekday() == 5 else d + timedelta(days=1) if d.weekday() == 6 else d


def _nth_weekday(year, month, weekday, n):
    """The n-th `weekday` (0 = Monday) of a month; n = -1 for the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


@lru_cache(maxsize=None)
def us_holidays(year):
    """Full-day NYSE closures in a year (the regular rules; one-off closures such as national days of mourning are
    not included)."""
    days = {
        _nth_weekday(year, 1, 0, 3),   # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),   # Washington's Birthday
        easter(year) - timedelta(days=2),  # Good Friday
        _nth_weekday(year, 5, 0, -1),  # Memorial Day
        _observed(date(year, 7, 4)),   # Independence Day
        _nth_weekday(year, 9, 0, 1),   # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),  # Christmas
    }
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))  # Juneteenth
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:  # on a Saturday the NYSE does not close the Friday before (Dec 31)
        days.add(_observed(new_year))
    return days


def us_market_open(day):
    """Whether the US market trades on `day` (a date): a weekday that is not an NYSE holiday."""
    return day.weekday() < 5 and day not in us_holidays(day.year)


def us_close_hour(day):
    """The US close on `day`, New York time: 13:00 on the regular early-close days (the day after Thanksgiving,
    Christmas Eve, and July 3 when it is Monday to Thursday), else 16:00."""
    early = {_nth_weekday(day.year, 11, 3, 4) + timedelta(days=1), date(day.year, 12, 24)}
    if date(day.year, 7, 3).weekday() < 4:
        early.add(date(day.year, 7, 3))
    return 13 if day in early else 16


def next_us_close(now=None):
    """The next US market close that hasn't happened yet, as a New York Timestamp."""
    now = now or pd.Timestamp.now(tz="America/New_York")
    day = now.date()
    while not (us_market_open(day) and (day > now.date() or now.hour < us_close_hour(day))):
        day += timedelta(days=1)
    return pd.Timestamp(day, tz="America/New_York") + pd.Timedelta(hours=us_close_hour(day))


def last_us_close_day(now=None):
    """The date (New York) of the most recent US market close that has already happened."""
    now = now or pd.Timestamp.now(tz="America/New_York")
    day = now.date()
    if not (us_market_open(day) and now.hour >= us_close_hour(day)):
        day -= timedelta(days=1)
    while not us_market_open(day):
        day -= timedelta(days=1)
    return day


# ---------- market crash monitor ----------
CRASH_STATE = config.LOG_DIR / "market_crash_state.json"
CORRECTION, BEAR, SHARP_DAY = -0.10, -0.20, -0.04
RECOVER_BAND = 0.01  # leave a correction / bear market only 1 point above its line


def market_status(spy_live=None):
    """Where the US market (SPY) stands: vs its 1-year high, its 200-day average, and today's move."""
    c = history("SPY")
    ny = pd.Timestamp.now(tz="America/New_York")
    if str(c.index[-1].date()) == ny.strftime("%Y-%m-%d") and ny.hour < us_close_hour(ny.date()):
        c = c.iloc[:-1]  # Yahoo's bar for a session still trading is not a close
    px = spy_live or float(c.iloc[-1])
    high = max(float(c.tail(252).max()), px)
    avg200 = float(c.tail(200).mean())
    dd, day = px / high - 1, px / float(c.iloc[-1]) - 1 if spy_live else float(c.iloc[-1] / c.iloc[-2] - 1)
    level = "Bear market" if dd <= BEAR else "Correction" if dd <= CORRECTION else "Normal"
    return {"price": px, "from_high": dd, "high": high, "avg200": avg200, "above_200d": px > avg200,
            "close": float(c.iloc[-1]), "close_above_200d": float(c.iloc[-1]) > avg200, "today": day,
            "level": level}


def crash_check(notify_fn, spy_live):
    """Alert once per change: the market entering a correction (-10%) or bear market (-20%), SPY closing below or
    back above its 200-day average, or a sharp day (-4%). Information only: in testing, selling everything at these
    points cut the 2020/2022 falls but missed the 2023-2026 rebound (experiments/stock_hold_picks.py)."""
    import json
    s = market_status(spy_live)
    try:
        state = json.loads(CRASH_STATE.read_text())
    except (OSError, ValueError):
        state = {"level": "Normal", "above_200d": True, "sharp_day": None}
    msgs = []
    order = ["Normal", "Correction", "Bear market"]
    prev = state.get("level", "Normal")
    level = s["level"]
    # A level is left only once the price is RECOVER_BAND past its line, so prices wiggling around -10% or -20%
    # don't alert every minute.
    if order.index(level) < order.index(prev):
        line = BEAR if prev == "Bear market" else CORRECTION
        if s["from_high"] < line + RECOVER_BAND:
            level = prev
    if order.index(level) > order.index(prev):
        msgs.append(f"{level}: the S&P 500 (SPY ${s['price']:,.2f}) is {s['from_high']:.1%} below its 1-year high.")
    elif level == "Normal" and prev != "Normal":
        msgs.append(f"Recovered: the S&P 500 is back within 10% of its high ({s['from_high']:.1%}).")
    # The 200-day line is judged on daily closes, as the tested rule does, not on the live price.
    if s["close_above_200d"] != state.get("above_200d", True):
        msgs.append(f"The S&P 500 closed {'above' if s['close_above_200d'] else 'below'} its 200-day average "
                    f"(${s['avg200']:,.2f}).")
    today = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    if s["today"] <= SHARP_DAY and state.get("sharp_day") != today:
        msgs.append(f"Sharp drop: the S&P 500 is {s['today']:+.1%} today.")
        state["sharp_day"] = today
    state.update(level=level, above_200d=s["close_above_200d"])
    with config.atomic(CRASH_STATE) as tmp:
        tmp.write_text(json.dumps(state))
    if msgs:
        notify_fn("Market warning: " + level, "\n".join(msgs) + "\nWhat testing found: selling everything at "
                  "such points cut the worst fall in 2020/2022 (-46% vs -59%) but missed the 2023-2026 rebound. The "
                  "tested rule sells a stock when it closes a month below its 200-day average. Check your risk limit.")
    return msgs, s


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
