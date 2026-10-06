"""Paper trading: pretend accounts traded live from the moment they are opened, for a fixed trial period.

All accounts start with the same balance at the same moment, split into one sleeve per coin (e.g. $250
each of $1,000), so one coin's fall can't drain the others. Fills use the live Binance price with a 0.1% fee
and 0.05% slippage per trade. Nothing real is bought or sold.

Accounts:
- ai          "Smart" (simulate.SMART): sized by next-1-day confidence, sells half if the 3-day view is still
              up, takes half profit at +10% once confidence fades, no stop-loss. Decides at every 4h close and,
              between closes, on the live readings once an action has held CONFIRM_MINUTES (then a
              COOLDOWN_MINUTES wait per coin; both guards are a judgement, not backtested). Plus shock dip-buys.
- trend_dip   50-day trend rule (hold while the daily close is above its 50-day average) plus shock dip-buys.
- trend_ai    trend ensemble (close above its 20/50/100/200-day averages, averaged) x AI sizer
              clip(0.5 + 2 x (P(up, next 3 days) - 0.5), 0, 1), rebalanced at each daily close; plus shock
              dip-buys. (experiments/trend_ai_strategies.py: about half the drawdown and half the return.)
- trend       benchmark: the 50-day trend rule alone.
- hold        benchmark: each sleeve bought at the start, never sold.

Shock dip-buy (experiments/shock_dip_buy.py): if a coin closes 10%+ below its close an hour earlier (15-minute
closes), buy with up to half the sleeve from its cash and sell 4 hours later.

When the trial ends, trading stops and a report is written to docs/backTestResult; a weekly update is sent
before that. Everything is saved in paper/ (kept off GitHub): account.json, trades.csv, balance.csv.
"""
import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, data, signals
from .simulate import SMART

FEE, SLIPPAGE = config.FEE, 0.0005
DIR = config.ROOT / "paper"
ACCOUNT, TRADES, BALANCE = DIR / "account.json", DIR / "trades.csv", DIR / "balance.csv"
TRADE_COLUMNS = ["time", "account", "symbol", "side", "price", "quantity", "total", "fee", "reason", "cash_after"]
ACCOUNTS = {"ai": "AI Smart", "trend_dip": "Trend + dip-buy", "trend_ai": "Trend x AI + dip-buy",
            "trend": "Trend rule (benchmark)", "hold": "Buy & hold (benchmark)"}
SHOCK_ACCOUNTS = ("ai", "trend_dip", "trend_ai")
TRIAL = pd.Timedelta("28D")

# Smart trading on the live readings: an action must be wanted CONFIRM_MINUTES in a row, then each coin waits
# COOLDOWN_MINUTES. A judgement, not backtested: there is no minute-by-minute history of the live readings.
CONFIRM_MINUTES, COOLDOWN_MINUTES = 10, 60
# Shock dip-buy, tested in experiments/shock_dip_buy.py. The size is a judgement.
SHOCK_DROP, SHOCK_HOLD, SHOCK_SIZE = 0.10, pd.Timedelta("4h"), 0.5


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_open():
    return ACCOUNT.exists()


def load():
    return json.loads(ACCOUNT.read_text()) if ACCOUNT.exists() else None


def _save(acct):
    with config.atomic(ACCOUNT) as tmp:
        tmp.write_text(json.dumps(acct, indent=2))


def active(acct):
    """True while the trial is running."""
    return acct is not None and pd.Timestamp.now(tz="UTC") < pd.Timestamp(acct["ends"])


def _sleeve(cash):
    return {"cash": cash, "qty": 0.0, "cost": 0.0, "tp_done": False, "half_sold": False, "shock": None}


def open_account(start_cash=1000.0, trial=TRIAL):
    """Start every account now. An existing paper/ folder is archived, never deleted."""
    if DIR.exists():
        shutil.move(str(DIR), str(DIR.with_name(f"paper_archive_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}")))
    DIR.mkdir()
    prices = data.prices(config.SYMBOLS)
    each = start_cash / len(config.SYMBOLS)
    now = pd.Timestamp.now(tz="UTC")
    acct = {"opened": now.isoformat(timespec="seconds"), "ends": (now + trial).isoformat(timespec="seconds"),
            "start_cash": start_cash, "fee": FEE, "slippage": SLIPPAGE, "names": ACCOUNTS,
            "accounts": {a: {s: _sleeve(each) for s in config.SYMBOLS} for a in ACCOUNTS},
            "last_decision": {"ai": None, "daily": None}, "weeks_reported": 0, "final_report": None}
    pd.DataFrame(columns=TRADE_COLUMNS).to_csv(TRADES, index=False)
    for sym, sl in acct["accounts"]["hold"].items():
        _buy("hold", sym, sl, each, prices[sym], "buy & hold benchmark")
    _save(acct)
    snapshot(acct, prices)
    return acct


# ---------- execution ----------
def _log_trade(account, symbol, side, price, qty, total, reason, cash_after):
    row = {"time": _now(), "account": account, "symbol": symbol, "side": side, "price": round(price, 8),
           "quantity": round(qty, 8), "total": round(total, 4), "fee": round(total * FEE, 4), "reason": reason,
           "cash_after": round(cash_after, 4)}
    pd.DataFrame([row], columns=TRADE_COLUMNS).to_csv(TRADES, mode="a", header=not TRADES.exists(), index=False)


def _buy(account, symbol, sl, usdt, price, reason):
    usdt = min(usdt, sl["cash"])
    if usdt < config.MIN_TRADE_USDT:
        return False
    fill = price * (1 + SLIPPAGE)
    qty = usdt * (1 - FEE) / fill
    if sl["qty"] == 0:
        sl["tp_done"] = False
    sl["qty"] += qty
    sl["cost"] += usdt
    sl["cash"] -= usdt
    _log_trade(account, symbol, "BUY", fill, qty, usdt, reason, sl["cash"])
    return True


def _sell(account, symbol, sl, share, price, reason):
    qty = sl["qty"] * share
    if qty <= 0:
        return False
    if (sl["qty"] - qty) * price < config.MIN_TRADE_USDT:
        share, qty = 1.0, sl["qty"]  # what would be LEFT is dust: sell it all rather than leave a scrap
    fill = price * (1 - SLIPPAGE)
    proceeds = qty * fill * (1 - FEE)
    sl["cash"] += proceeds
    sl["qty"] -= qty
    sl["cost"] *= 1 - share
    if sl["qty"] <= 1e-12:
        sl["qty"], sl["cost"] = 0.0, 0.0
    _log_trade(account, symbol, "SELL", fill, qty, qty * fill, reason, sl["cash"])
    return True


def _rebalance(account, symbol, sl, target, price, reason):
    """Move the sleeve's coin share to `target` (0-1); trades smaller than MIN_TRADE_USDT are skipped."""
    value = sl["cash"] + sl["qty"] * price
    gap = target * value - sl["qty"] * price
    if abs(gap) < config.MIN_TRADE_USDT:
        return False  # too small to be worth a trade (and its fee)
    if gap > 0:
        return _buy(account, symbol, sl, gap, price, reason)
    if gap < 0 and sl["qty"] > 0:
        return _sell(account, symbol, sl, min(1.0, -gap / (sl["qty"] * price)), price, reason)
    return False


# ---------- AI Smart: at 4h closes and on the live readings ----------
def step_ai(prices=None):
    """Smart decisions for every coin, from the confirmed signals of the last closed 4h candle."""
    acct = load()
    if not active(acct):
        return _none()
    s1 = signals.current("4h", refresh=False).set_index("symbol")
    s3 = signals.current("1d", refresh=False).set_index("symbol")
    candle = str(s1["candle_close"].iloc[0])
    if acct["last_decision"]["ai"] == candle:
        return _none()  # already traded (e.g. the service restarted and caught up)
    prices = prices or data.prices(config.SYMBOLS)
    before = _trade_count()
    for sym, sl in acct["accounts"]["ai"].items():
        p1, p3 = float(s1.loc[sym, "prob_up"]), float(s3.loc[sym, "prob_up"])
        if p1 > SMART["sell_below"]:
            sl["half_sold"] = False
        if _execute(sym, sl, _decide(sl, p1, p3, prices[sym]), prices[sym], "at 4h close"):
            acct.setdefault("last_trade", {})[sym] = _now()
    acct["last_decision"]["ai"] = candle
    _save(acct)
    snapshot(acct, prices)
    return _trades_since(before)


def step_ai_live(live):
    """Smart decisions on the live readings (preview.compute() output), acting once they persist."""
    acct = load()
    if not active(acct) or "4h" not in live.get("models", {}) or "1d" not in live["models"]:
        return _none()
    before = _trade_count()
    now = pd.Timestamp.now(tz="UTC")
    streaks = acct.setdefault("live_streak", {})
    for sym, sl in acct["accounts"]["ai"].items():
        p1, p3 = live["models"]["4h"][sym]["prob_up"], live["models"]["1d"][sym]["prob_up"]
        px = live["models"]["4h"][sym]["price"]
        if p1 > SMART["sell_below"]:
            sl["half_sold"] = False
        intents = _decide(sl, p1, p3, px)
        key = "|".join(i[0] for i in intents)
        old = streaks.get(sym, {"key": "", "count": 0})
        streak = {"key": key, "count": old["count"] + 1 if key and key == old["key"] else (1 if key else 0)}
        streaks[sym] = streak
        last = acct.get("last_trade", {}).get(sym)
        cooled = last is None or now - pd.Timestamp(last) >= pd.Timedelta(minutes=COOLDOWN_MINUTES)
        if key and streak["count"] >= CONFIRM_MINUTES and cooled:
            if _execute(sym, sl, intents, px, f"live, held {streak['count']} min"):
                acct.setdefault("last_trade", {})[sym] = _now()
            streaks[sym] = {"key": "", "count": 0}
    _save(acct)
    return _trades_since(before)


def _decide(sl, p1, p3, px):
    """The Smart rules for one sleeve: a list of (kind, amount, reason); nothing is executed here."""
    intents = []
    qty, half_sold = sl["qty"], sl["half_sold"] and p1 <= SMART["sell_below"]
    if qty > 0:
        avg = sl["cost"] / qty
        if not sl["tp_done"] and px >= avg * (1 + SMART["take_profit"]) and p1 < SMART["take_profit_below"]:
            intents.append(("take_profit", 0.5, f"take half profit: up {px / avg - 1:+.1%}, confidence {p1:.0%}"))
            qty *= 0.5
    if qty > 0 and p1 <= SMART["sell_below"]:
        if p3 >= SMART["keep_half_if_3day"]:
            if not half_sold:
                intents.append(("sell_half", 0.5, f"sell half: next day weak ({p1:.0%}), next 3 days up ({p3:.0%})"))
        else:
            intents.append(("sell_all", 1.0, f"sell all: next day weak ({p1:.0%}), next 3 days weak ({p3:.0%})"))
    target = next((share for level, share in SMART["tiers"] if p1 >= level), None)
    if target is not None and not intents:
        value = sl["cash"] + sl["qty"] * px
        gap = target - sl["qty"] * px / value
        if gap >= SMART["min_change"] and min(gap * value, sl["cash"]) >= config.MIN_TRADE_USDT:
            intents.append((f"buy_{target:.2f}", gap * value, f"confidence {p1:.0%} -> {target:.0%} of the sleeve"))
    return intents


def _execute(sym, sl, intents, px, when):
    traded = False
    for kind, amount, reason in intents:
        if kind == "take_profit":
            traded |= _sell("ai", sym, sl, amount, px, f"{reason} ({when})")
            sl["tp_done"] = True
        elif kind == "sell_half":
            traded |= _sell("ai", sym, sl, amount, px, f"{reason} ({when})")
            sl["half_sold"] = True
        elif kind == "sell_all":
            traded |= _sell("ai", sym, sl, amount, px, f"{reason} ({when})")
        else:
            traded |= _buy("ai", sym, sl, amount, px, f"{reason} ({when})")
    return traded


# ---------- trend accounts: at each daily close ----------
def step_daily(prices=None):
    """Daily-close decisions for the trend benchmark, trend + dip-buy, and trend x AI."""
    acct = load()
    if not active(acct):
        return _none()
    closes = {s: data.drop_open_candle(data.load_cached(s, "1d"), "1d")["close"] for s in config.SYMBOLS}
    day = str(closes[config.SYMBOLS[0]].index[-1])
    if acct["last_decision"]["daily"] == day:
        return _none()  # today's close was already acted on
    prices = prices or data.prices(config.SYMBOLS)
    p3 = signals.current("1d", refresh=False).set_index("symbol")["prob_up"]
    before = _trade_count()
    for sym in config.SYMBOLS:
        c = closes[sym]
        above = {n: bool(c.iloc[-1] > c.iloc[-n:].mean()) for n in (20, 50, 100, 200)}
        sma50 = 1.0 if above[50] else 0.0
        for account in ("trend", "trend_dip"):
            _rebalance(account, sym, acct["accounts"][account][sym], sma50, prices[sym],
                       "close above its 50-day average" if above[50] else "close below its 50-day average")
        ens = sum(above.values()) / 4
        sizer = float(np.clip(0.5 + 2 * (float(p3[sym]) - 0.5), 0, 1))
        target = ens * sizer
        _rebalance("trend_ai", sym, acct["accounts"]["trend_ai"][sym], target, prices[sym],
                   f"trend {ens:.0%} x AI sizer {sizer:.0%} (P(up, 3 days) {float(p3[sym]):.0%}) -> {target:.0%} invested")
    acct["last_decision"]["daily"] = day
    _save(acct)
    snapshot(acct, prices)
    return _trades_since(before)


step_trend = step_daily  # older name


# ---------- shock dip-buy: at each 15-minute close ----------
def step_shock(prices=None):
    """For each trading account: close shock lots after 4 hours; buy a fresh 10%+ one-hour drop."""
    acct = load()
    if not active(acct):
        return _none()
    before = _trade_count()
    now = pd.Timestamp.now(tz="UTC")
    prices = prices or data.prices(config.SYMBOLS)
    checked = acct.setdefault("shock_checked", {})
    drops = {}
    for sym in config.SYMBOLS:
        bars = data.drop_open_candle(data.latest(sym, "15m", limit=8), "15m")["close"]
        bar_close = str(bars.index[-1] + pd.Timedelta("15min"))
        if checked.get(sym) != bar_close:  # a new 15-minute close to check
            checked[sym] = bar_close
            drops[sym] = bars.iloc[-1] / bars.iloc[-5] - 1
    for account in SHOCK_ACCOUNTS:
        for sym, sl in acct["accounts"][account].items():
            lot = sl.get("shock")
            if lot and now >= pd.Timestamp(lot["sell_at"]):
                fill = prices[sym] * (1 - SLIPPAGE)
                proceeds = lot["qty"] * fill * (1 - FEE)
                sl["cash"] += proceeds
                _log_trade(account, sym, "SELL", fill, lot["qty"], lot["qty"] * fill,
                           f"shock dip-buy exit after 4h ({proceeds / lot['cost'] - 1:+.1%})", sl["cash"])
                sl["shock"] = None
            drop = drops.get(sym)
            if drop is not None and drop <= -SHOCK_DROP and not sl.get("shock"):
                spend = min(sl["cash"], SHOCK_SIZE * (sl["cash"] + sl["qty"] * prices[sym]))
                if spend >= config.MIN_TRADE_USDT:
                    fill = prices[sym] * (1 + SLIPPAGE)
                    qty = spend * (1 - FEE) / fill
                    sl["cash"] -= spend
                    sl["shock"] = {"qty": qty, "cost": spend, "sell_at": str(now + SHOCK_HOLD)}
                    _log_trade(account, sym, "BUY", fill, qty, spend,
                               f"shock dip-buy: {drop:+.1%} in 1 hour, sell in 4h", sl["cash"])
    _save(acct)
    return _trades_since(before)


# ---------- valuation, history, trial milestones ----------
def holdings(sl):
    """(coin quantity, cost) of a sleeve, including an open shock lot."""
    shock = sl.get("shock") or {}
    return sl["qty"] + shock.get("qty", 0.0), sl["cost"] + shock.get("cost", 0.0)


def value(acct, prices):
    """Current value of every account."""
    return {a: sum(sl["cash"] + holdings(sl)[0] * prices[s] for s, sl in sleeves.items())
            for a, sleeves in acct["accounts"].items()}


def snapshot(acct=None, prices=None):
    """Append every account's balance to the balance history."""
    acct = acct or load()
    if acct is None:
        return
    prices = prices or data.prices(config.SYMBOLS)
    row = {"time": _now(), **{a: round(v, 4) for a, v in value(acct, prices).items()}}
    pd.DataFrame([row]).to_csv(BALANCE, mode="a", header=not BALANCE.exists(), index=False)


def standings(acct, prices):
    """Accounts ranked by balance, as text lines."""
    v = value(acct, prices)
    start = acct["start_cash"]
    return [f"{acct['names'][a]}: ${x:,.2f} ({x / start - 1:+.2%})" for a, x in sorted(v.items(), key=lambda kv: -kv[1])]


def milestones(notify_fn):
    """Send a weekly update during the trial, and write the final report once it ends. Returns a message or None."""
    acct = load()
    if acct is None or acct.get("final_report"):
        return None
    prices = data.prices(config.SYMBOLS)
    opened, ends = pd.Timestamp(acct["opened"]), pd.Timestamp(acct["ends"])
    now = pd.Timestamp.now(tz="UTC")
    if now >= ends:
        snapshot(acct, prices)
        path = final_report(acct, prices)
        acct["final_report"] = str(path)
        _save(acct)
        notify_fn("Paper trading trial finished", "Final standings after 4 weeks:\n" + "\n".join(standings(acct, prices))
                  + f"\nReport: {path.name}")
        return f"trial finished, report {path}"
    week = int((now - opened) / pd.Timedelta("7D"))
    if week > acct.get("weeks_reported", 0):
        acct["weeks_reported"] = week
        _save(acct)
        notify_fn(f"Paper trading: week {week} of 4", "\n".join(standings(acct, prices)))
        return f"week {week} update sent"
    return None


def final_report(acct, prices):
    """Write an HTML report of the trial to docs/backTestResult and return its path."""
    v = value(acct, prices)
    start = acct["start_cash"]
    t = trades()
    hist = balance_history()
    rows = []
    for a, name in acct["names"].items():
        series = hist[a] if a in hist else pd.Series(dtype=float)
        dd = float((series / series.cummax() - 1).min()) if len(series) else float("nan")
        rows.append(f"<tr><td>{name}</td><td>${v[a]:,.2f}</td><td>{v[a] / start - 1:+.2%}</td><td>{dd:+.1%}</td>"
                    f"<td>{int((t['account'] == a).sum())}</td></tr>")
    opened, ends = pd.Timestamp(acct["opened"]), pd.Timestamp(acct["ends"])
    best = max(v, key=v.get)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Trading Trial</title><style>
:root {{ --bg:#fff; --text:#0a0b0d; --muted:#5b616e; --border:#dee1e7; --surface:#f7f8f9; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0a0b0d; --text:#fff; --muted:#8a919e; --border:#2a2c31; --surface:#16171a; }} }}
:root[data-theme="dark"] {{ --bg:#0a0b0d; --text:#fff; --muted:#8a919e; --border:#2a2c31; --surface:#16171a; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.6 Inter, "Segoe UI", sans-serif; }}
main {{ max-width:900px; margin:0 auto; padding:40px 16px; }} .note {{ color:var(--muted); font-size:13px; }}
table {{ border-collapse:collapse; width:100%; }} th, td {{ padding:8px 12px; border-bottom:1px solid var(--border); text-align:right; }}
th {{ background:var(--surface); color:var(--muted); font-weight:500; }} th:first-child, td:first-child {{ text-align:left; }}
</style></head><body><main>
<h1>Paper trading trial: {opened:%b %d} – {ends:%b %d, %Y}</h1>
<p>Each account started with ${start:,.0f} (${start / len(config.SYMBOLS):,.0f} per coin) on {opened:%Y-%m-%d %H:%M} UTC and
traded live until {ends:%Y-%m-%d %H:%M} UTC, with {acct['fee']:.1%} fee and {acct['slippage']:.2%} slippage per trade.
<b>{acct['names'][best]}</b> finished first with ${v[best]:,.2f}.</p>
<table><tr><th>Account</th><th>Final balance</th><th>Return</th><th>Worst drop</th><th>Trades</th></tr>{''.join(rows)}</table>
<p class="note">Four weeks is far too short to judge a strategy: a single month's Sharpe ratio is uncertain by about ±3.5.
Read this as a live sanity check of the backtests, not as proof. Trades and hourly balances are in the paper/ folder.</p>
</main></body></html>"""
    path = config.ROOT / "docs" / "backTestResult" / f"Paper_trading_trial_{opened:%Y-%m-%d}_to_{ends:%Y-%m-%d}.html"
    path.write_text(html, encoding="utf-8")
    return path


def trades():
    if not TRADES.exists():
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.read_csv(TRADES, parse_dates=["time"])


def balance_history():
    if not BALANCE.exists():
        return pd.DataFrame()
    return pd.read_csv(BALANCE, parse_dates=["time"])


def _none():
    return pd.DataFrame(columns=TRADE_COLUMNS)


def _trade_count():
    return len(trades())


def _trades_since(n):
    return trades().iloc[n:]
