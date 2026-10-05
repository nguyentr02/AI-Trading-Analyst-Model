"""Paper trading: a pretend account the AI trades live, from the moment it is opened.

- Starting balance split equally into one sleeve per coin (e.g. $250 each of $1,000), so one coin's fall
  can't drain the others.
- The "Smart" strategy (simulate.SMART) decides at every 4h candle close from the confirmed signals, and
  between closes on the live readings refreshed every minute (see step_ai_live):
  position size by next-1-day confidence, sell half if the 3-day view is still up, take half profit at
  +10% once confidence fades, no stop-loss. Unlike the backtest it buys at market (no 1%-lower limit
  orders; those filled for only 2% of buys in testing).
- Two benchmarks are tracked from the same start: buy & hold (each sleeve bought at the start) and the
  50-day trend rule (hold a coin while its daily close is above its 50-day average, checked daily).
- Fills use the live Binance price with a 0.1% fee and 0.05% slippage per trade.

Everything is saved in paper/ (kept off GitHub): account.json (balances and positions), trades.csv (every
trade with its reason) and balance.csv (the balance over time).
"""
import json
from datetime import datetime, timezone

import pandas as pd

from . import config, data, signals
from .simulate import SMART

FEE, SLIPPAGE = config.FEE, 0.0005
DIR = config.ROOT / "paper"
ACCOUNT, TRADES, BALANCE = DIR / "account.json", DIR / "trades.csv", DIR / "balance.csv"
TRADE_COLUMNS = ["time", "account", "symbol", "side", "price", "quantity", "total", "fee", "reason", "cash_after"]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_open():
    return ACCOUNT.exists()


def load():
    return json.loads(ACCOUNT.read_text()) if ACCOUNT.exists() else None


def _save(acct):
    with config.atomic(ACCOUNT) as tmp:
        tmp.write_text(json.dumps(acct, indent=2))


def _sleeve(cash):
    return {"cash": cash, "qty": 0.0, "cost": 0.0, "tp_done": False, "half_sold": False}


def open_account(start_cash=1000.0):
    """Open the pretend account now. Refuses if one is already open (close it first to start over)."""
    if is_open():
        raise RuntimeError("A paper account is already open; its history is in paper/.")
    DIR.mkdir(exist_ok=True)
    prices = data.prices(config.SYMBOLS)
    each = start_cash / len(config.SYMBOLS)
    acct = {
        "opened": _now(), "start_cash": start_cash, "fee": FEE, "slippage": SLIPPAGE,
        "strategy": "Smart (sized by confidence), decided at every 4h close",
        "accounts": {
            "ai": {s: _sleeve(each) for s in config.SYMBOLS},
            "trend": {s: _sleeve(each) for s in config.SYMBOLS},
        },
        "hold": {s: {"qty": each * (1 - FEE) / (prices[s] * (1 + SLIPPAGE)), "start_price": prices[s]}
                 for s in config.SYMBOLS},
        "last_decision": {"ai": None, "trend": None},
    }
    _save(acct)
    pd.DataFrame(columns=TRADE_COLUMNS).to_csv(TRADES, index=False)
    for s in config.SYMBOLS:
        _log_trade("hold", s, "BUY", prices[s] * (1 + SLIPPAGE), acct["hold"][s]["qty"], each, "buy & hold benchmark", 0.0)
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
        return
    fill = price * (1 + SLIPPAGE)
    qty = usdt * (1 - FEE) / fill
    if sl["qty"] == 0:
        sl["tp_done"] = False
    sl["qty"] += qty
    sl["cost"] += usdt
    sl["cash"] -= usdt
    _log_trade(account, symbol, "BUY", fill, qty, usdt, reason, sl["cash"])


def _sell(account, symbol, sl, share, price, reason):
    qty = sl["qty"] * share
    if qty * price < config.MIN_TRADE_USDT:
        share, qty = 1.0, sl["qty"]  # don't leave dust behind
    fill = price * (1 - SLIPPAGE)
    proceeds = qty * fill * (1 - FEE)
    sl["cash"] += proceeds
    sl["qty"] -= qty
    sl["cost"] *= 1 - share
    if sl["qty"] <= 1e-12:
        sl["qty"], sl["cost"] = 0.0, 0.0
    _log_trade(account, symbol, "SELL", fill, qty, qty * fill, reason, sl["cash"])


# ---------- the two traded accounts ----------
def step_ai(prices=None):
    """Smart decisions for every coin, from the confirmed signals of the last closed 4h candle."""
    acct = load()
    if acct is None:
        return []
    s1 = signals.current("4h", refresh=False).set_index("symbol")
    s4 = signals.current("4h_next", refresh=False).set_index("symbol")
    s3 = signals.current("1d", refresh=False).set_index("symbol")
    candle = str(s1["candle_close"].iloc[0])
    if acct["last_decision"]["ai"] == candle:
        return []  # this candle was already traded (e.g. the service restarted and caught up)
    prices = prices or data.prices(config.SYMBOLS)
    before = _trade_count()
    for sym, sl in acct["accounts"]["ai"].items():
        p1, p3 = float(s1.loc[sym, "prob_up"]), float(s3.loc[sym, "prob_up"])
        if p1 > SMART["sell_below"]:
            sl["half_sold"] = False  # confidence recovered: a later dip may sell half again
        if _execute(sym, sl, _decide(sl, p1, p3, prices[sym]), prices[sym], "at 4h close"):
            acct.setdefault("last_trade", {})[sym] = _now()
    acct["last_decision"]["ai"] = candle
    _save(acct)
    snapshot(acct, prices)
    return _trades_since(before)


# Trading at any moment on the live (provisional) readings, refreshed every minute. To keep a reading that
# hovers around a threshold from trading back and forth, an action must be wanted for CONFIRM_MINUTES in a
# row, and each coin waits COOLDOWN_MINUTES after a trade. Both numbers are a judgement, not backtested:
# there is no minute-by-minute history of the live readings to test them on.
CONFIRM_MINUTES, COOLDOWN_MINUTES = 10, 60


def step_ai_live(live):
    """Smart decisions on the live readings (preview.compute() output), acting once they persist."""
    acct = load()
    if acct is None or "4h" not in live.get("models", {}) or "1d" not in live["models"]:
        return []
    before = _trade_count()
    now = pd.Timestamp.now(tz="UTC")
    streaks = acct.setdefault("live_streak", {})
    for sym, sl in acct["accounts"]["ai"].items():
        p1 = live["models"]["4h"][sym]["prob_up"]
        p3 = live["models"]["1d"][sym]["prob_up"]
        px = live["models"]["4h"][sym]["price"]
        if p1 > SMART["sell_below"]:
            sl["half_sold"] = False
        intents = _decide(sl, p1, p3, px)
        key = "|".join(i[0] for i in intents)
        streak = streaks.get(sym, {"key": "", "count": 0})
        streak = {"key": key, "count": streak["count"] + 1 if key and key == streak["key"] else (1 if key else 0)}
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
    """The Smart rules for one coin's sleeve: a list of (kind, amount, reason); nothing is executed here."""
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
    """Carry out _decide's intents on the sleeve. Returns True if anything was traded."""
    traded = False
    for kind, amount, reason in intents:
        if kind == "take_profit":
            _sell("ai", sym, sl, amount, px, f"{reason} ({when})")
            sl["tp_done"] = True
        elif kind == "sell_half":
            _sell("ai", sym, sl, amount, px, f"{reason} ({when})")
            sl["half_sold"] = True
        elif kind == "sell_all":
            _sell("ai", sym, sl, amount, px, f"{reason} ({when})")
        else:
            _buy("ai", sym, sl, amount, px, f"{reason} ({when})")
        traded = True
    return traded


def step_trend(prices=None):
    """Trend rule benchmark: hold each coin while its last daily close is above its 50-day average."""
    acct = load()
    if acct is None:
        return []
    closes = {s: data.drop_open_candle(data.load_cached(s, "1d"), "1d")["close"] for s in config.SYMBOLS}
    day = str(closes[config.SYMBOLS[0]].index[-1])
    if acct["last_decision"]["trend"] == day:
        return []  # today's close was already acted on
    prices = prices or data.prices(config.SYMBOLS)
    before = _trade_count()
    for sym, sl in acct["accounts"]["trend"].items():
        d = closes[sym]
        above = d.iloc[-1] > d.iloc[-50:].mean()
        if above and sl["qty"] == 0:
            _buy("trend", sym, sl, sl["cash"], prices[sym], "close above its 50-day average")
        elif not above and sl["qty"] > 0:
            _sell("trend", sym, sl, 1.0, prices[sym], "close below its 50-day average")
    acct["last_decision"]["trend"] = day
    _save(acct)
    snapshot(acct, prices)
    return _trades_since(before)


# ---------- valuation and history ----------
def value(acct, prices):
    """Current value of the AI account, the trend benchmark and buy & hold."""
    def sleeves(name):
        return sum(sl["cash"] + sl["qty"] * prices[s] for s, sl in acct["accounts"][name].items())
    hold = sum(h["qty"] * prices[s] for s, h in acct["hold"].items())
    return {"ai": sleeves("ai"), "trend": sleeves("trend"), "hold": hold}


def snapshot(acct=None, prices=None):
    """Append the current balances to the balance history."""
    acct = acct or load()
    if acct is None:
        return
    prices = prices or data.prices(config.SYMBOLS)
    v = value(acct, prices)
    ai = acct["accounts"]["ai"]
    row = {"time": _now(), "ai": round(v["ai"], 4), "trend": round(v["trend"], 4), "hold": round(v["hold"], 4),
           "ai_cash": round(sum(sl["cash"] for sl in ai.values()), 4)}
    pd.DataFrame([row]).to_csv(BALANCE, mode="a", header=not BALANCE.exists(), index=False)


def trades():
    if not TRADES.exists():
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.read_csv(TRADES, parse_dates=["time"])


def balance_history():
    if not BALANCE.exists():
        return pd.DataFrame()
    return pd.read_csv(BALANCE, parse_dates=["time"])


def _trade_count():
    return len(trades())


def _trades_since(n):
    return trades().iloc[n:]
