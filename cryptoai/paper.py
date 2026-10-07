"""Paper trading: pretend accounts traded live from the moment they are opened, for a fixed trial period.

There are two separate books, each with its own coins, folder and trial dates:
- MAIN  the live coins (config.SYMBOLS), saved in paper/
- ALTS  altcoins (altcoins.TRIAL), saved in paper_alts/. Same accounts and rules (AI Smart also acts on the live
        readings every minute, from 2026-10-06). Experimental: the AI was trained on the main coins and tested only
        on NEAR and ZEC (experiments/altcoins_near_zec.py).

Every fill uses the Binance price at the moment of the trade. At a candle close the decision is made with the
current models and filled straight away; the models retrain afterwards (live.py).

All accounts in a book start with the same balance at the same moment, split into one sleeve per coin (e.g.
$250 each of $1,000), so one coin's fall can't drain the others. Fills use the live Binance price with a 0.1% fee
and 0.05% slippage per trade. Nothing real is bought or sold.

Accounts:
- ai          "Smart" (simulate.SMART): sized by next-1-day confidence, sells half if the 3-day view is still
              up, takes half profit at +10% once confidence fades, no stop-loss. Decides at every 4h close and
              (main book only), between closes, on the live readings once an action has held CONFIRM_MINUTES (then
              a COOLDOWN_MINUTES wait per coin; both guards are a judgement, not backtested). Plus shock dip-buys.
- trend_dip   50-day trend rule (hold while the daily close is above its 50-day average) plus shock dip-buys.
- trend_ai    trend ensemble (close above its 20/50/100/200-day averages, averaged) x AI sizer
              clip(0.5 + 2 x (P(up, next 3 days) - 0.5), 0, 1), rebalanced at each daily close; plus shock
              dip-buys. (experiments/trend_ai_strategies.py: about half the drawdown and half the return.)
- trend       benchmark: the 50-day trend rule alone.
- hold        benchmark: each sleeve bought at the start, never sold.

Shock dip-buy (experiments/shock_dip_buy.py): if a coin closes 10%+ below its close an hour earlier (15-minute
closes), buy with up to half the sleeve from its cash and sell 4 hours later.

When a trial ends, trading stops and a report is written to docs/backTestResult; a weekly update is sent before
that. Each book's folder (kept off GitHub) holds account.json, trades.csv and balance.csv.
"""
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import altcoins, config, data, signals
from .simulate import SMART

FEE, SLIPPAGE = config.FEE, 0.0005
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


@dataclass(frozen=True)
class Book:
    key: str
    title: str
    dir: Path
    symbols: tuple
    live: bool  # AI Smart also trades on the minute-by-minute live readings

    @property
    def account(self):
        return self.dir / "account.json"

    @property
    def trades(self):
        return self.dir / "trades.csv"

    @property
    def balance(self):
        return self.dir / "balance.csv"


MAIN = Book("main", "Top coins", config.ROOT / "paper", tuple(config.SYMBOLS), live=True)
ALTS = Book("alts", "Altcoins", config.ROOT / "paper_alts", tuple(altcoins.TRIAL), live=True)
BOOKS = (MAIN, ALTS)
DIR, ACCOUNT, TRADES, BALANCE = MAIN.dir, MAIN.account, MAIN.trades, MAIN.balance  # older names


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_open(book=MAIN):
    return book.account.exists()


def load(book=MAIN):
    return json.loads(book.account.read_text()) if book.account.exists() else None


def _save(acct, book):
    with config.atomic(book.account) as tmp:
        tmp.write_text(json.dumps(acct, indent=2))


def active(acct):
    """True while the trial is running."""
    return acct is not None and pd.Timestamp.now(tz="UTC") < pd.Timestamp(acct["ends"])


def _sleeve(cash):
    return {"cash": cash, "qty": 0.0, "cost": 0.0, "tp_done": False, "half_sold": False, "shock": None}


def open_account(start_cash=1000.0, trial=TRIAL, book=MAIN):
    """Start every account of `book` now. An existing folder is archived, never deleted."""
    if book.dir.exists():
        shutil.move(str(book.dir), str(book.dir.with_name(f"{book.dir.name}_archive_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}")))
    book.dir.mkdir()
    prices = data.prices(list(book.symbols))
    each = start_cash / len(book.symbols)
    now = pd.Timestamp.now(tz="UTC")
    acct = {"opened": now.isoformat(timespec="seconds"), "ends": (now + trial).isoformat(timespec="seconds"),
            "book": book.key, "symbols": list(book.symbols),
            "start_cash": start_cash, "fee": FEE, "slippage": SLIPPAGE, "names": ACCOUNTS,
            "accounts": {a: {s: _sleeve(each) for s in book.symbols} for a in ACCOUNTS},
            "last_decision": {"ai": None, "daily": None}, "weeks_reported": 0, "final_report": None}
    pd.DataFrame(columns=TRADE_COLUMNS).to_csv(book.trades, index=False)
    for sym, sl in acct["accounts"]["hold"].items():
        _buy(book, "hold", sym, sl, each, prices[sym], "buy & hold benchmark")
    _save(acct, book)
    snapshot(acct, prices, book)
    return acct


# ---------- execution ----------
def _log_trade(book, account, symbol, side, price, qty, total, reason, cash_after):
    row = {"time": _now(), "account": account, "symbol": symbol, "side": side, "price": round(price, 8),
           "quantity": round(qty, 8), "total": round(total, 4), "fee": round(total * FEE, 4), "reason": reason,
           "cash_after": round(cash_after, 4)}
    pd.DataFrame([row], columns=TRADE_COLUMNS).to_csv(book.trades, mode="a", header=not book.trades.exists(), index=False)


def _buy(book, account, symbol, sl, usdt, price, reason):
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
    _log_trade(book, account, symbol, "BUY", fill, qty, usdt, reason, sl["cash"])
    return True


def _sell(book, account, symbol, sl, share, price, reason):
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
    _log_trade(book, account, symbol, "SELL", fill, qty, qty * fill, reason, sl["cash"])
    return True


def _rebalance(book, account, symbol, sl, target, price, reason):
    """Move the sleeve's coin share to `target` (0-1); trades smaller than MIN_TRADE_USDT are skipped."""
    value = sl["cash"] + sl["qty"] * price
    gap = target * value - sl["qty"] * price
    if abs(gap) < config.MIN_TRADE_USDT:
        return False  # too small to be worth a trade (and its fee)
    if gap > 0:
        return _buy(book, account, symbol, sl, gap, price, reason)
    if gap < 0 and sl["qty"] > 0:
        return _sell(book, account, symbol, sl, min(1.0, -gap / (sl["qty"] * price)), price, reason)
    return False


# ---------- AI inputs per book ----------
def _ai_inputs(book):
    """(id of the last closed 4h candle, {symbol: (P(up, next 1 day), P(up, next 3 days))}), or (None, {})."""
    if book is MAIN:
        s1 = signals.current("4h", refresh=False).set_index("symbol")
        s3 = signals.current("1d", refresh=False).set_index("symbol")
        return str(s1["candle_close"].iloc[0]), {s: (float(s1.loc[s, "prob_up"]), float(s3.loc[s, "prob_up"]))
                                                 for s in book.symbols}
    coins = (altcoins.load() or {}).get("coins", {})
    if not all(s in coins and coins[s]["p_up_1d"] is not None for s in book.symbols):
        return None, {}
    return coins[book.symbols[0]]["candle_4h"], {s: (coins[s]["p_up_1d"], coins[s]["p_up_3d"]) for s in book.symbols}


# ---------- AI Smart: at 4h closes and on the live readings ----------
def step_ai(prices=None, book=MAIN):
    """Smart decisions for every coin, from the confirmed signals of the last closed 4h candle."""
    acct = load(book)
    if not active(acct):
        return _none()
    candle, probs = _ai_inputs(book)
    if candle is None or acct["last_decision"]["ai"] == candle:
        return _none()  # no signals yet, or already traded (e.g. the service restarted and caught up)
    prices = prices or data.prices(list(book.symbols))
    before = _trade_count(book)
    for sym, sl in acct["accounts"]["ai"].items():
        p1, p3 = probs[sym]
        if p1 > SMART["sell_below"]:
            sl["half_sold"] = False
        if _execute(book, sym, sl, _decide(sl, p1, p3, prices[sym]), prices[sym], "at 4h close"):
            acct.setdefault("last_trade", {})[sym] = _now()
    acct["last_decision"]["ai"] = candle
    _save(acct, book)
    snapshot(acct, prices, book)
    return _trades_since(before, book)


def step_ai_live(live, book=MAIN):
    """Smart decisions on the live readings (preview.compute() output), acting once they persist."""
    acct = load(book)
    if not book.live or not active(acct) or "4h" not in live.get("models", {}) or "1d" not in live["models"]:
        return _none()
    before = _trade_count(book)
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
            if _execute(book, sym, sl, intents, px, f"live, held {streak['count']} min"):
                acct.setdefault("last_trade", {})[sym] = _now()
            streaks[sym] = {"key": "", "count": 0}
    _save(acct, book)
    return _trades_since(before, book)


def _decide(sl, p1, p3, px):
    """The Smart rules for one sleeve: a list of (kind, amount, reason); nothing is executed here."""
    intents = []
    qty, half_sold = sl["qty"], sl["half_sold"] and p1 <= SMART["sell_below"]
    if qty > 0:
        avg = sl["cost"] / qty
        if not sl["tp_done"] and px >= avg * (1 + SMART["take_profit"]) and p1 < SMART["take_profit_below"]:
            intents.append(("take_profit", 0.5, f"take half profit: up {px / avg - 1:+.1%}, confidence {p1:.0%}"))
            qty *= 0.5
    if qty > 0 and p1 <= SMART["sell_below"] and SMART.get("fee_guard") and _below_fees(sl, px):
        return intents  # the sale wouldn't cover the buy + sell fees: hold (SMART["fee_guard"])
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


def _below_fees(sl, px):
    """True when the price is above the price paid but selling now would not cover the buy + sell fees."""
    avg = sl["cost"] / sl["qty"]  # includes the buy fee and slippage
    paid = avg * (1 - FEE) / (1 + SLIPPAGE)  # the market price at the time of buying
    return px >= paid and px * (1 - SLIPPAGE) * (1 - FEE) < avg


def note_rule_change(text, book=MAIN):
    """Record a rule change in a running trial, so its reports can say when the rules changed."""
    acct = load(book)
    if acct is None:
        return
    acct.setdefault("rule_changes", []).append({"time": _now(), "change": text})
    _save(acct, book)


def _execute(book, sym, sl, intents, px, when):
    traded = False
    for kind, amount, reason in intents:
        if kind == "take_profit":
            traded |= _sell(book, "ai", sym, sl, amount, px, f"{reason} ({when})")
            sl["tp_done"] = True
        elif kind == "sell_half":
            traded |= _sell(book, "ai", sym, sl, amount, px, f"{reason} ({when})")
            sl["half_sold"] = True
        elif kind == "sell_all":
            traded |= _sell(book, "ai", sym, sl, amount, px, f"{reason} ({when})")
        else:
            traded |= _buy(book, "ai", sym, sl, amount, px, f"{reason} ({when})")
    return traded


# ---------- trend accounts: at each daily close ----------
def step_daily(prices=None, book=MAIN):
    """Daily-close decisions for the trend benchmark, trend + dip-buy, and trend x AI."""
    acct = load(book)
    if not active(acct):
        return _none()
    closes = {s: data.drop_open_candle(data.load_cached(s, "1d"), "1d")["close"] for s in book.symbols}
    day = str(closes[book.symbols[0]].index[-1])
    if acct["last_decision"]["daily"] == day:
        return _none()  # today's close was already acted on
    _, probs = _ai_inputs(book)
    if not probs:
        return _none()
    prices = prices or data.prices(list(book.symbols))
    before = _trade_count(book)
    for sym in book.symbols:
        c = closes[sym]
        above = {n: bool(c.iloc[-1] > c.iloc[-n:].mean()) for n in (20, 50, 100, 200)}
        sma50 = 1.0 if above[50] else 0.0
        for account in ("trend", "trend_dip"):
            _rebalance(book, account, sym, acct["accounts"][account][sym], sma50, prices[sym],
                       "close above its 50-day average" if above[50] else "close below its 50-day average")
        p3 = probs[sym][1]
        ens = sum(above.values()) / 4
        sizer = float(np.clip(0.5 + 2 * (p3 - 0.5), 0, 1))
        target = ens * sizer
        _rebalance(book, "trend_ai", sym, acct["accounts"]["trend_ai"][sym], target, prices[sym],
                   f"trend {ens:.0%} x AI sizer {sizer:.0%} (P(up, 3 days) {p3:.0%}) -> {target:.0%} invested")
    acct["last_decision"]["daily"] = day
    _save(acct, book)
    snapshot(acct, prices, book)
    return _trades_since(before, book)


step_trend = step_daily  # older name


# ---------- shock dip-buy: at each 15-minute close ----------
def step_shock(prices=None, book=MAIN):
    """For each trading account: close shock lots after 4 hours; buy a fresh 10%+ one-hour drop."""
    acct = load(book)
    if not active(acct):
        return _none()
    before = _trade_count(book)
    now = pd.Timestamp.now(tz="UTC")
    prices = prices or data.prices(list(book.symbols))
    checked = acct.setdefault("shock_checked", {})
    drops = {}
    for sym in book.symbols:
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
                _log_trade(book, account, sym, "SELL", fill, lot["qty"], lot["qty"] * fill,
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
                    _log_trade(book, account, sym, "BUY", fill, qty, spend,
                               f"shock dip-buy: {drop:+.1%} in 1 hour, sell in 4h", sl["cash"])
    _save(acct, book)
    return _trades_since(before, book)


# ---------- valuation, history, trial milestones ----------
def holdings(sl):
    """(coin quantity, cost) of a sleeve, including an open shock lot."""
    shock = sl.get("shock") or {}
    return sl["qty"] + shock.get("qty", 0.0), sl["cost"] + shock.get("cost", 0.0)


def value(acct, prices):
    """Current value of every account."""
    return {a: sum(sl["cash"] + holdings(sl)[0] * prices[s] for s, sl in sleeves.items())
            for a, sleeves in acct["accounts"].items()}


def snapshot(acct=None, prices=None, book=MAIN):
    """Append every account's balance to the balance history."""
    acct = acct or load(book)
    if acct is None:
        return
    prices = prices or data.prices(list(book.symbols))
    row = {"time": _now(), **{a: round(v, 4) for a, v in value(acct, prices).items()}}
    pd.DataFrame([row]).to_csv(book.balance, mode="a", header=not book.balance.exists(), index=False)


def missing_hours(hist, end, gap=pd.Timedelta("75min")):
    """Whole hours with no balance point: inside any gap longer than `gap`, and after the last point up to `end`."""
    times = pd.to_datetime(hist["time"], utc=True, format="ISO8601").sort_values().tolist() + [end]
    hours = []
    for a, b in zip(times[:-1], times[1:]):
        if b - a > gap:
            hours += list(pd.date_range(a.ceil("h") + pd.Timedelta("1h") if a.ceil("h") == a else a.ceil("h"),
                                        b - pd.Timedelta("30min"), freq="h"))
    return pd.DatetimeIndex(hours)


def holdings_at(trades_df, t, accounts, start_cash, per_sleeve):
    """Each account's (cash, {symbol: coins}) at time t, rebuilt from the trade log. With per_sleeve, each coin has
    its own cash sleeve (crypto books); otherwise one cash balance per account (stock book)."""
    done = trades_df[pd.to_datetime(trades_df["time"], utc=True, format="ISO8601") <= t]
    out = {}
    for a, symbols in accounts.items():
        mine = done[done["account"] == a]
        qty = {s: float(mine.loc[(mine["symbol"] == s) & (mine["side"] == "BUY"), "quantity"].sum()
                     - mine.loc[(mine["symbol"] == s) & (mine["side"] == "SELL"), "quantity"].sum()) for s in symbols}
        if per_sleeve:
            cash = 0.0
            for s in symbols:
                last = mine[mine["symbol"] == s]
                cash += float(last["cash_after"].iloc[-1]) if len(last) else start_cash / len(symbols)
        else:
            cash = float(mine["cash_after"].iloc[-1]) if len(mine) else start_cash
        out[a] = (cash, qty)
    return out


def fill_gaps(balance_path, hist, trades_df, accounts, start_cash, per_sleeve, end, price_at):
    """Add an hourly balance point for every missing hour (rebuilt holdings x real hourly closes), keep the file in
    time order, and return how many were added. `price_at(symbols, hours)` returns a DataFrame of prices."""
    hours = missing_hours(hist, end)
    if len(hours) == 0:
        return 0
    symbols = sorted({s for syms in accounts.values() for s in syms})
    prices = price_at(symbols, hours)
    rows = []
    for h in hours:
        held = holdings_at(trades_df, h, accounts, start_cash, per_sleeve)
        row = {"time": h.isoformat(timespec="seconds")}
        for a, (cash, qty) in held.items():
            vals = [q * prices.at[h, s] for s, q in qty.items() if abs(q) > 1e-12]
            row[a] = round(cash + sum(vals), 4) if not any(pd.isna(v) for v in vals) else np.nan
        if not any(pd.isna(v) for k, v in row.items() if k != "time"):
            rows.append(row)
    if not rows:
        return 0
    merged = pd.concat([hist, pd.DataFrame(rows)], ignore_index=True)
    merged["_t"] = pd.to_datetime(merged["time"], utc=True, format="ISO8601")
    merged = merged.sort_values("_t").drop(columns="_t")
    with config.atomic(balance_path) as tmp:
        merged.to_csv(tmp, index=False)
    return len(rows)


def backfill_balance(book=MAIN):
    """After downtime: an hourly balance point for every hour the service missed, from the holdings at that hour
    (rebuilt from the trade log) and real hourly closes. Returns how many points were added."""
    acct = load(book)
    hist = balance_history(book)
    if acct is None or hist.empty:
        return 0
    end = min(pd.Timestamp.now(tz="UTC"), pd.Timestamp(acct["ends"]))

    def price_at(symbols, hours):
        out = {}
        for s in symbols:
            c = data.update(s, "1h")["close"]
            c.index = c.index + pd.Timedelta("1h")  # an hourly close is the price at the end of the hour
            out[s] = c.reindex(hours, method="ffill")
        return pd.DataFrame(out, index=hours)

    accounts = {a: list(sleeves) for a, sleeves in acct["accounts"].items()}
    return fill_gaps(book.balance, hist, trades(book), accounts, acct["start_cash"], True, end, price_at)


def standings(acct, prices):
    """Accounts ranked by balance, as text lines."""
    v = value(acct, prices)
    start = acct["start_cash"]
    return [f"{acct['names'][a]}: ${x:,.2f} ({x / start - 1:+.2%})" for a, x in sorted(v.items(), key=lambda kv: -kv[1])]


def milestones(notify_fn, book=MAIN):
    """Send a weekly update during the trial, and write the final report once it ends. Returns a message or None."""
    acct = load(book)
    if acct is None or acct.get("final_report"):
        return None
    prices = data.prices(list(book.symbols))
    opened, ends = pd.Timestamp(acct["opened"]), pd.Timestamp(acct["ends"])
    now = pd.Timestamp.now(tz="UTC")
    label = "Paper trading" if book is MAIN else f"{book.title} paper trading"
    if now >= ends:
        snapshot(acct, prices, book)
        path = final_report(acct, prices, book)
        acct["final_report"] = str(path)
        _save(acct, book)
        notify_fn(f"{label} trial finished", "Final standings after 4 weeks:\n" + "\n".join(standings(acct, prices))
                  + f"\nReport: {path.name}")
        return f"trial finished, report {path}"
    week = int((now - opened) / pd.Timedelta("7D"))
    if week > acct.get("weeks_reported", 0):
        acct["weeks_reported"] = week
        _save(acct, book)
        notify_fn(f"{label}: week {week} of 4", "\n".join(standings(acct, prices)))
        return f"week {week} update sent"
    return None


def final_report(acct, prices, book=MAIN):
    """Write an HTML report of the trial to docs/backTestResult and return its path."""
    v = value(acct, prices)
    start = acct["start_cash"]
    t = trades(book)
    hist = balance_history(book)
    rows = []
    for a, name in acct["names"].items():
        series = hist[a] if a in hist else pd.Series(dtype=float)
        dd = float((series / series.cummax() - 1).min()) if len(series) else float("nan")
        rows.append(f"<tr><td>{name}</td><td>${v[a]:,.2f}</td><td>{v[a] / start - 1:+.2%}</td><td>{dd:+.1%}</td>"
                    f"<td>{int((t['account'] == a).sum())}</td></tr>")
    opened, ends = pd.Timestamp(acct["opened"]), pd.Timestamp(acct["ends"])
    best = max(v, key=v.get)
    coins = ", ".join(s.split("/")[0] for s in book.symbols)
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
<h1>{book.title} paper trading trial: {opened:%b %d} – {ends:%b %d, %Y}</h1>
<p>Coins: {coins}. Each account started with ${start:,.0f} (${start / len(book.symbols):,.0f} per coin) on
{opened:%Y-%m-%d %H:%M} UTC and traded live until {ends:%Y-%m-%d %H:%M} UTC, with {acct['fee']:.1%} fee and
{acct['slippage']:.2%} slippage per trade. <b>{acct['names'][best]}</b> finished first with ${v[best]:,.2f}.</p>
<table><tr><th>Account</th><th>Final balance</th><th>Return</th><th>Worst drop</th><th>Trades</th></tr>{''.join(rows)}</table>
{''.join(f'<p class="note">Rule change on {c["time"][:16].replace("T", " ")} UTC: {c["change"]}</p>' for c in acct.get("rule_changes", []))}
<p class="note">Four weeks is far too short to judge a strategy: a single month's Sharpe ratio is uncertain by about ±3.5.
Read this as a live sanity check of the backtests, not as proof. Trades and hourly balances are in the {book.dir.name}/ folder.</p>
</main></body></html>"""
    name = "Paper_trading_trial" if book is MAIN else f"Paper_trading_trial_{book.key}"
    path = config.ROOT / "docs" / "backTestResult" / f"{name}_{opened:%Y-%m-%d}_to_{ends:%Y-%m-%d}.html"
    path.write_text(html, encoding="utf-8")
    return path


def trades(book=MAIN):
    if not book.trades.exists():
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.read_csv(book.trades, parse_dates=["time"])


def balance_history(book=MAIN):
    if not book.balance.exists():
        return pd.DataFrame()
    return pd.read_csv(book.balance, parse_dates=["time"])


def _none():
    return pd.DataFrame(columns=TRADE_COLUMNS)


def _trade_count(book):
    return len(trades(book))


def _trades_since(n, book):
    return trades(book).iloc[n:]
