"""Paper trading for US stocks: pretend accounts on Binance's stock perpetual prices, decided once a day.

Accounts (each starts with the same cash, split equally across stocks where it holds several):
- hold      buy every stock in stocks.STOCKS (ETFs excluded) at the start, equal weight, never sell
- index     all in the Nasdaq-100 ETF (QQQ), never sell
- dca       the same stocks as hold, bought in 4 weekly steps, never sell (buying in steps beat a lump sum
            when prices fell, experiments/dca_ai.py)
- trend200  each stock held only while its daily close is above its 200-day average (did NOT beat holding on any
            stock in testing; kept as a live check)
- ai_gate   each stock held unless the stock AI's P(up, 5 days) < 45% (did not beat holding in 2019-2022;
            kept as a live check)
Decisions once a day after the US close (step() is called by the live service). Fills at the Binance perpetual
price with 0.1% fee and 0.05% slippage. Nothing real is traded. Saved in paper_stocks/ (kept off GitHub).
"""
import json
import shutil
from datetime import datetime, timezone

import pandas as pd

from . import config, stockai, stocks

FEE, SLIPPAGE = config.FEE, 0.0005
DIR = config.ROOT / "paper_stocks"
ACCOUNT, TRADES, BALANCE = DIR / "account.json", DIR / "trades.csv", DIR / "balance.csv"
COLUMNS = ["time", "account", "symbol", "side", "price", "quantity", "total", "fee", "reason", "cash_after"]
ACCOUNTS = {"hold": "Hold all (equal weight)", "index": "Nasdaq-100 ETF (QQQ)", "dca": "Buy in 4 weekly steps",
            "trend200": "200-day trend rule", "ai_gate": "AI risk filter"}
PICKS = [t for t in stocks.STOCKS if t not in stocks.ETFS]
TRIAL = pd.Timedelta("28D")
DCA_STEPS = 4


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
    return acct is not None and pd.Timestamp.now(tz="UTC") < pd.Timestamp(acct["ends"])


def _prices():
    return {t: p[0] for t, p in stocks.live_prices().items()}


def open_account(start_cash=1000.0, trial=TRIAL):
    if DIR.exists():
        shutil.move(str(DIR), str(DIR.with_name(f"paper_stocks_archive_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}")))
    DIR.mkdir()
    now = pd.Timestamp.now(tz="UTC")
    acct = {"opened": now.isoformat(timespec="seconds"), "ends": (now + trial).isoformat(timespec="seconds"),
            "start_cash": start_cash, "fee": FEE, "slippage": SLIPPAGE, "names": ACCOUNTS,
            "accounts": {a: {"cash": start_cash, "qty": {}, "cost": {}} for a in ACCOUNTS},
            "dca_done": 0, "last_day": None, "weeks_reported": 0, "final_report": None}
    pd.DataFrame(columns=COLUMNS).to_csv(TRADES, index=False)
    prices = _prices()
    each = start_cash / len(PICKS)
    for t in PICKS:
        _buy(acct, "hold", t, each, prices[t], "hold: equal-weight start")
    _buy(acct, "index", "QQQ", start_cash, prices["QQQ"], "index: all in QQQ")
    _save(acct)
    step(force=True)
    return load()


def _log(account, symbol, side, price, qty, total, reason, cash):
    row = {"time": _now(), "account": account, "symbol": symbol, "side": side, "price": round(price, 6),
           "quantity": round(qty, 8), "total": round(total, 4), "fee": round(total * FEE, 4), "reason": reason,
           "cash_after": round(cash, 4)}
    pd.DataFrame([row], columns=COLUMNS).to_csv(TRADES, mode="a", header=not TRADES.exists(), index=False)


def _buy(acct, account, t, usdt, price, reason):
    a = acct["accounts"][account]
    usdt = min(usdt, a["cash"])
    if usdt < config.MIN_TRADE_USDT:
        return False
    fill = price * (1 + SLIPPAGE)
    qty = usdt * (1 - FEE) / fill
    a["qty"][t] = a["qty"].get(t, 0.0) + qty
    a["cost"][t] = a["cost"].get(t, 0.0) + usdt
    a["cash"] -= usdt
    _log(account, t, "BUY", fill, qty, usdt, reason, a["cash"])
    return True


def _sell_all(acct, account, t, price, reason):
    a = acct["accounts"][account]
    qty = a["qty"].get(t, 0.0)
    if qty * price < config.MIN_TRADE_USDT:
        return False
    fill = price * (1 - SLIPPAGE)
    a["cash"] += qty * fill * (1 - FEE)
    a["qty"][t], a["cost"][t] = 0.0, 0.0
    _log(account, t, "SELL", fill, qty, qty * fill, reason, a["cash"])
    return True


def step(force=False):
    """Once a day after the US close: weekly DCA buys, and the trend / AI accounts' in-or-out decisions."""
    acct = load()
    if not active(acct):
        return 0
    day = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    if acct["last_day"] == day and not force:
        return 0
    before = len(trades())
    prices = _prices()
    ai = (stockai.load() or {}).get("signals", {})
    sleeve = acct["start_cash"] / len(PICKS)
    # Buy in steps: one step on day 0, then one a week.
    weeks = int((pd.Timestamp.now(tz="UTC") - pd.Timestamp(acct["opened"])) / pd.Timedelta("7D"))
    while acct["dca_done"] < DCA_STEPS and acct["dca_done"] <= weeks:
        for t in PICKS:
            _buy(acct, "dca", t, sleeve / DCA_STEPS, prices[t], f"buy step {acct['dca_done'] + 1} of {DCA_STEPS}")
        acct["dca_done"] += 1
    for t in PICKS:
        c = stocks.history(t)
        above = bool(c.iloc[-1] > c.tail(200).mean())
        p = ai.get(t, {}).get("p_up_5d")
        for account, want, why in (
                ("trend200", above, "close above its 200-day average" if above else "close below its 200-day average"),
                ("ai_gate", p is None or p >= 0.45, f"AI P(up, 5 days) {p:.0%}" if p is not None else "no AI reading")):
            holding = acct["accounts"][account]["qty"].get(t, 0.0) > 0
            if want and not holding:
                _buy(acct, account, t, sleeve, prices[t], why)
            elif not want and holding:
                _sell_all(acct, account, t, prices[t], why)
    acct["last_day"] = day
    _save(acct)
    snapshot(acct, prices)
    return len(trades()) - before


def value(acct, prices):
    return {a: s["cash"] + sum(q * prices.get(t, 0.0) for t, q in s["qty"].items()) for a, s in acct["accounts"].items()}


def snapshot(acct=None, prices=None):
    acct = acct or load()
    if acct is None:
        return
    prices = prices or _prices()
    row = {"time": _now(), **{a: round(v, 4) for a, v in value(acct, prices).items()}}
    pd.DataFrame([row]).to_csv(BALANCE, mode="a", header=not BALANCE.exists(), index=False)


def trades():
    return pd.read_csv(TRADES, parse_dates=["time"]) if TRADES.exists() else pd.DataFrame(columns=COLUMNS)


def balance_history():
    return pd.read_csv(BALANCE, parse_dates=["time"]) if BALANCE.exists() else pd.DataFrame()


def milestones(notify_fn):
    """Weekly update during the trial and a final message when it ends."""
    acct = load()
    if acct is None or acct.get("final_report"):
        return None
    prices = _prices()
    v = value(acct, prices)
    lines = [f"{acct['names'][a]}: ${x:,.2f} ({x / acct['start_cash'] - 1:+.2%})" for a, x in sorted(v.items(), key=lambda kv: -kv[1])]
    now, opened = pd.Timestamp.now(tz="UTC"), pd.Timestamp(acct["opened"])
    if now >= pd.Timestamp(acct["ends"]):
        acct["final_report"] = "sent"
        _save(acct)
        notify_fn("Stock paper trading trial finished", "\n".join(lines))
        return "stock trial finished"
    week = int((now - opened) / pd.Timedelta("7D"))
    if week > acct.get("weeks_reported", 0):
        acct["weeks_reported"] = week
        _save(acct)
        notify_fn(f"Stock paper trading: week {week} of 4", "\n".join(lines))
        return f"stock week {week} update sent"
    return None
