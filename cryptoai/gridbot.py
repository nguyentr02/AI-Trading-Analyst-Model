"""Watch the user's Binance spot grid bots: where the price is against the grid, a replay of the grid on past
candles, and an alert when the price leaves the range.

A spot grid places buy orders at evenly spaced prices below the current price and sell orders above it. Each
time the price falls to a level it buys one share of the money, and when it rises one level above it sells that
share for one step's profit. Below the lowest level the bot has bought at every level and simply holds the coin
(it loses like holding, with no grid profit); above the highest it has sold everything and holds cash.

The settings live in grid_bots.json (kept off GitHub). Nothing here places orders.
"""
import json

import numpy as np
import pandas as pd

from . import config, data

BOTS_FILE = config.ROOT / "grid_bots.json"
STATE_FILE = config.LOG_DIR / "grid_state.json"  # last range status per bot, so alerts fire only on changes
NEAR_EDGE = 0.25  # warn when the price is within this share of one grid step of either end


def load_bots():
    if not BOTS_FILE.exists():
        return []
    return json.loads(BOTS_FILE.read_text(encoding="utf-8")).get("bots", [])


def save_bots(bots):
    with config.atomic(BOTS_FILE) as tmp:
        tmp.write_text(json.dumps({"bots": bots}, indent=2), encoding="utf-8")


def levels(bot):
    lo, hi, n = float(bot["lower"]), float(bot["upper"]), int(bot["grids"])
    if bot.get("type", "arithmetic").lower().startswith("geo"):
        return lo * (hi / lo) ** (np.arange(n + 1) / n)
    return np.linspace(lo, hi, n + 1)


def profit_per_grid(bot, fee=config.FEE):
    """Profit of one buy-then-sell round trip, after both fees: (lowest, highest) over the grid."""
    lv = levels(bot)
    gains = lv[1:] / lv[:-1] - 1 - 2 * fee
    return float(gains.min()), float(gains.max())


def status(bot, price):
    """Where the price is against the grid: 'in range', 'below' or 'above', and how far."""
    lv = levels(bot)
    step = lv[1] - lv[0]
    if price < lv[0]:
        return {"where": "below", "text": f"below the range by {lv[0] / price - 1:.1%}: the bot has bought at every "
                f"level and is just holding the coin, no grid profit until the price is back above "
                f"{lv[0]:,.4f}", "distance": price / lv[0] - 1}
    if price > lv[-1]:
        return {"where": "above", "text": f"above the range by {price / lv[-1] - 1:.1%}: the bot has sold "
                f"everything and holds cash, no grid profit until the price is back below {lv[-1]:,.4f}",
                "distance": price / lv[-1] - 1}
    near = ("near the bottom" if price - lv[0] < NEAR_EDGE * step else
            "near the top" if lv[-1] - price < NEAR_EDGE * step else "")
    held = int((lv[1:] > price).sum())  # grids whose sell order is above the price: bought, waiting to sell
    return {"where": "in range", "near": near, "held": held, "distance": 0.0,
            "text": f"in range{(', ' + near) if near else ''}: {held} of {len(lv) - 1} grids hold the coin"}


def replay(bot, candles, invest=1000.0, fee=config.FEE):
    """Run the grid on past candles (high/low path: down first on a red candle, up first on a green one).

    Starts at the first candle's open like Binance: the grids above the price are bought at the start (so they can
    sell on the way up), the ones below wait to buy. Returns totals and the round trips per day."""
    lv = levels(bot)
    n = len(lv) - 1
    per = invest / n
    first = float(candles["open"].iloc[0])
    holding = lv[1:] > first  # grid i spans lv[i]..lv[i+1]; holding = bought, waiting to sell at lv[i+1]
    qty = np.where(holding, per / first, 0.0)
    cash = invest - per * holding.sum() - per * holding.sum() * fee
    grid_profit, trades = 0.0, []
    for t, o, h, lo, c in zip(candles.index, candles["open"], candles["high"], candles["low"], candles["close"]):
        path = (o, h, lo, c) if c < o else (o, lo, h, c)
        for a, b in zip(path[:-1], path[1:]):
            if b < a:  # falling: buy at each empty grid's lower level crossed
                hit = (~holding) & (lv[:-1] >= b) & (lv[:-1] < a)
                if hit.any():
                    qty[hit] = per / lv[:-1][hit]
                    cash -= per * hit.sum() * (1 + fee)
                    holding |= hit
            elif b > a:  # rising: sell each held grid whose upper level is crossed
                hit = holding & (lv[1:] <= b) & (lv[1:] > a)
                if hit.any():
                    got = (qty[hit] * lv[1:][hit]).sum() * (1 - fee)
                    # Binance's "grid profit": one step per round trip; gains on coins bought at the start below
                    # the range count as price gains, not grid profit
                    grid_profit += (qty[hit] * (lv[1:][hit] * (1 - fee) - lv[:-1][hit] * (1 + fee))).sum()
                    cash += got
                    qty[hit] = 0.0
                    holding &= ~hit
                    trades += [t] * int(hit.sum())
    last = float(candles["close"].iloc[-1])
    value = cash + qty.sum() * last
    days = max((candles.index[-1] - candles.index[0]).total_seconds() / 86400, 1e-9)
    inside = ((candles["close"] >= lv[0]) & (candles["close"] <= lv[-1])).mean()
    return {"value": value, "return": value / invest - 1, "grid_profit": grid_profit,
            "round_trips": len(trades), "per_day": len(trades) / days, "days": days,
            "hold_return": last / first - 1, "in_range": float(inside), "holding_grids": int(holding.sum())}


def backtest(bot, days=(1, 3, 7, 30)):
    """The grid's exact settings replayed over the last few windows on 15-minute candles."""
    c = data.drop_open_candle(data.load_cached(bot["symbol"], "15m"), "15m")
    rows = []
    for d in days:
        part = c[c.index >= c.index[-1] - pd.Timedelta(days=d)]
        r = replay(bot, part, invest=float(bot.get("invest") or 1000.0))
        rows.append({"Last": f"{d} days", "Grid return": r["return"], "Grid profit only": r["grid_profit"]
                     / float(bot.get("invest") or 1000.0), "Hold the coin": r["hold_return"],
                     "Round trips": r["round_trips"], "Per day": r["per_day"], "Time in range": r["in_range"]})
    return pd.DataFrame(rows)


OUT_OF_RANGE_HOURS = 12  # suggest a new range after the price has been outside the range this long
REMIND_HOURS = 24  # repeat a change suggestion this often while it still applies
TARGET_STEP = 0.01  # suggested grids: about 1% apart (0.8% profit per grid after Binance's 0.1% fees)


def suggest_range(bot, price):
    """A new range around today's price: the last 7 days' low and high (including the price now) plus 3% either
    side, with grids about 1% apart. A rule of thumb from recent swings, not a tested setting."""
    c = data.drop_open_candle(data.load_cached(bot["symbol"], "15m"), "15m")
    week = c[c.index >= c.index[-1] - pd.Timedelta("7D")]
    lo, hi = min(float(week["low"].min()), price) * 0.97, max(float(week["high"].max()), price) * 1.03
    n = int(max(5, min(100, round(np.log(hi / lo) / np.log(1 + TARGET_STEP)))))
    digits = 4 if price < 10 else 2
    return {"lower": round(lo, digits), "upper": round(hi, digits), "grids": n}


def _reading(symbol):
    """Trend and drop warning for the bot's coin: the altcoin readings, or the main drop warning + daily candles."""
    from . import altcoins, droprisk
    r = (altcoins.load() or {}).get("coins", {}).get(symbol)
    if r:
        return {"above_50d": r["above"]["50"], "from_50d": r["from_50d"], "p_drop": r.get("p_drop")}
    try:
        d = data.drop_open_candle(data.load_cached(symbol, "1d"), "1d")["close"]
        avg = float(d.tail(50).mean())
        p = (droprisk.load() or {}).get("coins", {}).get(symbol, {}).get("p_drop")
        return {"above_50d": bool(d.iloc[-1] > avg), "from_50d": float(d.iloc[-1] / avg - 1), "p_drop": p}
    except Exception:
        return {}


def advice(bot, price, outside_hours=0.0, reading=None):
    """What to change on the bot now, as [(key, title, text)]. Empty when nothing needs changing."""
    from . import droprisk
    reading = reading if reading is not None else _reading(bot["symbol"])
    coin = bot["symbol"].split("/")[0]
    s = status(bot, price)
    out = []
    keep = ("Turn off 'Sell all base coins on stop' before stopping if you want to keep your "
            f"{coin}; with it on, stopping sells all of it at the market price.")
    if s["where"] != "in range" and outside_hours >= OUT_OF_RANGE_HOURS:
        if s["where"] == "below" or not bot.get("trailing_up"):
            new = suggest_range(bot, price)
            side = "below" if s["where"] == "below" else "above"
            out.append(("new_range", f"{coin} grid bot: consider a new range",
                        f"The price ({price:,.4f}) has been {side} your range ({bot['lower']}-{bot['upper']}) for "
                        f"{outside_hours:.0f} hours, so the bot is not trading. Suggested new range: "
                        f"{new['lower']}-{new['upper']} with {new['grids']} grids (about 1% apart; last 7 days' low "
                        f"and high plus 3%; a rule of thumb, not a tested setting). {keep}"))
    if reading and reading.get("above_50d") is False:
        out.append(("trend_down", f"{coin} grid bot: {coin} is in a downtrend",
                    f"{coin}'s daily close is below its 50-day average ({reading['from_50d']:+.0%}). The tested 50-day "
                    f"rule holds cash then. A spot grid is mostly a {coin} position, so consider stopping the bot or "
                    f"setting a stop loss. {keep}"))
    if reading and (reading.get("p_drop") or 0) >= droprisk.SELL_AT:
        stop = bot.get("stop_loss")
        out.append(("drop_high", f"{coin} grid bot: drop warning High",
                    f"The AI gives {coin} a {reading['p_drop']:.0%} chance of a sharp fall soon (High). "
                    + (f"Your stop loss is at {stop}." if stop else
                       "Your bot has no stop loss: consider setting one (Edit -> Stop Loss) a little below the recent "
                       "lows, or stopping the bot.")))
    lo_p, _ = profit_per_grid(bot)
    if lo_p < 0.002:
        out.append(("tiny_grids", f"{coin} grid bot: grids too close",
                    f"The smallest grid earns only {lo_p:.2%} per round trip after fees. Use fewer grids or a wider "
                    "range (Binance's fees are 0.1% each way)."))
    return out


def _bot_state(state, key):
    v = state.get(key)
    return v if isinstance(v, dict) else {"tag": v} if v else {}


def check(notify_fn, prices=None):
    """Every minute in the live service: alert when a bot's price leaves its range, nears an edge, or comes back,
    and when something about the bot should change (advice). Each suggestion is sent when it starts to apply and
    repeated every REMIND_HOURS while it still does."""
    bots = load_bots()
    if not bots:
        return []
    prices = prices or data.prices(list({b["symbol"] for b in bots}))
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    now = pd.Timestamp.now(tz="UTC")
    msgs = []
    for b in bots:
        key = f"{b['symbol']} {b['lower']}-{b['upper']}"
        st = _bot_state(state, key)
        first_seen = key not in state
        px = float(prices[b["symbol"]])
        s = status(b, px)
        coin = b["symbol"].split("/")[0]
        tag = s["where"] + (f" ({s['near']})" if s.get("near") else "")
        if st.get("tag") != tag:
            if not first_seen or s["where"] != "in range":  # no alert for a bot first seen in range
                msg = f"{coin} grid bot ({b['lower']}-{b['upper']}): price {px:,.4f} is {s['text']}."
                notify_fn(f"{coin} grid bot: {tag}", msg)
                msgs.append(msg)
            st["tag"] = tag
        if s["where"] == "in range":
            st.pop("outside_since", None)
        else:
            st.setdefault("outside_since", now.isoformat(timespec="seconds"))
        outside = (now - pd.Timestamp(st["outside_since"])).total_seconds() / 3600 if "outside_since" in st else 0.0
        sent = st.get("advice_sent", {})
        active = advice(b, px, outside)
        for k, title, text in active:
            last = sent.get(k)
            if last is None or (now - pd.Timestamp(last)).total_seconds() >= REMIND_HOURS * 3600:
                notify_fn(title, text)
                msgs.append(text)
                sent[k] = now.isoformat(timespec="seconds")
        st["advice_sent"] = {k: v for k, v in sent.items() if k in {a[0] for a in active}}  # re-alert if it returns
        state[key] = st
    with config.atomic(STATE_FILE) as tmp:
        tmp.write_text(json.dumps(state, indent=2))
    return msgs
