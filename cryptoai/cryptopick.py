"""The crypto verdict and top pick, the crypto version of the stock buy-for-hold list (stocks.buy_list).

The tested crypto core is the 50-day trend rule: hold a coin while its daily close is above its 50-day average,
cash while below. So right now:
- coins above their average are ones the rule would hold (buy in steps); none above -> "don't buy crypto now";
- coins you hold that are below their average are ones the rule would have sold; a High drop warning on a held
  coin is a reason to sell some.
Top pick (experiments/crypto_top_pick.py, 2026-10-07): if only one coin, the one with the best 90-day return among
those above their average, reviewed monthly. Chosen on 2022-2024 but it did not beat the 50-day rule list in both
periods and its worst fall was far deeper, so it is shown only as "if you only buy one".
"""
import json

import numpy as np
import pandas as pd

from . import config, data, droprisk

STATE = config.LOG_DIR / "crypto_top_pick.json"  # held pick per group, kept until the monthly review
RECORD = {"pick_2224": 1.07, "list_2224": 1.18, "pick_2526": 3.25, "list_2526": 0.72,
          "pick_worst": -0.70, "list_worst": -0.24}


def trend_table(symbols, live=None):
    """Per coin: price (live if given), 50-day average, above it?, 90-day return and volatility."""
    live = live or {}
    rows = []
    for s in symbols:
        try:
            d = data.drop_open_candle(data.load_cached(s, "1d"), "1d")["close"]
        except Exception:
            continue
        if len(d) < 91:
            continue
        price = float(live.get(s) or d.iloc[-1])
        ma50 = float(d.tail(50).mean())
        rows.append({"symbol": s, "price": price, "ma50": ma50, "from_50d": price / ma50 - 1,
                     "in": price > ma50, "mom90": price / float(d.iloc[-91]) - 1,
                     "vol90": float(np.log(d).diff().tail(90).std() * np.sqrt(365))})
    return pd.DataFrame(rows)


def top_pick(table):
    """Today's best option: the best 90-day return among the coins above their 50-day average."""
    cand = table[table["in"]] if len(table) else table
    return None if not len(cand) else cand.loc[cand["mom90"].idxmax()]


def last_review():
    """The latest monthly review that has happened: the start of this month (UTC daily closes)."""
    return pd.Timestamp.now(tz="UTC").normalize().replace(day=1)


def next_review():
    return last_review() + pd.offsets.MonthBegin(1)


def held_top_pick(group, table):
    """(held pick, today's best, state). Held until the next monthly review; switching daily did worse for stocks
    and is not tested better for crypto, so the daily check is only reported. Re-picked early if the held coin
    falls below its 50-day average (the rule itself would sell it)."""
    best = top_pick(table)
    try:
        all_state = json.loads(STATE.read_text())
    except (OSError, ValueError):
        all_state = {}
    state = all_state.get(group, {})
    review = str(last_review().date())
    still_in = set(table.loc[table["in"], "symbol"]) if len(table) else set()
    if best is not None and (state.get("review") != review or state.get("symbol") not in still_in):
        state = {"symbol": best["symbol"], "review": review, "price": float(best["price"]),
                 "chosen": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
        all_state[group] = state
        with config.atomic(STATE) as tmp:
            tmp.write_text(json.dumps(all_state, indent=1))
    held = table[table["symbol"] == state.get("symbol")] if len(table) else table
    return (held.iloc[0] if len(held) else best), best, state


def portfolio_actions(holdings, table, live=None):
    """What the tested rules say about the coins you hold: [(symbol, 'sell'|'sell some', reason)]."""
    held = {}
    for h in holdings:
        held[h["symbol"]] = held.get(h["symbol"], 0.0) + float(h["amount"])
    t = table.set_index("symbol") if len(table) else table
    drop = {s: r["p_drop"] for s, r in (droprisk.load() or {}).get("coins", {}).items()}
    try:
        from . import altcoins
        drop.update({s: r.get("p_drop") for s, r in (altcoins.load() or {}).get("coins", {}).items()})
    except Exception:
        pass
    out = []
    for s, amount in held.items():
        if amount <= 0 or s not in t.index:
            continue
        r = t.loc[s]
        if r["price"] * amount < config.MIN_TRADE_USDT:
            continue
        if not r["in"]:
            out.append((s, "sell", f"below its 50-day average ({r['from_50d']:+.1%}); the tested rule holds cash"))
        elif (drop.get(s) or 0) >= droprisk.SELL_AT:
            out.append((s, "sell some", f"drop warning High ({drop[s]:.0%} chance of a sharp fall)"))
    return out
