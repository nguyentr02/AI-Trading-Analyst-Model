"""Turn the AI's signals into buy / sell / hold advice for your own portfolio, and alert when it changes.

The rules are the ones tested in the backtest and the simulator, using the "next 1 day" model for
direction and the "next 4 hours" model only as a timing hint:

- You hold a coin and its next-1-day P(up) is at or below EXIT_PROB (48%)  -> SELL it, unless you are up by
  less than the buy + sell fees (selling would lock in a loss) and the drop warning is not High -> HOLD.
- You hold a coin otherwise                                                 -> HOLD.
- You don't hold a coin and its next-1-day P(up) is at or above ENTER_PROB  -> BUY, with a suggested amount.
- Otherwise                                                                 -> WAIT.

Suggested buy amounts split the spare cash (plus what the suggested sells would free up) equally across
the coins to buy, with no more than MAX_PER_COIN of the whole portfolio in any one coin. These are
estimates with a small edge, not financial advice.
"""
import json

import pandas as pd

from . import config, data, droprisk, notify, portfolio, signals

DIRECTION, TIMING = "4h", "4h_next"  # model names: next 1 day, next 4 hours


def advise(holdings, cash, sig_direction, sig_timing, prices, drop=None):
    """Advice per tracked coin (and for any held coin the AI doesn't track), as a DataFrame.

    `drop` ({symbol: P(sharp drop)}) lets a High drop warning override the fee guard below.
    """
    drop = drop or {}
    held = {}
    for h in holdings:
        held.setdefault(h["symbol"], {"amount": 0.0, "cost": 0.0})
        held[h["symbol"]]["amount"] += float(h["amount"])
        held[h["symbol"]]["cost"] += float(h["amount"]) * float(h["avg_cost"])
    value = {s: v["amount"] * prices.get(s, 0.0) for s, v in held.items()}
    total = cash + sum(value.values())

    rows, buys = [], []
    for sym in config.SYMBOLS:
        p1, p4 = float(sig_direction.loc[sym, "prob_up"]), float(sig_timing.loc[sym, "prob_up"])
        price = prices[sym]
        row = {"symbol": sym, "price": price, "p_1day": p1, "p_4h": p4, "held_value": value.get(sym, 0.0),
               "amount_usdt": 0.0, "amount_coin": 0.0}
        if value.get(sym, 0.0) >= config.MIN_TRADE_USDT:
            avg = held[sym]["cost"] / held[sym]["amount"]
            pnl = price / avg - 1
            # Fee guard (simulate.SMART["fee_guard"]): above the price paid, but selling wouldn't cover the buy +
            # sell fees -> hold, unless the drop warning is High (a stop is needed).
            fees = (1 + config.FEE) / (1 - config.FEE) - 1
            in_fee_zone = 0 <= pnl < fees and drop.get(sym, 0.0) < droprisk.SELL_AT
            if p1 <= config.EXIT_PROB and in_fee_zone:
                row.update(action="HOLD", reason=f"The AI expects a fall (P(up) {p1:.1%}), but you are only "
                                                 f"{pnl:+.2%} on this coin: selling now wouldn't cover the buy and "
                                                 f"sell fees (~{fees:.1%}). Holding unless the drop risk turns High.")
            elif p1 <= config.EXIT_PROB:
                row.update(action="SELL", amount_usdt=value[sym], amount_coin=held[sym]["amount"],
                           reason=f"The AI expects a fall over the next day (P(up) {p1:.1%}). "
                                  f"You are {pnl:+.1%} on this coin.")
                if p4 >= config.ENTER_PROB:
                    row["timing"] = "A short bounce is likely in the next 4 hours; selling a few hours later may get a better price."
            else:
                row.update(action="HOLD", reason=f"No sell signal (next-1-day P(up) {p1:.1%}). You are {pnl:+.1%} on this coin.")
        elif p1 >= config.ENTER_PROB:
            row.update(action="BUY", reason=f"The AI expects a rise over the next day (P(up) {p1:.1%}).")
            if p4 <= config.EXIT_PROB:
                row["timing"] = "A dip is likely in the next 4 hours; buying a few hours later may get a better price."
            buys.append(row)
        else:
            row.update(action="WAIT", reason=f"No buy signal yet (next-1-day P(up) {p1:.1%}).")
        rows.append(row)

    # Size the buys: spare cash plus the proceeds of the suggested sells, split equally, capped per coin.
    available = cash + sum(r["amount_usdt"] for r in rows if r["action"] == "SELL") * (1 - config.FEE)
    cap = config.MAX_PER_COIN * total
    for i, r in enumerate(sorted(buys, key=lambda r: -r["p_1day"])):
        amount = min(available / (len(buys) - i), cap)
        if amount < config.MIN_TRADE_USDT:
            r.update(action="WAIT", reason=r["reason"] + " Not enough spare cash to buy; add cash on the Portfolio page.")
            continue
        r.update(amount_usdt=amount, amount_coin=amount * (1 - config.FEE) / r["price"])
        available -= amount

    for sym, v in held.items():
        if sym not in config.SYMBOLS and v["amount"] > 0:
            rows.append({"symbol": sym, "action": "NOT TRACKED", "price": prices.get(sym), "held_value": value[sym],
                         "reason": "The AI only analyses BTC, ETH, BNB and SOL.", "amount_usdt": 0.0, "amount_coin": 0.0})
    df = pd.DataFrame(rows)
    if "timing" not in df:
        df["timing"] = None
    return df


def current(refresh=False):
    """Advice for the saved portfolio, using the latest signals."""
    holdings, cash = portfolio.load(), portfolio.load_cash()
    sig1 = signals.current(DIRECTION, refresh).set_index("symbol")
    sig4 = signals.current(TIMING, refresh).set_index("symbol")
    syms = sorted(set(config.SYMBOLS) | {h["symbol"] for h in holdings})
    try:
        prices = data.prices(syms)
    except Exception:  # offline: fall back to the last candle close
        prices = sig1["price"].to_dict()
    drop = {s: r["p_drop"] for s, r in (droprisk.load() or {}).get("coins", {}).items()}
    return advise(holdings, cash, sig1, sig4, prices, drop)


def check_and_notify(refresh=False):
    """Compute advice, and send an alert for every coin whose action changed since the last check."""
    advice = current(refresh)
    previous = _load_state()
    changed = advice[[previous.get(r.symbol) != r.action for r in advice.itertuples()]]
    changed = changed[changed["action"].isin(["BUY", "SELL"])]
    if len(changed):
        title = "Crypto AI: " + ", ".join(f"{r.action} {r.symbol.split('/')[0]}" for r in changed.itertuples())
        notify.send(title, message(changed))
    _save_state({r.symbol: r.action for r in advice.itertuples()})
    return advice, changed


def message(rows):
    lines = []
    for r in rows.itertuples():
        coin = r.symbol.split("/")[0]
        if r.action == "BUY":
            lines.append(f"BUY {coin}: about ${r.amount_usdt:,.0f} (~{r.amount_coin:.4f} {coin}) at ${r.price:,.2f}.")
        elif r.action == "SELL":
            lines.append(f"SELL {coin}: all {r.amount_coin:.4f} {coin} (~${r.amount_usdt:,.0f}) at ${r.price:,.2f}.")
        lines.append(f"  {r.reason}")
        if isinstance(r.timing, str):
            lines.append(f"  Timing: {r.timing}")
    lines.append("Estimates with a small edge, not financial advice.")
    return "\n".join(lines)


def _load_state():
    if not config.ADVICE_STATE.exists():
        return {}
    try:
        return json.loads(config.ADVICE_STATE.read_text())
    except ValueError:
        return {}


def _save_state(state):
    with config.atomic(config.ADVICE_STATE) as tmp:
        tmp.write_text(json.dumps(state, indent=2))
