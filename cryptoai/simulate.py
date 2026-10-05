"""Paper-trade one coin over a past period, starting with a cash balance, the way the AI would have.

    python -m cryptoai simulate BNB/USDT 2026-01-01 2026-06-01 --cash 1000
    python -m cryptoai simulate BNB/USDT 2026-01-01 2026-06-01 --stop trailing --stop-pct 0.10

No look-ahead: at the start of every month the models are retrained using only candles whose outcome
was already known before that month, then they trade that month. (The live service retrains at every
candle close; monthly is a close, much faster stand-in.) Spot trading: the AI holds either the coin or
cash, can buy or sell at any candle close, and pays the exchange fee on every trade.

Optional stop-loss: "fixed" sells if the price falls `pct` below the buy price, "trailing" if it falls
`pct` below the highest close since buying. Stops are checked against each candle's low and fill at the
stop price (or the candle's open, if it gapped below). After a stop, the AI only buys again once its buy
signal has switched off and back on, so it doesn't jump straight back into the same fall.
"""
import pandas as pd

from . import backtest, config, data, metrics, model

# Each strategy: which models it reads, and its buy and sell rules (fixed before testing).
STRATEGIES = {
    "Next 1 day": {"models": ["4h"], "buy": lambda p: p["4h"] >= 0.55, "sell": lambda p: p["4h"] <= 0.48},
    "Next 4 hours": {"models": ["4h_next"], "buy": lambda p: p["4h_next"] >= 0.55,
                     "sell": lambda p: p["4h_next"] <= 0.48},
    "Combined (1 day + 4 hours)": {
        "models": ["4h", "4h_next"],
        "buy": lambda p: p["4h"] >= 0.55 and p["4h_next"] >= 0.50,
        "sell": lambda p: p["4h"] <= 0.48 or p["4h_next"] <= 0.45,
    },
    "Next 3 days": {"models": ["1d"], "buy": lambda p: p["1d"] >= 0.55, "sell": lambda p: p["1d"] <= 0.48},
    "Smart (sized by confidence)": {"models": ["4h", "4h_next", "1d"], "engine": "smart"},
}

# The "Smart" strategy's rules, fixed before it was first tested (2026-10-05). See _smart_trade.
# The stop-loss was then removed: across 9 half-years x 4 coins, 7% and 15% stops sold near short-term
# bottoms (price higher a day later 60% of the time). Chosen on 2022-2024, confirmed on 2025-2026; see
# docs/backTestResult. Set "stop" (and "protect_after") to numbers to turn a stop back on.
SMART = {
    "tiers": [(0.60, 1.0), (0.57, 0.67), (0.55, 0.33)],  # next-1-day P(up) -> share of the account in the coin
    "sell_below": 0.48,  # next-1-day P(up) at or below this -> reduce
    "keep_half_if_3day": 0.55,  # ...but sell only half if next-3-days P(up) is still at least this
    "take_profit": 0.10, "take_profit_below": 0.55,  # up 10% and confidence faded -> sell half
    "stop": None,  # e.g. 0.07: sell all 7% below the average buy price (None = no stop)
    "protect_after": None,  # e.g. 0.10: once up 10%, the stop moves to break-even (None = never)
    "dip_p4": 0.45, "limit_below": 0.01, "limit_candles": 2,  # dip expected -> limit order 1% lower for 8 hours
    "min_change": 0.10,  # ignore position changes smaller than 10% of the account
}


def predictions(name, start, end):
    """Out-of-sample P(up) for every coin between start and end, retraining at the start of every month.

    Returns a DataFrame indexed by candle open time with columns "symbol" and "prob".
    """
    spec = config.MODELS[name]
    step = pd.Timedelta(spec["timeframe"])
    ds = model.dataset(name, refresh=False)
    cols = model.feature_cols(ds)
    months = pd.date_range(start, end, freq="MS", tz="UTC")
    if len(months) == 0 or months[0] > start:
        months = months.insert(0, start)
    out = []
    for m0, m1 in zip(months, list(months[1:]) + [end]):
        # A row's label needs the close `horizon` candles later, known at index + (horizon + 1) * step.
        train = ds[ds.index <= m0 - (spec["horizon"] + 1) * step].dropna(subset=["y"])
        test = ds[(ds.index >= m0) & (ds.index < m1)]
        if test.empty:
            continue
        fitted = model._new_model().fit(train[cols], train["y"])
        out.append(pd.DataFrame({"symbol": test["symbol"], "prob": fitted.predict_proba(test[cols])[:, 1]},
                                index=test.index))
    return pd.concat(out)


def simulate(symbol, start, end, cash=1000.0, fee=config.FEE, strategies=STRATEGIES, stop=None, probs=None):
    """Run every strategy plus buy & hold on one coin.

    `stop` is None or {"type": "fixed" | "trailing", "pct": 0.10}. `probs` ({model: predictions(...)}) can be
    passed in to reuse predictions across runs. Returns (summary DataFrame, {strategy: trades}, balance DataFrame).
    """
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    needed = sorted({m for s in strategies.values() for m in s["models"]})
    probs = probs or {}
    for m in needed:
        if m not in probs:
            probs[m] = predictions(m, start, end)

    summaries, trade_logs, curves = [], {}, {}
    for label, strat in strategies.items():
        tf = config.MODELS[strat["models"][0]]["timeframe"]
        candles = data.closed(tf, refresh=False, symbols=[symbol])[symbol]
        if strat.get("engine") == "smart":
            p = _smart_inputs(probs, symbol, start, end)
        else:
            p = pd.concat([_for(probs[m], symbol, start, end).rename(m) for m in strat["models"]], axis=1).dropna()
        # Decide at each candle close; the last decision is one candle before `end` so every position is
        # valued at a close inside the period.
        p = p[p.index <= end - 2 * pd.Timedelta(tf)]
        if strat.get("engine") == "smart":
            result = _smart_trade(p, candles, cash, fee, {**SMART, **strat.get("rules", {})})
        else:
            result = _trade(p, candles, strat, cash, fee, tf, stop)
        result["summary"].update(_risk(result["equity"], tf))
        summaries.append({"strategy": label, **result["summary"]})
        trade_logs[label] = result["trades"]
        curves[label] = result["equity"]

    # Buy & hold: buy at the close of the first 4h candle, sell at the last close of the period.
    closes = data.closed("4h", refresh=False, symbols=[symbol])[symbol]["close"]
    held = closes[(closes.index >= start) & (closes.index <= end - pd.Timedelta("4h"))]
    coins = cash * (1 - fee) / held.iloc[0]
    hold_eq = coins * held
    hold_eq.iloc[-1] *= 1 - fee  # selling at the end pays the fee too, as for the strategies
    hold_eq.index = hold_eq.index + pd.Timedelta("4h")
    curves["Buy & hold"] = hold_eq
    fees = cash * fee + coins * held.iloc[-1] * fee
    summaries.append({"strategy": "Buy & hold",
                      **_summary(hold_eq, cash, 1, float(hold_eq.iloc[-1] > cash), fees, 1.0, 0),
                      **_risk(hold_eq, "4h")})
    return pd.DataFrame(summaries).set_index("strategy"), trade_logs, pd.DataFrame(curves).sort_index().ffill()


def _for(pred, symbol, start, end):
    rows = pred[(pred["symbol"] == symbol) & (pred.index >= start) & (pred.index < end)]
    return rows["prob"]


def _smart_inputs(probs, symbol, start, end):
    """4h decision rows with p1 (next 1 day), p4 (next 4 hours) and p3 (next 3 days, from the last closed day)."""
    p = pd.concat([_for(probs["4h"], symbol, start, end).rename("p1"),
                   _for(probs["4h_next"], symbol, start, end).rename("p4")], axis=1).dropna()
    daily = _for(probs["1d"], symbol, start - pd.Timedelta("3D"), end).rename("p3").to_frame()
    daily["known_at"] = daily.index + pd.Timedelta("1D")  # a daily candle's prediction exists once it closes
    p["known_at"] = p.index + pd.Timedelta("4h")
    p = pd.merge_asof(p.reset_index().sort_values("known_at"), daily.reset_index(drop=True).sort_values("known_at"),
                      on="known_at", direction="backward").set_index(p.index.name or "index")
    p.index.name = "time"
    return p.drop(columns="known_at").dropna()


def _smart_trade(p, candles, cash, fee, rules=SMART):
    """Size the position by confidence, sell partly or fully, take profit, stop out, and buy dips with limits.

    Every rule is in SMART; the comments on it say what each one does.
    """
    step = pd.Timedelta("4h")
    start_cash, coins, cost, fees = cash, 0.0, 0.0, 0.0  # cost = total paid for the coins held (incl. fees)
    protected = tp_done = half_sold = wait_reset = False
    pending = None  # limit order: {"price", "left", "target"}
    actions, equity, exposure, stops = [], {p.index[0] + step: cash}, [], 0

    def value(px):
        return cash + coins * px

    def buy(when, px, target, what, reason):
        nonlocal cash, coins, cost, fees, protected, tp_done
        spend = min((target - coins * px / value(px)) * value(px), cash)
        if spend < config.MIN_TRADE_USDT:
            return
        fees += spend * fee
        if coins == 0:
            protected = tp_done = False
        coins += spend * (1 - fee) / px
        cost += spend
        cash -= spend
        actions.append({"time": when, "action": what, "price": px, "usdt": spend, "reason": reason, "profit": None})

    def sell(when, px, share, what, reason):
        nonlocal cash, coins, cost, fees
        qty = coins * share
        proceeds = qty * px * (1 - fee)
        fees += qty * px * fee
        profit = proceeds - cost * share
        cash += proceeds
        coins -= qty
        cost *= 1 - share
        actions.append({"time": when, "action": what, "price": px, "usdt": proceeds, "reason": reason, "profit": profit})

    def target_for(p1):
        return next((share for level, share in rules["tiers"] if p1 >= level), None)

    for t, row in p.iterrows():
        c = candles.loc[t]
        when, px = t + step, c["close"]

        # 1. A waiting limit order fills if the price dipped to it during this candle.
        if pending and c["low"] <= pending["price"]:
            buy(when, pending["price"], pending["target"], "BUY (limit)", "bought the expected dip 1% lower")
            pending = None

        # 2. Stop-loss: 7% under the average buy price, or break-even once the position has been up 10%.
        if coins > 0 and rules["stop"] is not None:
            avg = cost / coins
            level = avg * 1.002 if protected else avg * (1 - rules["stop"])
            if c["low"] <= level:
                fill = min(level, c["open"])
                sell(when, fill, 1.0, "SELL (stop)", "break-even stop" if protected else f"stop-loss -{rules['stop']:.0%}")
                stops += 1
                wait_reset, pending = True, None
            elif rules["protect_after"] is not None and px >= avg * (1 + rules["protect_after"]):
                protected = True

        p1, p4, p3 = row["p1"], row["p4"], row["p3"]
        if wait_reset and p1 < rules["tiers"][-1][0]:
            wait_reset = False  # buy signal switched off: buying is allowed again once it switches back on
        if p1 > rules["sell_below"]:
            half_sold = False

        # 3. Take some profit: up 10% and confidence has faded -> sell half, once per position.
        if coins > 0 and not tp_done and px >= cost / coins * (1 + rules["take_profit"]) and p1 < rules["take_profit_below"]:
            sell(when, px, 0.5, "SELL half (profit)", f"up {px / (cost / coins) - 1:+.0%}, confidence faded")
            tp_done = True

        # 4. Sell signal: all of it, or half if the 3-day view still expects more to come.
        if coins > 0 and p1 <= rules["sell_below"]:
            if p3 >= rules["keep_half_if_3day"]:
                if not half_sold:
                    sell(when, px, 0.5, "SELL half", f"next day looks weak ({p1:.0%}) but next 3 days still up ({p3:.0%})")
                    half_sold = True
            else:
                sell(when, px, 1.0, "SELL all", f"next day looks weak ({p1:.0%}) and next 3 days too ({p3:.0%})")
            pending = None

        # 5. Buy signal: size by confidence; buy lower with a limit order if a dip is expected.
        target = target_for(p1)
        if target is not None and not wait_reset:
            gap = target - coins * px / value(px)
            if gap >= rules["min_change"]:
                if pending:
                    pending["left"] -= 1
                    pending["target"] = target
                    if pending["left"] <= 0:
                        buy(when, px, target, "BUY", f"limit not filled; confidence {p1:.0%}")
                        pending = None
                elif p4 <= rules["dip_p4"]:
                    pending = {"price": px * (1 - rules["limit_below"]), "left": rules["limit_candles"], "target": target}
                else:
                    buy(when, px, target, "BUY", f"confidence {p1:.0%} -> {target:.0%} of account")
        elif pending:
            pending = None  # confidence gone before the dip came: cancel the order

        exposure.append(coins * px / value(px))
        equity[when] = value(px)

    last_t = p.index[-1] + step
    if coins > 0:
        sell(last_t + step, candles.loc[last_t, "close"], 1.0, "SELL all", "end of the test period")
    equity[last_t + step] = cash

    eq = pd.Series(equity).sort_index()
    trades = pd.DataFrame(actions)
    sells = trades[trades["profit"].notna()] if len(trades) else trades
    win_rate = float((sells["profit"] > 0).mean()) if len(sells) else float("nan")
    invested = sum(e > 0 for e in exposure) / len(exposure)
    return {"summary": _summary(eq, start_cash, len(trades), win_rate, fees, invested, stops),
            "trades": trades, "equity": eq}


def _trade(p, candles, strat, cash, fee, tf, stop):
    """Walk through the decisions: buy with all cash on the buy rule, sell everything on the sell rule or stop."""
    step = pd.Timedelta(tf)
    start_cash, coins, fees, in_market, stops = cash, 0.0, 0.0, 0, 0
    trades, open_trade, peak, wait_for_reset = [], None, 0.0, False
    equity = {p.index[0] + step: cash}

    def sell(when, price, note=None):
        nonlocal cash, coins, fees, open_trade
        fees += coins * price * fee
        cash, coins = coins * price * (1 - fee), 0.0
        trades.append({**open_trade, "sold": when, "sell price": price, "proceeds": cash,
                       **({"note": note} if note else {})})
        open_trade = None

    for t, row in p.iterrows():
        candle = candles.loc[t]  # the candle that has just closed; the decision is made at its close
        when = t + step

        # Stop-loss: did this candle trade through the stop while we held?
        if coins > 0 and stop:
            level = (open_trade["buy price"] if stop["type"] == "fixed" else peak) * (1 - stop["pct"])
            if candle["low"] <= level:
                sell(when, min(level, candle["open"]), f"{stop['type']} stop at -{stop['pct']:.0%}")
                stops += 1
                wait_for_reset = True
        buy_signal = strat["buy"](row)
        if wait_for_reset and not buy_signal:
            wait_for_reset = False

        price = candle["close"]
        if coins == 0 and buy_signal and not wait_for_reset:
            fees += cash * fee
            open_trade = {"bought": when, "buy price": price, "cost": cash}
            coins, cash, peak = cash * (1 - fee) / price, 0.0, price
        elif coins > 0 and strat["sell"](row):
            sell(when, price)
        if coins > 0:
            peak = max(peak, price)
        in_market += coins > 0
        equity[when] = cash + coins * price

    # Anything still held is sold at the last close of the period (paying the fee).
    last_t = p.index[-1] + step
    if coins > 0:
        sell(last_t + step, candles.loc[last_t, "close"], "still open at the end; sold at the last close")
    equity[last_t + step] = cash

    eq = pd.Series(equity).sort_index()
    trades = pd.DataFrame(trades)
    if len(trades):
        trades["profit"] = trades["proceeds"] - trades["cost"]
        trades["return"] = trades["proceeds"] / trades["cost"] - 1
    win_rate = float((trades["profit"] > 0).mean()) if len(trades) else float("nan")
    return {"summary": _summary(eq, start_cash, len(trades), win_rate, fees, in_market / len(p), stops),
            "trades": trades, "equity": eq}


def _risk(equity, tf):
    """Sharpe (with its standard error), Sortino and the chance the Sharpe is above 0, from the balance curve.

    Over a few months the standard error is large (about 1.4 for 6 months), so read these with care.
    """
    q = backtest.PERIODS_PER_YEAR[tf]
    r = equity.pct_change().dropna()
    return {"sharpe": metrics.sharpe(r, q), "sharpe_se": metrics.sharpe_se(r, q),
            "sortino": metrics.sortino(r, q), "chance sharpe > 0": metrics.psr(r, q)}


def _summary(eq, start_cash, n_trades, win_rate, fees, time_in_market, stops):
    return {
        "final balance": eq.iloc[-1],
        "return": eq.iloc[-1] / start_cash - 1,
        "worst drop": (eq / eq.cummax() - 1).min(),
        "trades": n_trades,
        "win rate": win_rate,
        "fees paid": fees,
        "time invested": time_in_market,
        "stops hit": stops,
    }


def balance_chart(equity, symbol, cash):
    """Interactive chart of every strategy's balance over the period."""
    import plotly.graph_objects as go

    colors = {"Next 1 day": "#0052FF", "Next 4 hours": "#F4B000", "Combined (1 day + 4 hours)": "#C855E8",
              "Next 3 days": "#098551", "Buy & hold": "#8A919E"}
    fig = go.Figure()
    for name in equity:
        fig.add_trace(go.Scatter(x=equity.index, y=equity[name], name=name, mode="lines",
                                 line=dict(color=colors.get(name), width=2.4 if name == "Buy & hold" else 1.8,
                                           dash="dot" if name == "Buy & hold" else None),
                                 hovertemplate="$%{y:,.2f}"))
    fig.add_hline(y=cash, line=dict(color="#8A919E", width=1, dash="dash"), annotation_text="starting balance")
    fig.update_layout(title=f"{symbol}: balance of a ${cash:,.0f} account", template="plotly_white",
                      hovermode="x unified", yaxis_tickprefix="$", height=560,
                      legend=dict(orientation="h", y=-0.12))
    return fig
