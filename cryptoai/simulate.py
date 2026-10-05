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
        p = pd.concat([_for(probs[m], symbol, start, end).rename(m) for m in strat["models"]], axis=1).dropna()
        # Decide at each candle close; the last decision is one candle before `end` so every position is
        # valued at a close inside the period.
        p = p[p.index <= end - 2 * pd.Timedelta(tf)]
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
