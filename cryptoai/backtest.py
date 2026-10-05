"""Simulate trading the model's out-of-sample signals: long or flat, with fees."""
import numpy as np
import pandas as pd

from . import config

PERIODS_PER_YEAR = {"4h": 6 * 365, "1d": 365}


def positions(prob, enter=config.ENTER_PROB, exit_=config.EXIT_PROB):
    """Long when prob > enter, stay long until prob < exit (hysteresis cuts over-trading)."""
    pos, out = 0, []
    for p in prob:
        if pos == 0 and p > enter:
            pos = 1
        elif pos == 1 and p < exit_:
            pos = 0
        out.append(pos)
    return pd.Series(out, index=prob.index)


def run(oos_symbol, timeframe, enter=config.ENTER_PROB, exit_=config.EXIT_PROB, fee=config.FEE):
    """oos_symbol: walk-forward rows for one symbol (columns prob, fwd_ret_1)."""
    df = oos_symbol.dropna(subset=["fwd_ret_1"]).sort_index()
    pos = positions(df["prob"], enter, exit_)
    # Decision at candle close t earns the return of candle t+1 (fwd_ret_1); pay fee on every switch.
    trades = pos.diff().abs().fillna(pos.iloc[0])
    strat = pos * df["fwd_ret_1"] - trades * fee
    equity = pd.DataFrame(
        {"strategy": (1 + strat).cumprod(), "buy_hold": (1 + df["fwd_ret_1"]).cumprod(), "position": pos}
    )
    return equity, _stats(strat, df["fwd_ret_1"], pos, trades, timeframe)


def _stats(strat, bh, pos, trades, timeframe):
    ppy = PERIODS_PER_YEAR[timeframe]

    def summary(r):
        eq = (1 + r).cumprod()
        years = len(r) / ppy
        return {
            "total_return": eq.iloc[-1] - 1,
            "cagr": eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
            "sharpe": r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else np.nan,
            "max_drawdown": (eq / eq.cummax() - 1).min(),
        }

    s, b = summary(strat), summary(bh)
    # Per-trade win rate: group consecutive long candles into trades.
    trade_id = (pos.diff() == 1).cumsum()[pos == 1]
    trade_rets = (1 + strat[pos == 1]).groupby(trade_id).prod() - 1
    return {
        "strategy": s,
        "buy_hold": b,
        "num_trades": int((pos.diff() == 1).sum() + (pos.iloc[0] == 1)),
        "win_rate": float((trade_rets > 0).mean()) if len(trade_rets) else np.nan,
        "time_in_market": float(pos.mean()),
        "fees_paid_pct": float(trades.sum() * config.FEE),
    }
