"""Your holdings and spare cash, stored locally in portfolio.json.

File format: {"cash": 500.0, "holdings": [{"symbol": "BTC/USDT", "amount": 0.01, "avg_cost": 80000}, ...]}.
An older file that is just the holdings list still loads (with no spare cash).
"""
import json

import pandas as pd

from . import config, data


def _read():
    if not config.PORTFOLIO_FILE.exists():
        return {"cash": 0.0, "holdings": []}
    raw = json.loads(config.PORTFOLIO_FILE.read_text())
    if isinstance(raw, list):
        return {"cash": 0.0, "holdings": raw}
    return {"cash": float(raw.get("cash", 0.0)), "holdings": raw.get("holdings", [])}


def load():
    """Holdings as a list of {"symbol", "amount", "avg_cost"}."""
    return _read()["holdings"]


def load_cash():
    """Spare cash in USDT, available for buying."""
    return _read()["cash"]


def save(holdings=None, cash=None):
    """Save holdings and/or spare cash; whichever is left out keeps its saved value."""
    current = _read()
    new = {"cash": current["cash"] if cash is None else float(cash),
           "holdings": current["holdings"] if holdings is None else holdings}
    config.PORTFOLIO_FILE.write_text(json.dumps(new, indent=2))


def valued(holdings=None):
    """Holdings with current price, value and unrealised P&L."""
    holdings = load() if holdings is None else holdings
    if not holdings:
        return pd.DataFrame(columns=["symbol", "amount", "avg_cost", "price", "value", "pnl", "pnl_pct"])
    df = pd.DataFrame(holdings)
    px = data.prices(sorted(df["symbol"].unique()))
    df["price"] = df["symbol"].map(px)
    df["value"] = df["amount"] * df["price"]
    cost = df["amount"] * df["avg_cost"]
    df["pnl"] = df["value"] - cost
    df["pnl_pct"] = df["pnl"] / cost
    return df
