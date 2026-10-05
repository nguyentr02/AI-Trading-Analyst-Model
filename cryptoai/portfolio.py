"""Your holdings, stored locally in portfolio.json."""
import json

import pandas as pd

from . import config, data


def load():
    if not config.PORTFOLIO_FILE.exists():
        return []
    return json.loads(config.PORTFOLIO_FILE.read_text())


def save(holdings):
    config.PORTFOLIO_FILE.write_text(json.dumps(holdings, indent=2))


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
