"""Live market snapshot: Binance 24h ticker plus CoinGecko market data (both public, no API key)."""
import json
import threading
import time

import pandas as pd
import requests
from websockets.sync.client import connect

from . import config, data

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
BINANCE_WS = "wss://stream.binance.com:9443/stream?streams="


class LiveFeed:
    """Background thread holding Binance 24h tickers, pushed over WebSocket once per second."""

    def __init__(self, symbols=config.SYMBOLS):
        self.symbols = {s.replace("/", "").lower(): s for s in symbols}
        self.latest = {}
        self.error = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        url = BINANCE_WS + "/".join(f"{s}@ticker" for s in self.symbols)
        while True:
            try:
                with connect(url, open_timeout=10, close_timeout=2) as ws:
                    self.error = None
                    for msg in ws:
                        self._handle(json.loads(msg)["data"])
            except Exception as e:  # network drop or Binance's 24h disconnect: reconnect
                self.error = str(e)
                time.sleep(2)

    def _handle(self, t):
        sym = self.symbols[t["s"].lower()]
        self.latest[sym] = {
            "symbol": sym,
            "price": float(t["c"]),
            "change_24h": float(t["P"]) / 100,
            "high_24h": float(t["h"]),
            "low_24h": float(t["l"]),
            "volume_24h": float(t["v"]),
            "quote_volume_24h": float(t["q"]),
            "bid": float(t["b"]),
            "ask": float(t["a"]),
            "updated": pd.to_datetime(t["E"], unit="ms", utc=True),
        }

    def tickers(self):
        """Latest ticker per symbol, or None until every symbol has received its first update."""
        if len(self.latest) < len(self.symbols):
            return None
        return pd.DataFrame([self.latest[s] for s in self.symbols.values()])


def tickers(symbols=config.SYMBOLS):
    """Binance 24h stats: last price, change, high/low and volume. Updates in real time."""
    raw = data.exchange().fetch_tickers(symbols)
    rows = []
    for sym in symbols:
        t = raw[sym]
        rows.append(
            {
                "symbol": sym,
                "price": t["last"],
                "change_24h": t["percentage"] / 100 if t["percentage"] is not None else None,
                "high_24h": t["high"],
                "low_24h": t["low"],
                "volume_24h": t["baseVolume"],  # in coins
                "quote_volume_24h": t["quoteVolume"],  # in USDT, Binance only
                "bid": t["bid"],
                "ask": t["ask"],
                "updated": pd.to_datetime(t["timestamp"], unit="ms", utc=True),
            }
        )
    return pd.DataFrame(rows)


def coingecko(symbols=config.SYMBOLS):
    """Market cap, rank, supply and all-time high from CoinGecko. Updates about once a minute."""
    ids = {config.COINGECKO_IDS[s]: s for s in symbols}
    resp = requests.get(
        COINGECKO_URL,
        params={"vs_currency": "usd", "ids": ",".join(ids), "price_change_percentage": "1h,7d"},
        timeout=10,
    )
    resp.raise_for_status()
    rows = []
    for c in resp.json():
        rows.append(
            {
                "symbol": ids[c["id"]],
                "market_cap": c["market_cap"],
                "rank": c["market_cap_rank"],
                "fdv": c["fully_diluted_valuation"],
                "total_volume_usd": c["total_volume"],  # all exchanges
                "change_1h": _pct(c.get("price_change_percentage_1h_in_currency")),
                "change_7d": _pct(c.get("price_change_percentage_7d_in_currency")),
                "circulating_supply": c["circulating_supply"],
                "max_supply": c["max_supply"],
                "ath": c["ath"],
                "from_ath": _pct(c["ath_change_percentage"]),
            }
        )
    return pd.DataFrame(rows)


def _pct(v):
    return v / 100 if v is not None else None


def snapshot(symbols=config.SYMBOLS):
    """Binance and CoinGecko data joined per coin. CoinGecko columns are empty if it is unreachable."""
    df = tickers(symbols)
    try:
        df = df.merge(coingecko(symbols), on="symbol", how="left")
    except requests.RequestException:
        pass
    return df
