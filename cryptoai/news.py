"""Market news and mood for crypto and US stocks, plus the Crypto Fear & Greed Index.

Headlines (free RSS, no keys): CoinDesk and Cointelegraph, Google News searches per coin, and Yahoo Finance per stock.
Each headline is scored with VADER (a word-based sentiment scorer: it cannot "know" later prices, so it cannot leak
the future) plus a small finance word list, from -1 (negative) to +1 (positive). Mood = average score of the
headlines from the last 24 hours.

News mood is shown next to the signals but does NOT change them yet: news sentiment can't be tested honestly on
past data with free sources, and its published edge for big assets is small and fading
(docs/research/stock-signals.md). Instead, every 4h close the live service logs each asset's news mood with the AI's
reading (logs/news_log.csv), so whether news adds anything can be tested on data collected from now on.

The Crypto Fear & Greed Index (alternative.me) has daily history from 2018, so it IS tested
(experiments/fear_greed.py).
"""
import email.utils
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import config

HEADERS = {"User-Agent": "Mozilla/5.0"}
CRYPTO_FEEDS = ["https://www.coindesk.com/arc/outboundfeeds/rss/", "https://cointelegraph.com/rss"]
COIN_QUERIES = {"BTC/USDT": "Bitcoin", "ETH/USDT": "Ethereum", "BNB/USDT": "BNB Binance coin", "SOL/USDT": "Solana",
                "NEAR/USDT": "NEAR Protocol", "ZEC/USDT": "Zcash", "XRP/USDT": "XRP Ripple", "DOGE/USDT": "Dogecoin",
                "AVAX/USDT": "Avalanche AVAX", "LINK/USDT": "Chainlink"}
COIN_WORDS = {"BTC/USDT": ["bitcoin", "btc"], "ETH/USDT": ["ethereum", "ether", "eth"], "BNB/USDT": ["bnb", "binance"],
              "SOL/USDT": ["solana", "sol"]}
FINANCE_WORDS = {  # added to VADER's lexicon (its scale is about -4..+4)
    "surge": 2.0, "surges": 2.0, "surged": 2.0, "soar": 2.2, "soars": 2.2, "soared": 2.2, "rally": 2.0, "rallies": 2.0,
    "rallied": 2.0, "jump": 1.5, "jumps": 1.5, "jumped": 1.5, "gain": 1.2, "gains": 1.2, "rise": 1.0, "rises": 1.0,
    "climbs": 1.2, "bullish": 2.0, "inflows": 1.2, "upgrade": 1.5, "upgraded": 1.5, "beats": 1.5, "outperform": 1.5,
    "breakout": 1.5, "rebound": 1.2, "rebounds": 1.2, "approval": 1.5, "approved": 1.5, "adoption": 1.0,
    "plunge": -2.5, "plunges": -2.5, "plunged": -2.5, "crash": -3.0, "crashes": -3.0, "crashed": -3.0, "dip": -1.0,
    "dips": -1.0, "fall": -1.5, "falls": -1.5, "fell": -1.5, "drop": -1.5, "drops": -1.5, "dropped": -1.5,
    "slump": -2.0, "slumps": -2.0, "tumble": -2.0, "tumbles": -2.0, "bearish": -2.0, "outflows": -1.2,
    "downgrade": -1.5, "downgraded": -1.5, "misses": -1.5, "hack": -2.5, "hacked": -2.5, "exploit": -2.0,
    "lawsuit": -1.5, "sued": -1.5, "ban": -2.0, "bans": -2.0, "liquidations": -1.5, "selloff": -2.0, "sell-off": -2.0,
    "fraud": -3.0, "probe": -1.2, "investigation": -1.2, "layoffs": -1.5, "bankruptcy": -3.0, "warning": -1.2,
}
_analyzer = None
NEWS_LOG = config.LOG_DIR / "news_log.csv"


def analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(FINANCE_WORDS)
    return _analyzer


def score(text):
    return float(analyzer().polarity_scores(text)["compound"])


def _rss(url):
    """[(published UTC, title, link)] from an RSS feed."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except requests.RequestException:
        return []
    out = []
    for item in re.findall(r"<item>(.*?)</item>", r.text, flags=re.S):
        title = re.search(r"<title>(.*?)</title>", item, flags=re.S)
        link = re.search(r"<link>(.*?)</link>", item, flags=re.S)
        date = re.search(r"<pubDate>(.*?)</pubDate>", item, flags=re.S)
        if not title:
            continue
        t = html.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", title.group(1))).strip()
        try:
            when = pd.Timestamp(email.utils.parsedate_to_datetime(date.group(1).strip())).tz_convert("UTC") if date else None
        except (TypeError, ValueError):
            when = None
        out.append((when, t, link.group(1).strip() if link else ""))
    return out


def _google(query):
    return _rss(f"https://news.google.com/rss/search?q={quote(query)}+when:2d&hl=en-US&gl=US&ceid=US:en")


def headlines(symbols_crypto=(), tickers_stock=()):
    """{asset: DataFrame(time, title, link, score)} of headlines from the last 2 days, newest first."""
    jobs = {}
    with ThreadPoolExecutor(8) as pool:
        general = [pool.submit(_rss, u) for u in CRYPTO_FEEDS]
        for s in symbols_crypto:
            jobs[s] = pool.submit(_google, COIN_QUERIES.get(s, s.split("/")[0]) + " crypto")
        for t in tickers_stock:
            jobs[t] = pool.submit(_rss, f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US")
        general_items = [i for f in general for i in f.result()]
        out = {}
        for asset, fut in jobs.items():
            items = list(fut.result())
            words = COIN_WORDS.get(asset)
            if words:  # crypto: also the general crypto feeds' headlines that mention the coin
                items += [i for i in general_items if any(re.search(rf"\b{w}\b", i[1].lower()) for w in words)]
            df = pd.DataFrame(items, columns=["time", "title", "link"]).drop_duplicates("title")
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta("2D")
            df = df[df["time"].isna() | (df["time"] >= cutoff)]
            df["score"] = df["title"].map(score)
            out[asset] = df.sort_values("time", ascending=False, na_position="last").reset_index(drop=True)
    return out


def mood(df, hours=24):
    """(average score, number of headlines) over the last `hours`."""
    recent = df[df["time"].notna() & (df["time"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours))]
    return (float(recent["score"].mean()) if len(recent) else float("nan")), len(recent)


def label(m):
    if pd.isna(m):
        return "No news"
    return "Positive" if m >= 0.15 else "Negative" if m <= -0.15 else "Mixed"


def fear_greed(limit=0):
    """Crypto Fear & Greed Index: daily values (0 = extreme fear, 100 = extreme greed), oldest first."""
    r = requests.get("https://api.alternative.me/fng/", params={"limit": limit, "format": "json"}, timeout=30)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["data"])
    d["time"] = pd.to_datetime(d["timestamp"].astype(int), unit="s", utc=True)
    d["value"] = d["value"].astype(int)
    return d.set_index("time")[["value", "value_classification"]].sort_index()


def log_moods(readings):
    """Append each asset's news mood with the AI's reading, for testing later whether news adds anything.
    `readings`: {asset: P(up) of the main model}."""
    hl = headlines([a for a in readings if "/" in a], [a for a in readings if "/" not in a])
    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    rows = []
    for asset, p in readings.items():
        m, n = mood(hl.get(asset, pd.DataFrame(columns=["time", "score"])))
        rows.append({"time": now, "asset": asset, "news_mood": m, "headlines": n, "ai_p_up": p})
    pd.DataFrame(rows).to_csv(NEWS_LOG, mode="a", header=not NEWS_LOG.exists(), index=False)
    return rows
