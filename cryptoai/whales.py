"""Whale watch: very large trades, liquidation cascades and open-interest jumps on Binance, logged and alerted.

Runs inside the live service (live.py starts it) as three background threads, all on free public Binance data:
1. Spot trades (aggTrade stream): a trade of at least BIG_TRADE[coin] is logged. An alert goes out when
   BURST_COUNT or more such trades on the same side land within BURST_WINDOW seconds, or when a single one is at
   least HUGE_MULT times the threshold.
2. Futures liquidations (!forceOrder@arr stream): forced sales of leveraged positions on our 4 coins' USDT
   perpetuals. An alert goes out when more than LIQ_ALERT is liquidated within LIQ_WINDOW seconds. Binance sends
   at most one liquidation per coin per second, so totals are undercounts.
3. Open interest (REST, every OI_EVERY seconds): the total of open futures positions. An alert goes out when the
   1-hour change is unusually large (at least OI_Z standard deviations of recent 1-hour changes and OI_MIN).

Alerts are information only: nothing trades on them. Research (docs/research/whale-tracking.md) found whale and
flow data mostly predicts volatility, not direction. Events go to logs/whale_events.csv, a summary to
logs/whale_status.json for the dashboard. Switch alerts off with "whales": false in notify.json.
"""
import csv
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import requests
from websockets.sync.client import connect

from . import config, notify

BIG_TRADE = {"BTC/USDT": 2_000_000, "ETH/USDT": 1_000_000, "BNB/USDT": 300_000, "SOL/USDT": 300_000}
BURST_COUNT, BURST_WINDOW, HUGE_MULT = 3, 60, 5
LIQ_LOG, LIQ_WINDOW, LIQ_ALERT = 100_000, 300, 10_000_000
OI_EVERY, OI_Z, OI_MIN = 300, 3.0, 0.03
COOLDOWN = {"burst": 1800, "huge": 900, "cascade": 1800, "oi": 3600}
EVENTS = config.LOG_DIR / "whale_events.csv"
STATUS = config.LOG_DIR / "whale_status.json"
FIELDS = ["time", "kind", "symbol", "side", "usd", "price", "note"]
SPOT_WS = "wss://stream.binance.com:9443/stream?streams="
FUTURES_WS = "wss://fstream.binance.com/ws/!forceOrder@arr"
FAPI = "https://fapi.binance.com"


def _now():
    return datetime.now(timezone.utc)


def _log(msg):
    print(f"{_now():%Y-%m-%d %H:%M:%S} UTC  {msg}", flush=True)


def coin(symbol):
    return symbol.split("/")[0]


class WhaleWatch:
    def __init__(self, symbols=config.SYMBOLS):
        self.symbols = {s.replace("/", ""): s for s in symbols}
        self.lock = threading.Lock()
        self.trades = {s: {"BUY": deque(), "SELL": deque()} for s in symbols}  # (time, usd) of big trades
        self.liqs = deque()  # (time, symbol, side, usd)
        self.oi = {s: deque(maxlen=600) for s in symbols}  # (time, open interest in coins)
        self.last_alert = {}
        self.status = {"started": _now().isoformat(timespec="seconds"), "oi": {}, "trades_1h": {}, "liquidations_1h": {}}

    def start(self):
        for target in (self._spot_loop, self._liquidation_loop, self._oi_loop):
            threading.Thread(target=target, daemon=True, name=target.__name__).start()
        _log("Whale watch started: big trades, liquidations, open interest")

    # ---------- recording and alerting ----------
    def _record(self, kind, symbol, side, usd, price, note=""):
        row = {"time": _now().isoformat(timespec="seconds"), "kind": kind, "symbol": symbol, "side": side,
               "usd": round(usd), "price": price, "note": note}
        with self.lock:
            new = not EVENTS.exists()
            with open(EVENTS, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)

    def _alert(self, key, cooldown_kind, title, body):
        t = time.time()
        if t - self.last_alert.get(key, 0) < COOLDOWN[cooldown_kind]:
            return
        self.last_alert[key] = t
        _log(f"WHALE ALERT: {title} - {body}")
        if notify.settings().get("whales", True):
            threading.Thread(target=notify.send, args=(title, body), daemon=True).start()  # toasts take ~1 s

    def _run_forever(self, name, body):
        backoff = 2
        while True:
            try:
                body()
                backoff = 2
            except Exception as e:  # network drop, Binance's 24h disconnect: reconnect
                _log(f"whale watch: {name} stream lost ({type(e).__name__}: {e}); retrying in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)

    # ---------- 1. big spot trades ----------
    def _spot_loop(self):
        url = SPOT_WS + "/".join(f"{s.lower()}@aggTrade" for s in self.symbols)

        def body():
            with connect(url, open_timeout=15, close_timeout=5, max_size=2**20) as ws:
                for msg in ws:
                    self._on_trade(json.loads(msg)["data"])
        self._run_forever("trade", body)

    def _on_trade(self, d):
        sym = self.symbols.get(d["s"])
        price, usd = float(d["p"]), float(d["p"]) * float(d["q"])
        if sym is None or usd < BIG_TRADE[sym]:
            return
        side = "SELL" if d["m"] else "BUY"  # buyer was the maker -> the seller hit the bid
        self._record("trade", sym, side, usd, price)
        now, q = time.time(), self.trades[sym][side]
        q.append((now, usd))
        while q and now - q[0][0] > 3600:
            q.popleft()
        recent = [u for t, u in q if now - t <= BURST_WINDOW]
        verb = "buying" if side == "BUY" else "selling"
        if usd >= HUGE_MULT * BIG_TRADE[sym]:
            self._alert(f"huge:{sym}:{side}", "huge", f"Whale {verb} {coin(sym)}",
                        f"One ${usd / 1e6:,.1f}M market {side.lower()} at ${price:,.2f} on Binance spot.")
        elif len(recent) >= BURST_COUNT:
            self._alert(f"burst:{sym}:{side}", "burst", f"Whales {verb} {coin(sym)}",
                        f"{len(recent)} market {side.lower()}s of ${BIG_TRADE[sym] / 1e6:g}M+ in {BURST_WINDOW} s "
                        f"(${sum(recent) / 1e6:,.1f}M) near ${price:,.2f}.")

    # ---------- 2. liquidations ----------
    def _liquidation_loop(self):
        def body():
            with connect(FUTURES_WS, open_timeout=15, close_timeout=5, max_size=2**20) as ws:
                for msg in ws:
                    self._on_liquidation(json.loads(msg)["o"])
        self._run_forever("liquidation", body)

    def _on_liquidation(self, o):
        sym = self.symbols.get(o["s"])
        if sym is None:
            return
        price = float(o.get("ap") or o["p"])
        usd = price * float(o.get("z") or o["q"])
        side = "LONG" if o["S"] == "SELL" else "SHORT"  # a forced SELL closes a long
        if usd >= LIQ_LOG:
            self._record("liquidation", sym, side, usd, price)
        now = time.time()
        self.liqs.append((now, sym, side, usd))
        while self.liqs and now - self.liqs[0][0] > 3600:
            self.liqs.popleft()
        recent = [x for x in self.liqs if now - x[0] <= LIQ_WINDOW]
        for s in ("LONG", "SHORT"):
            total = sum(x[3] for x in recent if x[2] == s)
            if total >= LIQ_ALERT:
                by_coin = {}
                for x in recent:
                    if x[2] == s:
                        by_coin[coin(x[1])] = by_coin.get(coin(x[1]), 0) + x[3]
                parts = ", ".join(f"{k} ${v / 1e6:,.1f}M" for k, v in sorted(by_coin.items(), key=lambda kv: -kv[1]))
                what = "longs force-sold (price falling fast)" if s == "LONG" else "shorts force-bought (price rising fast)"
                self._record("cascade", "ALL", s, total, 0.0, parts)
                self._alert(f"cascade:{s}", "cascade", "Liquidation cascade",
                            f"${total / 1e6:,.1f}M of {what} in {LIQ_WINDOW // 60} min: {parts}.")

    # ---------- 3. open interest ----------
    def _oi_loop(self):
        def seed():
            for s in self.symbols:
                rows = requests.get(f"{FAPI}/futures/data/openInterestHist",
                                    params={"symbol": s, "period": "5m", "limit": 500}, timeout=30).json()
                self.oi[self.symbols[s]].extend((r["timestamp"] / 1000, float(r["sumOpenInterest"])) for r in rows)

        def body():
            if not any(self.oi.values()):
                seed()
            while True:
                for s, sym in self.symbols.items():
                    r = requests.get(f"{FAPI}/fapi/v1/openInterest", params={"symbol": s}, timeout=30).json()
                    self.oi[sym].append((time.time(), float(r["openInterest"])))
                    self._check_oi(sym)
                self._save_status()
                time.sleep(OI_EVERY)
        self._run_forever("open interest", body)

    def _check_oi(self, sym):
        t, v = map(np.array, zip(*self.oi[sym]))
        hour_ago = np.searchsorted(t, t[-1] - 3600, side="right") - 1
        if hour_ago < 0 or len(t) < 100:
            return
        chg = np.log(v[-1] / v[hour_ago])
        # recent 1-hour changes, from the stored series (about one value every 5 minutes)
        idx = np.searchsorted(t, t - 3600, side="right") - 1
        ok = idx >= 0
        hist = np.log(v[ok] / v[idx[ok]])[:-1]
        sd = hist.std() if len(hist) > 30 else np.nan
        self.status["oi"][sym] = {"open_interest": float(v[-1]), "change_1h": float(chg)}
        if np.isfinite(sd) and abs(chg) >= max(OI_Z * sd, OI_MIN):
            what = "leverage piling in" if chg > 0 else "positions being closed"
            self._record("oi", sym, "UP" if chg > 0 else "DOWN", 0.0, 0.0, f"{chg:+.1%} in 1 h ({chg / sd:+.1f} sd)")
            self._alert(f"oi:{sym}", "oi", f"Open interest jump: {coin(sym)}",
                        f"Futures open interest {chg:+.1%} in 1 hour ({what}), {abs(chg) / sd:.1f}x its usual move.")

    def _save_status(self):
        now = time.time()
        trades = {}
        for sym, sides in self.trades.items():
            trades[sym] = {side: round(sum(u for t, u in q if now - t <= 3600)) for side, q in sides.items()}
        liqs = {"LONG": 0.0, "SHORT": 0.0}
        for t, _, side, usd in self.liqs:
            if now - t <= 3600:
                liqs[side] += usd
        self.status.update({"updated": _now().isoformat(timespec="seconds"), "trades_1h": trades,
                            "liquidations_1h": {k: round(v) for k, v in liqs.items()}})
        with config.atomic(STATUS) as tmp:
            tmp.write_text(json.dumps(self.status, indent=2))


def load_status():
    try:
        return json.loads(STATUS.read_text())
    except (OSError, ValueError):
        return None


def load_events(limit=200):
    import pandas as pd
    if not EVENTS.exists():
        return pd.DataFrame(columns=FIELDS)
    df = pd.read_csv(EVENTS)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    return df.tail(limit)
