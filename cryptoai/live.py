"""Always-on service: learn from every closed candle and publish a new signal within seconds.

    python -m cryptoai live

1. Catch up: download every candle (and funding/premium value) missed while the service was down,
   retrain on it, and log fresh signals.
2. Listen to Binance's kline WebSocket for all symbols and timeframes.
3. When a candle closes for every symbol, retrain every model built on that timeframe on all data up
   to it (kept only if it still passes MIN_AUC), then compute and log the new signals. A 4h close
   retrains "4h_next" and "4h"; the daily close also retrains "1d".
4. If the connection drops, reconnect and catch up again, so no candle is ever skipped.
"""
import json
import socket
import time
import traceback
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import ccxt
import pandas as pd
import requests
from websockets.sync.client import connect

from . import (advisor, altcoins, config, droprisk, explain, model, notify, paper, preview, signals, stockai,
               stockpaper, stocks, whales)

STREAM_URL = "wss://stream.binance.com:9443/stream?streams="
WAIT_FOR_ALL = 20  # seconds to wait for every symbol's close message before processing anyway
HEARTBEAT = 60  # seconds between status file updates, so the dashboard can tell the service is alive
NEW_YORK = ZoneInfo("America/New_York")  # the US market closes at 16:00 here (summer and winter time handled)
STOCK_STATE = config.LOG_DIR / "stock_state.json"  # the last US close the stock jobs handled


def last_us_close():
    """Date (New York) of the most recent US market close (16:00, Monday to Friday) that has already happened."""
    ny = datetime.now(NEW_YORK)
    day = ny.date() if ny.hour >= 16 else ny.date() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return str(day)


# Problems reaching Binance. (Not all OSErrors: a PermissionError on a locked file is a different problem.)
NETWORK_ERRORS = (ccxt.NetworkError, requests.RequestException, ConnectionError, TimeoutError, socket.gaierror)


def log(msg):
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC  {msg}", flush=True)


class LiveService:
    def __init__(self, symbols=config.SYMBOLS, timeframes=config.TIMEFRAMES):
        self.symbols = {s.replace("/", "").lower(): s for s in symbols}
        self.timeframes = timeframes
        self.pending = {}  # (timeframe, open_time_ms) -> (first_seen, set of symbols closed)
        self.status = {"started": _now(), "state": "starting"}
        self._saved_at = 0.0
        self._next_preview = 0.0  # due immediately
        self._preview_failing = False
        self._paper_hour = None

    # ---------- learning ----------
    def learn(self, tf, reason):
        """Fetch new data, retrain, and log the signals for every model built on timeframe `tf`."""
        self.trade_first(tf)
        for name in [n for n, s in config.MODELS.items() if s["timeframe"] == tf]:
            self._learn_model(name, reason)
        if tf == "4h":
            self.learn_drop_risk(reason)
        if tf == "1d":
            try:
                explain.compute()  # refresh "what the AI learnt" once a day
            except Exception:
                log(f"pattern explanation failed:\n{traceback.format_exc()}")
        if tf == advisor_timeframe():
            self.advise()
        self.paper_trade(tf)

    def trade_first(self, tf):
        """At a candle close, decide with the current models and fill at the live price straight away. Retraining
        takes minutes and comes after, so no trade waits for it (the retrained models decide from the next close)."""
        if not any(paper.is_open(b) for b in paper.BOOKS):
            return
        t0 = time.time()
        try:
            name = next(n for n, s in config.MODELS.items() if s["timeframe"] == tf)
            signals.current(name, refresh=True, fast=True)  # downloads the candle that just closed
        except NETWORK_ERRORS:
            raise
        except Exception:
            log(f"trade-first signals failed:\n{traceback.format_exc()}")
            return
        self.paper_trade(tf, started=t0)

    def learn_drop_risk(self, reason):
        """Retrain the drop warning (droprisk.py) on the 4h data just updated, and log each coin's reading."""
        t0 = time.time()
        try:
            m = droprisk.train(refresh=False)
        except NETWORK_ERRORS:
            raise
        except Exception:
            log(f"drop warning failed:\n{traceback.format_exc()}")
            return
        verdict = "updated" if m["accepted"] else f"REJECTED (AUC < {config.MIN_AUC}), kept previous model"
        log(f"drop warning {reason}: AUC {m['oos_auc']}, model {verdict} [{time.time() - t0:.0f}s]")
        for sym, r in (droprisk.load() or {}).get("coins", {}).items():
            log(f"    {sym:<9} P(sharp drop first, 3 days) {r['p_drop']:.0%}  {r['level']}")
        self.status["last_learn_drop"] = _now()
        self._save_status()

    def paper_trade(self, tf, started=None):
        """Let each open paper book act on the fresh signals: the AI at 4h closes, the trend accounts daily.
        With `started`, log how many seconds after the candle close each book's decisions were filled."""
        for book in paper.BOOKS:
            if not paper.is_open(book):
                continue
            try:
                if book is not paper.MAIN:
                    altcoins.compute()  # the altcoin readings from the models just retrained
                step = paper.step_ai if tf == "4h" else paper.step_daily if tf == "1d" else None
                done = step(book=book) if step else []
            except NETWORK_ERRORS:
                raise
            except Exception:
                log(f"paper trading step ({book.key}) failed:\n{traceback.format_exc()}")
                continue
            tag = "PAPER" if book is paper.MAIN else f"PAPER {book.key.upper()}"
            for r in done.itertuples() if len(done) else []:
                log(f"    {tag} {r.account.upper()}: {r.side} {r.symbol} {r.quantity:.6f} at {r.price:,.4f} "
                    f"(${r.total:,.2f}) - {r.reason}")
            if started is not None:
                log(f"    {tag}: {tf} close decided and filled {time.time() - started:.0f}s after starting "
                    f"({len(done)} trades), before retraining")

    def advise(self):
        """Re-check the portfolio advice with the fresh signals and alert on any change (Windows, Zalo)."""
        try:
            advice, changed = advisor.check_and_notify()
        except NETWORK_ERRORS:
            raise
        except Exception:
            log(f"advice check failed:\n{traceback.format_exc()}")
            return
        for r in changed.itertuples():
            log(f"    ALERT: {r.action} {r.symbol}" + (f" ~${r.amount_usdt:,.0f}" if r.amount_usdt else ""))

    def _learn_model(self, tf, reason):
        t0 = time.time()
        m = model.train(tf, min_auc=config.MIN_AUC)  # downloads any missing candles first
        allsig, changed = signals.check_and_log([tf])
        verdict = "updated" if m["accepted"] else f"REJECTED (AUC < {config.MIN_AUC}), kept previous model"
        log(f"{tf} {reason}: data to {m['last_candle'][:16]}, AUC {m['oos_auc']}, model {verdict} "
            f"[{time.time() - t0:.0f}s]")
        for r in allsig.itertuples():
            flag = "  <-- CHANGED" if r.changed else ""
            log(f"    {r.symbol:<9} {tf}  P(up) {r.prob_up:.0%}  {r.signal}{flag}")
        self.status[f"last_learn_{tf}"] = _now()
        self.status[f"last_candle_{tf}"] = m["last_candle"]
        self._save_status()

    def catch_up(self):
        self.backfill()
        for tf in self.timeframes:
            try:
                self.learn(tf, "catch-up")
            except NETWORK_ERRORS as e:
                log(f"{tf} catch-up failed, no connection to Binance ({type(e).__name__}); will retry")
            except Exception:
                log(f"{tf} catch-up failed:\n{traceback.format_exc()}")

    def refresh_preview(self):
        """Re-run the models on the live price (provisional signals, see preview.py). Never stops the service."""
        self._next_preview = time.time() + config.PREVIEW_EVERY
        try:
            live_now = preview.compute()
            self.status["last_preview"] = _now()
            for book in paper.BOOKS:  # the paper AI may trade at any moment, at that moment's price
                if not paper.is_open(book):
                    continue
                readings = live_now if book is paper.MAIN else preview.compute(
                    extra=book.symbols, path=preview.ALT_PREVIEW)
                done = paper.step_ai_live(readings, book=book)
                shock = paper.step_shock(book=book)
                done = pd.concat([d for d in (done, shock) if len(d)]) if len(done) or len(shock) else done
                tag = "PAPER AI" if book is paper.MAIN else f"PAPER {book.key.upper()}"
                for r in done.itertuples() if len(done) else []:
                    log(f"    {tag} (live): {r.side} {r.symbol} {r.quantity:.6f} at {r.price:,.4f} "
                        f"(${r.total:,.2f}) - {r.reason}")
            self.stock_close()
            self.market_crash_check()
            hour = time.strftime("%Y%m%d%H", time.gmtime())
            if hour != self._paper_hour:  # balance history: one point an hour
                self._paper_hour = hour
                self.stock_jobs()
                for book in paper.BOOKS:
                    if not paper.is_open(book):
                        continue
                    paper.snapshot(book=book)
                    news = paper.milestones(lambda title, body: notify.send(title, body), book=book)
                    if news:
                        log(f"    PAPER {book.key.upper()}: {news}")
            self._preview_failing = False
        except Exception as e:
            if not self._preview_failing:  # log the first failure of a run of them, not one per minute
                log(f"live preview failed ({type(e).__name__}: {e}); will keep trying every minute")
            self._preview_failing = True

    def stock_jobs(self):
        """Hourly: the stock paper trial's balance point and weekly updates."""
        try:
            if stockpaper.is_open():
                stockpaper.snapshot()
                news = stockpaper.milestones(lambda title, body: notify.send(title, body))
                if news:
                    log(f"    STOCKS: {news}")
        except Exception:
            log(f"stock snapshot failed:\n{traceback.format_exc()}")

    def market_crash_check(self):
        """Every minute: the US market crash monitor on the live S&P 500 price (Binance SPY perpetual)."""
        try:
            spy = stocks.live_prices(["SPY"])["SPY"][0]
            msgs, _ = stocks.crash_check(lambda title, body: notify.send(title, body), spy)
            for m in msgs:
                log(f"    MARKET WARNING: {m}")
        except Exception as e:
            if not getattr(self, "_crash_failing", False):
                log(f"market crash check failed ({type(e).__name__}: {e}); will keep trying")
            self._crash_failing = True
            return
        self._crash_failing = False

    def stock_close(self, late=False):
        """Every minute: at the US market close (16:00 New York time, Monday to Friday) decide and fill the stock
        paper trades at that moment's Binance price, send stock alerts, then retrain the stock AI. After downtime
        (late=True, from catch_up) the last missed close is handled straight away, at the current price, and the
        trades say so. The last close handled is kept in STOCK_STATE, so a restart never skips or repeats one."""
        close_day = last_us_close()
        if self._stock_state().get("last_close") == close_day:
            return
        ny = datetime.now(NEW_YORK)
        if not late and ny.strftime("%Y-%m-%d") != close_day:
            return  # between closes: wait for the next one
        self._save_stock_state(close_day)  # once per close, even if a step below fails
        note = " (late: the service was offline at the US close)" if late else ""
        try:
            t0 = time.time()
            for t in stocks.STOCKS:
                stocks.candles(t, max_age_hours=0)  # the latest closes
            if stockpaper.is_open():
                n = stockpaper.step(force=late, note=note)
                log(f"    STOCKS paper for the {close_day} US close{note}: {n} trades [{time.time() - t0:.0f}s]")
            for s in stocks.alerts(lambda title, body: notify.send(title, body)):
                log(f"    STOCK ALERT: {s}")
            r = stockai.train_and_test()
            log(f"stock AI retrained: AUC {r['auc']['2023-now']:.3f} (2023-now); trading on it "
                f"{'beats' if r['trade_on_ai'] else 'does not beat'} holding [{time.time() - t0:.0f}s]")
        except Exception:
            log(f"stock close jobs failed:\n{traceback.format_exc()}")

    @staticmethod
    def _stock_state():
        try:
            return json.loads(STOCK_STATE.read_text())
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _save_stock_state(close_day):
        with config.atomic(STOCK_STATE) as tmp:
            tmp.write_text(json.dumps({"last_close": close_day}))

    def backfill(self):
        """After downtime: fill the paper trials' balance history for the hours missed, and handle a missed US close.
        Runs before the catch-up trades, while holdings are still as they were when the service stopped."""
        for book in paper.BOOKS:
            if paper.is_open(book):
                try:
                    n = paper.backfill_balance(book)
                    if n:
                        log(f"    backfilled {n} hourly balance points for the {book.key} paper trial")
                except Exception as e:
                    log(f"    balance backfill ({book.key}) failed: {type(e).__name__}: {e}")
        if stockpaper.is_open():
            try:
                n = stockpaper.backfill_balance()
                if n:
                    log(f"    backfilled {n} hourly balance points for the stock paper trial")
            except Exception as e:
                log(f"    stock balance backfill failed: {type(e).__name__}: {e}")
        self.stock_close(late=True)

    # ---------- streaming ----------
    def _url(self):
        return STREAM_URL + "/".join(f"{s}@kline_{tf}" for s in self.symbols for tf in self.timeframes)

    def _on_message(self, msg):
        k = json.loads(msg)["data"]["k"]
        self.status["last_message"] = _now()
        if not k["x"]:  # candle still forming
            return
        key = (k["i"], k["t"])
        first_seen, done = self.pending.setdefault(key, (time.time(), set()))
        done.add(k["s"].lower())

    def _process_ready(self):
        for key in sorted(self.pending):
            first_seen, done = self.pending[key]
            if len(done) == len(self.symbols) or time.time() - first_seen > WAIT_FOR_ALL:
                del self.pending[key]
                tf, open_ms = key
                closed_at = pd.Timestamp(open_ms, unit="ms", tz="UTC") + pd.Timedelta(tf)
                try:
                    self.learn(tf, f"candle closed {closed_at:%Y-%m-%d %H:%M}")
                except NETWORK_ERRORS as e:
                    # Reconnect and catch up straight away instead of waiting for the next candle.
                    raise ConnectionError(f"{tf} learning failed, no connection to Binance") from e
                except Exception:
                    log(f"{tf} learning failed, will retry at next candle:\n{traceback.format_exc()}")

    def run(self):
        log(f"Live service starting: {', '.join(self.symbols.values())} on {', '.join(self.timeframes)}")
        whales.WhaleWatch().start()  # background threads: big trades, liquidations, open interest (alerts only)
        backoff = 2
        while True:
            self.status["state"] = "catching up"
            self._save_status()
            self.catch_up()
            try:
                # No client keepalive pings: retraining at a candle close occupies the process for minutes,
                # so our own pings timed out and dropped the stream every time. Binance pings us instead; the
                # client answers in the background, but only while it keeps reading the socket. With the default
                # 16-message queue it stopped reading once the queue filled during retraining, missed Binance's
                # pings and was dropped ("Pong timeout"), so the queue is unbounded (a few hundred small messages).
                with connect(self._url(), open_timeout=15, close_timeout=5, max_size=2**20,
                             ping_interval=None, max_queue=None) as ws:
                    log("Connected to Binance kline stream; waiting for candles to close")
                    self.status["state"] = "listening"
                    self._save_status()
                    backoff = 2
                    while True:
                        try:
                            self._on_message(ws.recv(timeout=5))
                        except TimeoutError:
                            pass
                        self._process_ready()
                        if time.time() >= self._next_preview:
                            self.refresh_preview()
                        if time.time() - self._saved_at > HEARTBEAT:
                            self._save_status()
            except KeyboardInterrupt:
                log("Stopped")
                self.status["state"] = "stopped"
                self._save_status()
                return
            except Exception as e:  # network drop, Binance's 24h disconnect, PC waking from sleep
                log(f"Connection lost ({e}); reconnecting in {backoff}s and catching up")
                self.status["state"] = "reconnecting"
                self._save_status()
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)

    def _save_status(self):
        self._saved_at = time.time()
        self.status["updated"] = _now()
        with config.atomic(config.LIVE_STATUS) as tmp:
            tmp.write_text(json.dumps(self.status, indent=2))


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def advisor_timeframe():
    """Timeframe of the models the advice reads; it is re-checked after their candle closes."""
    return config.MODELS[advisor.DIRECTION]["timeframe"]


def load_status():
    if not config.LIVE_STATUS.exists():
        return None
    try:
        return json.loads(config.LIVE_STATUS.read_text())
    except (OSError, ValueError):
        return None
