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
from datetime import datetime, timezone

import ccxt
import pandas as pd
import requests
from websockets.sync.client import connect

from . import advisor, config, model, preview, signals

STREAM_URL = "wss://stream.binance.com:9443/stream?streams="
WAIT_FOR_ALL = 20  # seconds to wait for every symbol's close message before processing anyway
HEARTBEAT = 60  # seconds between status file updates, so the dashboard can tell the service is alive


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

    # ---------- learning ----------
    def learn(self, tf, reason):
        """Fetch new data, retrain, and log the signals for every model built on timeframe `tf`."""
        for name in [n for n, s in config.MODELS.items() if s["timeframe"] == tf]:
            self._learn_model(name, reason)
        if tf == advisor_timeframe():
            self.advise()

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
            preview.compute()
            self.status["last_preview"] = _now()
            self._preview_failing = False
        except Exception as e:
            if not self._preview_failing:  # log the first failure of a run of them, not one per minute
                log(f"live preview failed ({type(e).__name__}: {e}); will keep trying every minute")
            self._preview_failing = True

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
        backoff = 2
        while True:
            self.status["state"] = "catching up"
            self._save_status()
            self.catch_up()
            try:
                with connect(self._url(), open_timeout=15, close_timeout=5, max_size=2**20) as ws:
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
