"""Send alerts as Windows notifications and Zalo messages.

Settings live in notify.json next to the project (not on GitHub, since it holds your bot token):

    {"windows": true, "zalo": {"token": "123456:abc...", "chat_id": "6ede9afa66b88fe6d6a9"}}

Set up Zalo once with `python -m cryptoai zalo-setup <BOT_TOKEN>` (see the README).
"""
import base64
import csv
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import requests

from . import config

ZALO_API = "https://bot-api.zaloplatforms.com/bot{token}/{method}"
# Windows only shows notifications from registered apps; PowerShell's own app ID always is.
POWERSHELL_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
ALERT_LOG = config.LOG_DIR / "alerts.csv"  # every alert sent; synced to a viewing PC, which shows them too
VIEWER_STATE = config.ROOT / "viewer_state.json"  # the last alert this PC has shown (this PC only, never synced)


def settings():
    if not config.NOTIFY_FILE.exists():
        return {"windows": True, "zalo": None}
    return {"windows": True, "zalo": None, **json.loads(config.NOTIFY_FILE.read_text(encoding="utf-8"))}


def save_settings(s):
    config.NOTIFY_FILE.write_text(json.dumps(s, indent=2), encoding="utf-8")


def send(title, body):
    """Send to every channel that is switched on. Returns {channel: "sent" or an error message}."""
    s, results = settings(), {}
    try:
        _log_alert(title, body)
    except OSError:
        pass  # a locked log file never stops the alert itself
    if s.get("windows") and sys.platform == "win32":
        results["windows"] = _try(windows_toast, title, body)
    if s.get("zalo"):
        results["zalo"] = _try(zalo_send, s["zalo"]["token"], s["zalo"]["chat_id"], f"{title}\n\n{body}")
    return results


def _log_alert(title, body):
    new = not ALERT_LOG.exists()
    with open(ALERT_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "host", "title", "body"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), socket.gethostname(), title, body])


def recent_alerts(limit=50):
    """The latest alerts, newest first, as a list of dicts (time, host, title, body)."""
    if not ALERT_LOG.exists():
        return []
    with open(ALERT_LOG, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[::-1][:limit]


def mirror(every=15):
    """On a PC that only views: show the alerts another PC sends (logs/alerts.csv, synced here) as Windows
    notifications. Starts from now, so old alerts are not replayed; remembers the last one shown in VIEWER_STATE."""
    me = socket.gethostname()
    try:
        last = json.loads(VIEWER_STATE.read_text())["last_alert"]
    except (OSError, ValueError, KeyError):
        last = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC  alert viewer started; showing alerts after {last}",
          flush=True)
    while True:
        try:
            new = [r for r in recent_alerts(limit=10_000)[::-1] if r["time"] > last and r["host"] != me]
            for r in new:
                windows_toast(r["title"], r["body"])
                last = r["time"]
            if new:
                VIEWER_STATE.write_text(json.dumps({"last_alert": last}))
        except Exception as e:  # e.g. the file is being replaced by Syncthing right now
            print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC  {type(e).__name__}: {e}", flush=True)
        time.sleep(every)


def _try(fn, *args):
    try:
        fn(*args)
        return "sent"
    except Exception as e:
        return f"failed: {e}"


def windows_toast(title, body):
    """A Windows notification. It only appears when someone is logged in to this PC."""
    xml = (f"<toast><visual><binding template='ToastGeneric'><text>{escape(title)}</text>"
           f"<text>{escape(body)}</text></binding></visual></toast>")
    script = f"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null
$doc = New-Object Windows.Data.Xml.Dom.XmlDocument
$doc.LoadXml(@'
{xml}
'@)
$toast = New-Object Windows.UI.Notifications.ToastNotification $doc
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{POWERSHELL_APP_ID}').Show($toast)
"""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                   check=True, capture_output=True, timeout=30)


def zalo_send(token, chat_id, text):
    r = requests.post(ZALO_API.format(token=token, method="sendMessage"),
                      json={"chat_id": chat_id, "text": text[:2000]}, timeout=15)
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(f"Zalo said: {body}")


def zalo_find_chat_id(token):
    """The chat ID of the most recent person who messaged the bot (they must message it first)."""
    r = requests.post(ZALO_API.format(token=token, method="getUpdates"), json={"timeout": "10"}, timeout=20)
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(f"Zalo said: {body}")
    updates = body.get("result") or []
    if isinstance(updates, dict):
        updates = [updates]
    for u in reversed(updates):
        msg = u.get("message") or {}
        chat = msg.get("chat") or {}
        if chat.get("id"):
            return chat["id"], (msg.get("from") or {}).get("display_name", "")
    return None, None
