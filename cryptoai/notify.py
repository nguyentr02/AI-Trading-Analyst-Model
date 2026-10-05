"""Send alerts as Windows notifications and Zalo messages.

Settings live in notify.json next to the project (not on GitHub, since it holds your bot token):

    {"windows": true, "zalo": {"token": "123456:abc...", "chat_id": "6ede9afa66b88fe6d6a9"}}

Set up Zalo once with `python -m cryptoai zalo-setup <BOT_TOKEN>` (see the README).
"""
import base64
import json
import subprocess
import sys
from xml.sax.saxutils import escape

import requests

from . import config

ZALO_API = "https://bot-api.zaloplatforms.com/bot{token}/{method}"
# Windows only shows notifications from registered apps; PowerShell's own app ID always is.
POWERSHELL_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"


def settings():
    if not config.NOTIFY_FILE.exists():
        return {"windows": True, "zalo": None}
    return {"windows": True, "zalo": None, **json.loads(config.NOTIFY_FILE.read_text(encoding="utf-8"))}


def save_settings(s):
    config.NOTIFY_FILE.write_text(json.dumps(s, indent=2), encoding="utf-8")


def send(title, body):
    """Send to every channel that is switched on. Returns {channel: "sent" or an error message}."""
    s, results = settings(), {}
    if s.get("windows") and sys.platform == "win32":
        results["windows"] = _try(windows_toast, title, body)
    if s.get("zalo"):
        results["zalo"] = _try(zalo_send, s["zalo"]["token"], s["zalo"]["chat_id"], f"{title}\n\n{body}")
    return results


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
