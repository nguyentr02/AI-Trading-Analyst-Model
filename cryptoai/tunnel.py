"""Put the dashboard online through a free Cloudflare quick tunnel, and send you the link.

    python -m cryptoai tunnel

Runs Cloudflare's `cloudflared` (downloaded into tools/ on first use) to give the local dashboard a public
https://<random>.trycloudflare.com address. The address changes whenever the tunnel restarts, so every new
one is saved to logs/public_url.txt and sent to you (Windows notification and Zalo). Refuses to run until a
dashboard login is set, so the dashboard is never online unprotected.
"""
import re
import subprocess
import time
from datetime import datetime, timezone

import requests

from . import auth, config, notify

DOWNLOAD = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def log(msg):
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC  {msg}", flush=True)


def cloudflared():
    """Path to cloudflared.exe, downloading Cloudflare's official build the first time."""
    exe = config.ROOT / "tools" / "cloudflared.exe"
    if not exe.exists():
        exe.parent.mkdir(exist_ok=True)
        log(f"Downloading cloudflared from {DOWNLOAD}")
        r = requests.get(DOWNLOAD, timeout=120)
        r.raise_for_status()
        tmp = exe.with_suffix(".tmp")
        tmp.write_bytes(r.content)
        tmp.replace(exe)
    return exe


def public_url():
    """The dashboard's current public address, or None when the tunnel isn't running."""
    if not config.PUBLIC_URL_FILE.exists():
        return None
    return config.PUBLIC_URL_FILE.read_text().strip() or None


def run(port=8501):
    if not auth.is_set():
        raise SystemExit("No dashboard login set. Run `python -m cryptoai set-login` first; "
                         "the tunnel won't put the dashboard online without one.")
    exe = cloudflared()
    while True:
        log("Starting Cloudflare tunnel")
        proc = subprocess.Popen([str(exe), "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace")
        try:
            for line in proc.stdout:
                m = URL_PATTERN.search(line)
                if m:
                    url = m.group(0)
                    if url != public_url():
                        config.PUBLIC_URL_FILE.write_text(url)
                        log(f"Dashboard online at {url}")
                        results = notify.send("Crypto AI dashboard link",
                                              f"Your dashboard is online at {url}\nSign in with your dashboard username and password.")
                        log(f"Link sent: {results}")
        finally:
            proc.kill()
            if config.PUBLIC_URL_FILE.exists():
                config.PUBLIC_URL_FILE.unlink()
        log(f"Tunnel stopped (exit code {proc.wait()}); restarting in 15s")
        time.sleep(15)
