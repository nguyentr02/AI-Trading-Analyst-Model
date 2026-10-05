"""Login for opening the dashboard over the internet: username + password, then an access and a refresh token.

- The password is stored only as a salted PBKDF2 hash, in dashboard_auth.json (kept off GitHub), together
  with a random secret that signs the tokens.
- Access token: a signed JWT (HS256) valid for 15 minutes. It proves who is signed in.
- Refresh token: a signed JWT valid for 7 days, used to get a new access token without typing the password.
  Every use swaps it for a new one (rotation), and only the current ones are accepted: their IDs are kept in
  auth_tokens.json, so signing out revokes them.

Set or change the login with:  python -m cryptoai set-login
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time

from . import config

ITERATIONS = 200_000
ACCESS_TTL = 15 * 60
REFRESH_TTL = 7 * 24 * 3600


# ---------- the login ----------
def is_set():
    return config.AUTH_FILE.exists()


def _saved():
    return json.loads(config.AUTH_FILE.read_text())


def set_login(username, password):
    """Save the username and a hash of the password; also creates a new signing secret (signs everyone out)."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    config.AUTH_FILE.write_text(json.dumps({
        "username": username, "salt": salt.hex(), "hash": digest.hex(), "iterations": ITERATIONS,
        "secret": secrets.token_hex(32)}, indent=2))
    _save_sessions({})


def check_login(username, password):
    if not is_set():
        return False
    saved = _saved()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(saved["salt"]), saved["iterations"])
    return hmac.compare_digest(username, saved["username"]) & hmac.compare_digest(digest.hex(), saved["hash"])


# ---------- tokens (JWT, HS256) ----------
def _b64(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload):
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(bytes.fromhex(_saved()["secret"]), f"{header}.{body}".encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{_b64(sig)}"


def _decode(token, kind):
    """The token's payload if its signature, type and expiry are valid; otherwise None."""
    if not token or not is_set():
        return None
    try:
        header, body, sig = token.split(".")
        expected = hmac.new(bytes.fromhex(_saved()["secret"]), f"{header}.{body}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_unb64(sig), expected):
            return None
        payload = json.loads(_unb64(body))
    except (ValueError, KeyError):
        return None
    if payload.get("typ") != kind or payload.get("exp", 0) < time.time():
        return None
    return payload


def issue_tokens(username):
    """A new (access token, refresh token) pair after a successful sign-in or refresh."""
    now = int(time.time())
    jti = secrets.token_hex(16)
    sessions = _load_sessions()
    sessions[_hash(jti)] = now + REFRESH_TTL
    _save_sessions({k: exp for k, exp in sessions.items() if exp > now})  # drop expired ones
    access = _sign({"sub": username, "typ": "access", "iat": now, "exp": now + ACCESS_TTL})
    refresh = _sign({"sub": username, "typ": "refresh", "jti": jti, "iat": now, "exp": now + REFRESH_TTL})
    return access, refresh


def verify_access(token):
    """The signed-in username, or None if the access token is missing, forged or expired."""
    payload = _decode(token, "access")
    return payload["sub"] if payload else None


def refresh(refresh_token):
    """Swap a valid refresh token for a new pair (the old one stops working), or None if it isn't valid."""
    payload = _decode(refresh_token, "refresh")
    if not payload:
        return None
    sessions = _load_sessions()
    if sessions.pop(_hash(payload["jti"]), None) is None:
        return None  # already used or signed out
    _save_sessions(sessions)
    return issue_tokens(payload["sub"])


def revoke(refresh_token):
    """Sign out: the refresh token can no longer be used."""
    payload = _decode(refresh_token, "refresh")
    if payload:
        sessions = _load_sessions()
        sessions.pop(_hash(payload["jti"]), None)
        _save_sessions(sessions)


def _hash(jti):
    return hashlib.sha256(jti.encode()).hexdigest()


def _load_sessions():
    if not config.AUTH_SESSIONS_FILE.exists():
        return {}
    try:
        return json.loads(config.AUTH_SESSIONS_FILE.read_text())
    except ValueError:
        return {}


def _save_sessions(sessions):
    with config.atomic(config.AUTH_SESSIONS_FILE) as tmp:
        tmp.write_text(json.dumps(sessions))
