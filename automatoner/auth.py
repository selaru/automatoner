"""Access token for the web interface and API.

Off by default on this computer. Required whenever the server listens on a non-loopback address, and enforced whenever a
token exists. The token comes from AUTOMATONER_TOKEN, else from the `token` file in the data folder (owner-only permissions).
Create one with `automatoner token create`. The browser signs in once and receives an HttpOnly cookie; scripts send
`Authorization: Bearer <token>`.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import stat
import sys
from pathlib import Path

from . import config

MIN_LENGTH = 20
COOKIE = "automatoner_session"


class TokenRequired(ValueError):
    pass


def token_path() -> Path:
    return config.home() / "token"


def configured_token() -> str | None:
    env = os.environ.get("AUTOMATONER_TOKEN", "").strip()
    if env:
        if len(env) < MIN_LENGTH:
            raise TokenRequired(f"AUTOMATONER_TOKEN is too short (use at least {MIN_LENGTH} characters). Generate one with: automatoner token create")
        return env
    p = token_path()
    if p.exists():
        t = p.read_text(encoding="utf-8").strip()
        return t or None
    return None


def source() -> str | None:
    if os.environ.get("AUTOMATONER_TOKEN", "").strip():
        return "environment variable AUTOMATONER_TOKEN"
    return str(token_path()) if token_path().exists() and token_path().read_text(encoding="utf-8").strip() else None


def create_token(rotate: bool = False) -> str:
    p = token_path()
    if p.exists() and p.read_text(encoding="utf-8").strip() and not rotate:
        raise ValueError("a token already exists. Use `automatoner token rotate` to replace it.")
    t = secrets.token_urlsafe(32)
    p.write_text(t + "\n", encoding="utf-8")
    if sys.platform != "win32":
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    return t


def remove_token() -> bool:
    p = token_path()
    if p.exists():
        p.unlink()
        return True
    return False


def session_value(token: str) -> str:
    """What the browser cookie holds: derived from the token, so the token itself is never stored in the browser."""
    return hmac.new(token.encode(), b"automatoner-ui-session", hashlib.sha256).hexdigest()


def same(a: str | None, b: str) -> bool:
    return a is not None and hmac.compare_digest(a.encode(), b.encode())


def authorised(token: str, authorization: str | None, cookie: str | None) -> bool:
    if authorization and authorization.lower().startswith("bearer ") and same(authorization[7:].strip(), token):
        return True
    return same(cookie, session_value(token))


def require_for_remote(token: str | None) -> None:
    if not token:
        raise TokenRequired(
            "refusing to listen on a non-loopback address without an access token. "
            "Run `automatoner token create` (or set AUTOMATONER_TOKEN to a random value of 20+ characters), then start again. "
            "Put the server behind TLS (a VPN or reverse proxy) as well: the token is sent in clear text over plain HTTP.")
