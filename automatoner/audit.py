"""Append-only audit log: who did what, when. One JSON object per line.

Never records secrets, prompts, dataset rows or model output. LLM calls are logged by size only. The file lives in the data
folder (owner-only permissions) and is rotated at 10 MB. Point AUTOMATONER_AUDIT_FILE somewhere else, or ship the file to your
log system, if you need it kept elsewhere.
"""
from __future__ import annotations

import contextvars
import json
import os
import stat
import sys
import threading
from pathlib import Path
from typing import Any

from . import config, util

MAX_BYTES = 10 * 1024 * 1024
KEEP = 5
_actor: contextvars.ContextVar[str] = contextvars.ContextVar("automatoner_actor", default="system")
_lock = threading.Lock()
_SECRETISH = ("key", "token", "secret", "password", "prompt", "content", "text")


def set_actor(name: str) -> contextvars.Token:
    return _actor.set(name)


def reset_actor(token: contextvars.Token) -> None:
    _actor.reset(token)


def actor() -> str:
    return _actor.get()


def path() -> Path:
    env = os.environ.get("AUTOMATONER_AUDIT_FILE")
    return Path(env).expanduser() if env else config.home() / "audit.log"


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 200 else value[:200] + "..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): ("<hidden>" if any(s in str(k).lower() for s in _SECRETISH) else _clean(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in list(value)[:50]]
    return str(value)[:200]


def _rotate(p: Path) -> None:
    if not p.exists() or p.stat().st_size < MAX_BYTES:
        return
    for i in range(KEEP - 1, 0, -1):
        src = p.with_name(f"{p.name}.{i}")
        if src.exists():
            src.replace(p.with_name(f"{p.name}.{i + 1}"))
    p.replace(p.with_name(f"{p.name}.1"))


def record(action: str, **details: Any) -> None:
    """Add one entry. Never raises: a full disk must not stop training, but the failure is reported on stderr."""
    entry = {"time": util.now_iso(), "actor": actor(), "action": action, **_clean(details)}
    try:
        p = path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            _rotate(p)
            new = not p.exists()
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if new and sys.platform != "win32":
                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as e:
        print(f"audit log write failed: {e}", file=sys.stderr)


def read(limit: int = 100, action: str | None = None) -> list[dict[str, Any]]:
    """Newest entries first."""
    p = path()
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in reversed(p.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if action and not str(e.get("action", "")).startswith(action):
            continue
        out.append(e)
        if len(out) >= limit:
            break
    return out
