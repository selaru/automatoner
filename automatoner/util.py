from __future__ import annotations

import re
import secrets
import time
import uuid
from datetime import datetime, timezone


def new_id(prefix: str = "") -> str:
    return f"{prefix}{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return s or "item"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"
