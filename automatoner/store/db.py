"""Tiny SQLite document store. One table per kind, each row is an id plus a JSON document."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from .. import config

KINDS = ("jobs", "datasets", "exports", "downloads")
_local = threading.local()


def _db_path():
    return config.home() / "automatoner.db"


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    path = str(_db_path())
    conn = getattr(_local, "conn", None)
    if conn is None or getattr(_local, "path", None) != path:
        conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        for k in KINDS:
            conn.execute(f"CREATE TABLE IF NOT EXISTS {k} (id TEXT PRIMARY KEY, created REAL, doc TEXT NOT NULL)")
        conn.commit()
        _local.conn, _local.path = conn, path
    yield conn


def put(kind: str, id: str, doc: dict[str, Any]) -> dict[str, Any]:
    assert kind in KINDS
    with _conn() as c:
        row = c.execute(f"SELECT created FROM {kind} WHERE id=?", (id,)).fetchone()
        created = row[0] if row else time.time()
        c.execute(f"INSERT OR REPLACE INTO {kind} (id, created, doc) VALUES (?,?,?)", (id, created, json.dumps(doc)))
        c.commit()
    return doc


def get(kind: str, id: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(f"SELECT doc FROM {kind} WHERE id=?", (id,)).fetchone()
    return json.loads(row[0]) if row else None


def update(kind: str, id: str, **fields: Any) -> dict[str, Any] | None:
    doc = get(kind, id)
    if doc is None:
        return None
    doc.update(fields)
    put(kind, id, doc)
    return doc


def list_(kind: str) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(f"SELECT doc FROM {kind} ORDER BY created DESC").fetchall()
    return [json.loads(r[0]) for r in rows]


def delete(kind: str, id: str) -> None:
    with _conn() as c:
        c.execute(f"DELETE FROM {kind} WHERE id=?", (id,))
        c.commit()
