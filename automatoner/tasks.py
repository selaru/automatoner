"""In-process background tasks (dataset generation, downloads, installs, exports).

Training is different: it runs in its own subprocess, see training/runner.py.
"""
from __future__ import annotations

import contextvars
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from . import util


@dataclass
class Task:
    id: str
    kind: str
    title: str
    status: str = "running"  # running | done | error | cancelled
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    log: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: float | None = None
    cancel_flag: threading.Event = field(default_factory=threading.Event)

    def say(self, line: str) -> None:
        self.log.append(line.rstrip("\n"))
        if len(self.log) > 2000:
            del self.log[:500]

    def set(self, progress: float | None = None, message: str | None = None) -> None:
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        if message is not None:
            self.message = message

    @property
    def cancelled(self) -> bool:
        return self.cancel_flag.is_set()

    def public(self, tail: int = 200) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
            "progress": self.progress, "message": self.message, "result": self.result,
            "error": self.error, "log": self.log[-tail:], "started": self.started, "finished": self.finished,
        }


_tasks: dict[str, Task] = {}
_lock = threading.Lock()


def start(kind: str, title: str, fn: Callable[[Task], Any], *, id: str | None = None) -> Task:
    task = Task(id=id or util.new_id("t-"), kind=kind, title=title)
    with _lock:
        _tasks[task.id] = task

    def runner() -> None:
        try:
            task.result = fn(task)
            task.status = "cancelled" if task.cancelled else "done"
            if task.status == "done":
                task.progress = 1.0
        except Exception as e:  # noqa: BLE001
            task.status = "error"
            task.error = f"{type(e).__name__}: {e}"
            task.say(traceback.format_exc())
        finally:
            task.finished = time.time()

    ctx = contextvars.copy_context()  # keeps the audit actor attached to work done in the background
    threading.Thread(target=lambda: ctx.run(runner), daemon=True, name=f"task-{task.id}").start()
    return task


def get(id: str) -> Task | None:
    return _tasks.get(id)


def all_tasks(kind: str | None = None) -> list[Task]:
    with _lock:
        items = list(_tasks.values())
    items.sort(key=lambda t: t.started, reverse=True)
    return [t for t in items if kind is None or t.kind == kind]
