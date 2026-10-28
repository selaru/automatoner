"""Create, launch, monitor and cancel training jobs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .. import config, util
from ..catalogue import service as catalogue
from ..datasets import service as datasets
from ..store import db
from . import plugins

_procs: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()


class JobError(ValueError):
    pass


def job_dir(job_id: str) -> Path:
    return util.safe_child(config.jobs_dir(), job_id)


def defaults_for(plugin_id: str, category: str, entry: dict[str, Any]) -> dict[str, Any]:
    plugin = plugins.load(plugin_id)
    cfg = {f["key"]: f["default"] for f in plugin.schema(category, entry)}
    cfg.update({k: v for k, v in (entry.get("defaults") or {}).items() if k not in ("mode",) or True})
    return cfg


def schema_for(plugin_id: str, category: str, model_id: str) -> list[dict[str, Any]]:
    entry = catalogue.get_entry(model_id)
    if entry is None:
        raise JobError(f"unknown model {model_id}")
    return plugins.load(plugin_id).schema(category, entry)


def create(model_id: str, dataset_id: str, trainer: str | None, category: str | None, user_config: dict[str, Any] | None) -> dict[str, Any]:
    entry = catalogue.get_entry(model_id)
    if entry is None:
        raise JobError(f"unknown model {model_id}")
    ds = datasets.get(dataset_id)
    if ds is None:
        raise JobError(f"unknown dataset {dataset_id}")
    if ds.get("status") != "ready":
        raise JobError(f"dataset is {ds.get('status')}; fix it first ({'; '.join((ds.get('report') or {}).get('errors', []))})")
    category = category or ds["category"]
    if category != ds["category"]:
        raise JobError(f"dataset is for {ds['category']}, not {category}")
    if category not in entry["categories"]:
        raise JobError(f"{entry['name']} cannot be trained for {category}")
    options = catalogue.entry_view(entry)["trainer_options"].get(category, [])
    if not options:
        raise JobError(f"Training {entry['name']} for {category} is not available yet.")
    trainer = trainer or options[0]
    if trainer not in options:
        raise JobError(f"trainer {trainer} is not available for this model; choose one of {options}")
    av = plugins.available(trainer)
    if not av["host_ok"]:
        raise JobError(f"{av['name']} does not run on this computer ({av['hardware']}).")
    if av["missing"]:
        raise JobError(f"{av['name']} needs the '{av['extra']}' extra (missing: {', '.join(av['missing'])}). Install it from the Doctor page.")
    if not catalogue.is_ready(model_id):
        raise JobError(f"{entry['name']} is not downloaded yet. Download it from the Models page first.")

    cfg = defaults_for(trainer, category, entry)
    cfg.update(user_config or {})
    job_id = util.new_id("job-")
    jd = job_dir(job_id)
    jd.mkdir(parents=True)
    doc = {"id": job_id, "status": "queued", "model_id": model_id, "model_name": entry["name"], "dataset_id": dataset_id,
           "dataset_name": ds["name"], "trainer": trainer, "category": category, "config": cfg, "created": util.now_iso(),
           "started": None, "finished": None, "error": None, "summary": None}
    (jd / "job.json").write_text(json.dumps({
        **doc, "model": entry, "model_path": catalogue.local_path(model_id), "dataset": ds,
        "dataset_dir": str(datasets.dataset_dir(dataset_id)),
    }, indent=2), encoding="utf-8")
    db.put("jobs", job_id, doc)
    _launch(job_id)
    return db.get("jobs", job_id)  # type: ignore[return-value]


def _launch(job_id: str) -> None:
    jd = job_dir(job_id)
    log = (jd / "job.log").open("ab")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false"}
    kw: dict[str, Any] = {}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen([sys.executable, "-m", "automatoner.training.worker", str(jd)], stdout=log, stderr=subprocess.STDOUT,
                            cwd=str(jd), env=env, **kw)
    with _lock:
        _procs[job_id] = proc
    db.update("jobs", job_id, status="running", started=util.now_iso(), pid=proc.pid)
    threading.Thread(target=_monitor, args=(job_id, proc), daemon=True, name=f"job-{job_id}").start()


def _worker_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE and any("automatoner.training.worker" in part for part in p.cmdline())
    except Exception:  # noqa: BLE001 - gone, or not ours
        return False


def _finalize(job_id: str, code: int | None = None) -> dict[str, Any] | None:
    """Record how a finished worker ended. Safe to call from any process, and more than once."""
    jd = job_dir(job_id)
    cur = db.get("jobs", job_id)
    if cur is None or cur.get("status") in ("done", "error", "cancelled", "interrupted"):
        return cur
    status: dict[str, Any] = {"status": "interrupted", "error": "the worker stopped without reporting a result"} if code is None else \
        {"status": "error", "error": f"worker exited with code {code}"}
    sf = jd / "status.json"
    if sf.exists():
        status = json.loads(sf.read_text(encoding="utf-8"))
    summary = None
    mf = jd / "model" / "manifest.json"
    if status["status"] == "done" and mf.exists():
        m = json.loads(mf.read_text(encoding="utf-8"))
        summary = {"metrics": m.get("metrics"), "size_bytes": m.get("size_bytes"), "artifacts": list((m.get("artifacts") or {}).keys())}
    return db.update("jobs", job_id, status=status["status"], error=status.get("error"), finished=util.now_iso(), summary=summary)


def _monitor(job_id: str, proc: subprocess.Popen) -> None:
    code = proc.wait()
    cur = db.get("jobs", job_id) or {}
    if cur.get("status") == "cancelled":
        db.update("jobs", job_id, finished=util.now_iso())
    else:
        _finalize(job_id, code)
    with _lock:
        _procs.pop(job_id, None)


def refresh(job_id: str) -> dict[str, Any] | None:
    """The job record, brought up to date. A job whose worker has ended is finalized even if the process that started it is gone."""
    doc = db.get("jobs", job_id)
    if doc and doc.get("status") in ("queued", "running") and job_id not in _procs and not _worker_alive(doc.get("pid")):
        return _finalize(job_id)
    return doc


def list_jobs() -> list[dict[str, Any]]:
    return [refresh(j["id"]) or j for j in db.list_("jobs")]


def cancel(job_id: str) -> bool:
    with _lock:
        proc = _procs.get(job_id)
    doc = db.get("jobs", job_id) or {}
    pid = proc.pid if proc is not None and proc.poll() is None else (doc.get("pid") if _worker_alive(doc.get("pid")) else None)
    if pid is None:
        return False
    db.update("jobs", job_id, status="cancelled")
    try:
        import psutil
        parent = psutil.Process(pid)
        kids = parent.children(recursive=True)
        for p in kids + [parent]:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(kids + [parent], timeout=5)
        for p in kids + [parent]:
            if p.is_running():
                p.kill()
    except Exception:  # noqa: BLE001
        if proc is not None:
            proc.kill()
    return True


def mark_interrupted() -> None:
    """On startup, settle jobs left running. Jobs whose worker is still alive (started by another process) are left alone."""
    for j in db.list_("jobs"):
        if j["status"] in ("running", "queued"):
            refresh(j["id"])


def read_log(job_id: str, offset: int = 0) -> tuple[str, int]:
    p = job_dir(job_id) / "job.log"
    if not p.exists():
        return "", 0
    with p.open("rb") as f:
        f.seek(offset)
        data = f.read()
    return data.decode("utf-8", errors="replace"), offset + len(data)


def series(job_id: str) -> list[dict[str, Any]]:
    text, _ = read_log(job_id)
    out = []
    for line in text.splitlines():
        if line.startswith("@@ "):
            try:
                out.append(json.loads(line[3:]))
            except json.JSONDecodeError:
                pass
    return out


def manifest(job_id: str) -> dict[str, Any] | None:
    mf = job_dir(job_id) / "model" / "manifest.json"
    return json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else None


def delete(job_id: str) -> None:
    cancel(job_id)
    import shutil
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
    db.delete("jobs", job_id)
