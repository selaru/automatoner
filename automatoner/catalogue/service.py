"""Catalogue: merge catalogue.yaml with local download state, and download models on request."""
from __future__ import annotations

import shutil
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .. import audit, config, tasks
from ..store import db

CATALOGUE_FILE = Path(__file__).with_name("catalogue.yaml")


@lru_cache(maxsize=1)
def _raw() -> dict[str, Any]:
    return yaml.safe_load(CATALOGUE_FILE.read_text(encoding="utf-8"))


def reload() -> None:
    _raw.cache_clear()


def categories() -> list[dict[str, Any]]:
    return _raw()["categories"]


def _trainer_list(entry: dict[str, Any], category: str) -> list[str]:
    t = (entry.get("trainers") or {}).get(category)
    if t is None:
        return []
    return list(t) if isinstance(t, list) else [t]


def _local_dir(entry: dict[str, Any]) -> Path:
    return config.models_dir() / entry["id"]


def _state(entry: dict[str, Any]) -> dict[str, Any]:
    if entry["source"] == "builtin":
        return {"status": "builtin", "path": None}
    rec = db.get("downloads", entry["id"])
    if rec and rec.get("status") == "done" and Path(rec["path"]).exists():
        return {"status": "downloaded", "path": rec["path"], "bytes": rec.get("bytes")}
    t = tasks.get(f"dl-{entry['id']}")
    if t and t.status == "running":
        return {"status": "downloading", "path": None, "progress": t.progress, "message": t.message}
    if t and t.status == "error":
        return {"status": "error", "path": None, "error": t.error}
    return {"status": "not_downloaded", "path": None}


def entry_view(entry: dict[str, Any]) -> dict[str, Any]:
    cats = {}
    for c in entry["categories"]:
        cats[c] = _trainer_list(entry, c)
    return {**entry, "trainable": {c: bool(t) for c, t in cats.items()}, "trainer_options": cats, "state": _state(entry)}


def list_models(category: str | None = None) -> list[dict[str, Any]]:
    out = []
    for e in _raw()["models"]:
        if category and category not in e["categories"]:
            continue
        out.append(entry_view(e))
    return out


def get_entry(model_id: str) -> dict[str, Any] | None:
    for e in _raw()["models"]:
        if e["id"] == model_id:
            return e
    return None


def local_path(model_id: str) -> str | None:
    """Path to downloaded weights, or None for builtin algorithms / not downloaded."""
    e = get_entry(model_id)
    if not e or e["source"] == "builtin":
        return None
    s = _state(e)
    return s["path"]


def is_ready(model_id: str) -> bool:
    e = get_entry(model_id)
    if not e:
        return False
    return e["source"] == "builtin" or _state(e)["status"] == "downloaded"


# --- download ----------------------------------------------------------------------------------

def start_download(model_id: str) -> tasks.Task:
    entry = get_entry(model_id)
    if entry is None:
        raise KeyError(model_id)
    if entry["source"] == "builtin":
        raise ValueError("Builtin algorithms need no download.")
    audit.record("model.download", model=model_id, repo=entry.get("repo"))
    existing = tasks.get(f"dl-{model_id}")
    if existing and existing.status == "running":
        return existing

    def run(task: tasks.Task) -> dict[str, Any]:
        from huggingface_hub import HfApi, snapshot_download
        from tqdm.auto import tqdm

        token = config.hf_token()

        total = 0
        try:
            info = HfApi().model_info(entry["repo"], token=token, files_metadata=True)
            total = sum((s.size or 0) for s in info.siblings or [])
        except Exception as e:  # noqa: BLE001 - gated or offline; the download itself will report it
            task.say(f"could not read repo metadata: {e}")
        done = {"n": 0}
        lock = threading.Lock()

        class Progress(tqdm):  # type: ignore[misc]
            def __init__(self, *a: Any, **kw: Any) -> None:
                kw["disable"] = False
                super().__init__(*a, **kw)

            def update(self, n: float | None = 1) -> bool | None:
                r = super().update(n)
                if self.unit == "B" and n:
                    with lock:
                        done["n"] += int(n)
                    if total:
                        task.set(done["n"] / total, f"{done['n'] / 1e6:.0f} of {total / 1e6:.0f} MB")
                if task.cancelled:
                    raise InterruptedError("download cancelled")
                return r

        task.say(f"fetching {entry['repo']} (Hugging Face cache; files already there are reused)")
        path = snapshot_download(repo_id=entry["repo"], token=token, tqdm_class=Progress, allow_patterns=entry.get("allow_patterns"))
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        db.put("downloads", model_id, {"id": model_id, "status": "done", "path": path, "bytes": size, "repo": entry["repo"]})
        return {"path": path, "bytes": size}

    return tasks.start("download", f"Download {entry['name']}", run, id=f"dl-{model_id}")


def remove(model_id: str) -> None:
    """Forget the model and delete its files from the Hugging Face cache."""
    audit.record("model.remove", model=model_id)
    entry = get_entry(model_id)
    if entry is None:
        raise KeyError(model_id)
    if entry["source"] == "hf":
        try:
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir()
            for repo in info.repos:
                if repo.repo_id == entry["repo"] and repo.repo_type == "model":
                    info.delete_revisions(*[r.commit_hash for r in repo.revisions]).execute()
        except Exception:  # noqa: BLE001 - cache layout differences must not block forgetting the record
            pass
    shutil.rmtree(_local_dir(entry), ignore_errors=True)
    db.delete("downloads", model_id)
