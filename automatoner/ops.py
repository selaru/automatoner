"""Operations shared by the command line and the MCP server.

Everything returns plain JSON-able dicts and raises ValueError with a readable message. Nothing here prints.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from . import config, recipes as recipes_mod, tasks
from .catalogue import service as catalogue
from .datasets import service as datasets
from .doctor import checks as doctor_checks
from .export import service as exports
from .store import db
from .training import runner


# --- read -------------------------------------------------------------------------------------------

def doctor() -> dict[str, Any]:
    results = doctor_checks.run_all()
    return {"summary": doctor_checks.summary(results), "checks": results}


def categories() -> list[dict[str, Any]]:
    models = catalogue.list_models()
    return [{"id": c["id"], "name": c["name"], "what_it_does": c["blurb"], "dataset_format": c["dataset"],
             "models": sum(1 for m in models if c["id"] in m["categories"]),
             "trainable_models": sum(1 for m in models if m["trainable"].get(c["id"]))} for c in catalogue.categories()]


def models(category: str | None = None) -> list[dict[str, Any]]:
    out = []
    for m in catalogue.list_models(category):
        out.append({"id": m["id"], "name": m["name"], "params": m["params"], "size_mb": m["size_mb"], "license": m["license"],
                    "gated": bool(m.get("gated")), "status": m["state"]["status"], "good_for": m["good_for"],
                    "trainers": {c: t for c, t in m["trainer_options"].items() if not category or c == category}})
    return out


def datasets_list(category: str | None = None) -> list[dict[str, Any]]:
    return [dataset_summary(d) for d in datasets.list_datasets(category)]


def dataset_summary(d: dict[str, Any]) -> dict[str, Any]:
    return {k: d.get(k) for k in ("id", "name", "category", "source", "status", "rows", "splits", "label_column", "labels", "created")} | \
        {"errors": (d.get("report") or {}).get("errors", []), "warnings": (d.get("report") or {}).get("warnings", []), "error": d.get("error")}


def dataset(ds_id: str, preview_rows: int = 5) -> dict[str, Any]:
    d = datasets.get(ds_id)
    if d is None:
        raise ValueError(f"unknown dataset {ds_id}")
    out = dataset_summary(d)
    out["columns"] = d.get("columns")
    if d.get("status") == "ready" and preview_rows > 0:
        out["preview"] = datasets.preview(ds_id, min(preview_rows, 20))
    return out


def recipes_list() -> list[dict[str, Any]]:
    return [{"id": r["id"], "name": r["name"], "what_it_does": r["blurb"], "category": r["category"], "model": r["model"],
             "needs_llm": bool(r.get("needs_llm")), "export": (r.get("export") or {}).get("exporter")} for r in recipes_mod.list_recipes()]


def _metrics_of(doc: dict[str, Any]) -> Any:
    return (doc.get("summary") or {}).get("metrics")


def job_summary(doc: dict[str, Any]) -> dict[str, Any]:
    return {"id": doc["id"], "status": doc["status"], "model": doc["model_id"], "dataset": doc["dataset_id"], "dataset_name": doc.get("dataset_name"),
            "trainer": doc["trainer"], "category": doc["category"], "created": doc.get("created"), "finished": doc.get("finished"),
            "error": doc.get("error"), "metrics": _metrics_of(doc)}


def jobs(limit: int = 20) -> list[dict[str, Any]]:
    return [job_summary(j) for j in runner.list_jobs()[:limit]]


def job(job_id: str, log_lines: int = 0) -> dict[str, Any]:
    doc = runner.refresh(job_id)
    if doc is None:
        raise ValueError(f"unknown job {job_id}")
    out = job_summary(doc) | {"config": doc.get("config")}
    m = runner.manifest(job_id)
    if m:
        out["size_bytes"] = m.get("size_bytes")
        out["artifacts"] = list((m.get("artifacts") or {}).keys())
        out["threshold_source"] = m.get("threshold_source")
    if log_lines:
        out["log_tail"] = log_tail(job_id, log_lines)
    return out


def log_tail(job_id: str, lines: int = 40, width: int = 300) -> list[str]:
    text, _ = runner.read_log(job_id)
    rows = [ln[:width] for ln in text.splitlines() if ln.strip() and not ln.startswith("@@ ")]
    return rows[-lines:]


def export_options(job_id: str) -> list[dict[str, Any]]:
    try:
        opts = exports.options_for(job_id)
    except KeyError:
        raise ValueError(f"unknown job {job_id}")
    return [{"id": o["id"], "name": o["name"], "available": o["available"], "reason": o["reason"] or None, "what_it_makes": o["blurb"],
             "options": [{"key": f["key"], "default": f.get("default"), "choices": f.get("options")} for f in o["options"]]} for o in opts]


def task(task_id: str, log_lines: int = 15) -> dict[str, Any]:
    t = tasks.get(task_id)
    if t is None:
        raise ValueError(f"unknown task {task_id}")
    pub = t.public(tail=log_lines)
    pub["log"] = [ln[:300] for ln in pub["log"]]
    return {k: pub[k] for k in ("id", "kind", "title", "status", "progress", "message", "result", "error", "log")}


# --- act ----------------------------------------------------------------------------------------------

def start_download(model_id: str) -> dict[str, Any]:
    try:
        t = catalogue.start_download(model_id)
    except KeyError:
        raise ValueError(f"unknown model {model_id}")
    return {"task_id": t.id}


def add_dataset_file(path: str | Path, name: str, category: str, label_column: str | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"{p} is not a file")
    opts = dict(options or {})
    if label_column:
        opts["label_column"] = label_column
    return dataset_summary(datasets.import_upload(name, category, [(p.name, p.read_bytes())], opts))


def start_generation(name: str, category: str, spec: dict[str, Any], provider: str | None = None, model: str | None = None) -> dict[str, Any]:
    task_, ds_id = datasets.start_synthetic(name, category, spec, provider=provider, model=model)
    return {"task_id": task_.id, "dataset_id": ds_id}


def generation_size(category: str, spec: dict[str, Any]) -> int:
    """How many examples a generation request would produce, for limits."""
    if category.startswith("tabular"):
        return int(spec.get("rows", 5000))
    if category == "text_classification":
        return int(spec.get("per_label", 100)) * max(1, len(spec.get("labels", [])))
    return int(spec.get("total", 200))


def uses_llm(category: str, spec: dict[str, Any]) -> bool:
    return not category.startswith("tabular") or bool(spec.get("describe"))


def start_training(model_id: str, dataset_id: str, trainer: str | None = None, config_: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return job_summary(runner.create(model_id, dataset_id, trainer, None, config_ or {}))
    except runner.JobError as e:
        raise ValueError(str(e))


def cancel_job(job_id: str) -> dict[str, Any]:
    return {"cancelled": runner.cancel(job_id)}


def start_export(job_id: str, exporter: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        t, export_id = exports.start(job_id, exporter, options or {})
    except Exception as e:  # noqa: BLE001
        raise ValueError(str(e))
    return {"task_id": t.id, "export_id": export_id}


def export_result(export_id: str) -> dict[str, Any]:
    e = db.get("exports", export_id)
    if e is None:
        raise ValueError(f"unknown export {export_id}")
    r = e.get("result") or {}
    files = r.get("files") or []
    return {"id": export_id, "status": e["status"], "exporter": e["exporter"], "error": e.get("error"),
            "location": r.get("location") or (str(config.exports_dir() / export_id / r["root"]) if r.get("root") else None),
            "download_zip": str(config.exports_dir() / r["download"]) if r.get("download") else None,
            "files": [f["path"] for f in files[:30]], "notes": r.get("notes"), "smoke_test": r.get("smoke_test")}


def start_recipe(recipe_id: str, provider: str | None = None, model: str | None = None) -> dict[str, Any]:
    try:
        return {"task_id": recipes_mod.run(recipe_id, {}, provider, model).id}
    except KeyError:
        raise ValueError(f"unknown recipe {recipe_id}")


# --- waiting (CLI) ------------------------------------------------------------------------------------

def wait_task(task_id: str, on_log: Callable[[str], None] | None = None, timeout: float = 7200) -> dict[str, Any]:
    t = tasks.get(task_id)
    if t is None:
        raise ValueError(f"unknown task {task_id}")
    seen, t0 = 0, time.time()
    while t.status == "running":
        while on_log and seen < len(t.log):
            on_log(t.log[seen]); seen += 1
        if time.time() - t0 > timeout:
            raise ValueError("timed out waiting for the task")
        time.sleep(0.3)
    while on_log and seen < len(t.log):
        on_log(t.log[seen]); seen += 1
    return task(task_id)


def wait_job(job_id: str, on_log: Callable[[str], None] | None = None, timeout: float = 86400) -> dict[str, Any]:
    off, t0 = 0, time.time()
    while True:
        text, off = runner.read_log(job_id, off)
        if text and on_log:
            for ln in text.splitlines():
                if ln.strip() and not ln.startswith("@@ "):
                    on_log(ln)
        doc = runner.refresh(job_id)
        if doc and doc["status"] not in ("queued", "running"):
            return job(job_id)
        if time.time() - t0 > timeout:
            raise ValueError("timed out waiting for the job")
        time.sleep(0.5)
