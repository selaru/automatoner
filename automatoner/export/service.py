"""Export orchestration: which exporters fit a job, run one, record the result."""
from __future__ import annotations

import json
import shutil
from typing import Any

from .. import audit, config, tasks, util
from ..store import db
from ..training import runner
from .base import ExportContext, ExportError, Exporter

_registry: dict[str, Exporter] = {}


def registry() -> dict[str, Exporter]:
    if not _registry:
        from .plugins import all_exporters
        for e in all_exporters():
            _registry[e.id] = e
    return _registry


def options_for(job_id: str) -> list[dict[str, Any]]:
    job = db.get("jobs", job_id)
    if job is None:
        raise KeyError(job_id)
    manifest = runner.manifest(job_id)
    out = []
    for e in registry().values():
        if manifest is None:
            ok, why = False, "the job has no trained model yet"
        else:
            ok, why = e.applicable(manifest, job["category"])
        host_ok, host_why = (e.host_ok() if ok else (True, ""))
        out.append({"id": e.id, "name": e.name, "blurb": e.blurb, "available": ok and host_ok, "reason": why or host_why,
                    "options": list(e.options), "tools": list(e.tools)})
    out.sort(key=lambda o: not o["available"])  # stable: registry order is the preferred order
    return out


def start(job_id: str, exporter_id: str, options: dict[str, Any] | None) -> tuple[tasks.Task, str]:
    job = db.get("jobs", job_id)
    if job is None or job["status"] != "done":
        raise ExportError("export needs a finished training job")
    exporter = registry().get(exporter_id)
    if exporter is None:
        raise ExportError(f"unknown exporter {exporter_id}")
    manifest = runner.manifest(job_id) or {}
    ok, why = exporter.applicable(manifest, job["category"])
    if not ok:
        raise ExportError(f"{exporter.name}: {why}")
    ok, why = exporter.host_ok()
    if not ok:
        raise ExportError(f"{exporter.name}: {why}")
    export_id = util.new_id("exp-")
    audit.record("export.start", export=export_id, job=job_id, exporter=exporter_id)
    out_dir = config.exports_dir() / export_id
    out_dir.mkdir(parents=True)
    doc = {"id": export_id, "job_id": job_id, "exporter": exporter_id, "exporter_name": exporter.name, "status": "running",
           "created": util.now_iso(), "options": options or {}, "model_name": job["model_name"], "dataset_name": job["dataset_name"]}
    db.put("exports", export_id, doc)

    def run(task: tasks.Task) -> dict[str, Any]:
        ctx = ExportContext(job=job, manifest=manifest, model_dir=runner.job_dir(job_id) / "model", out_dir=out_dir,
                            options=options or {}, log=task.say)
        try:
            result = exporter.export(ctx)
            (out_dir / "export.json").write_text(json.dumps({**doc, "result": result}, indent=2), encoding="utf-8")
            db.update("exports", export_id, status="done", result=result)
            return {"export_id": export_id, **result}
        except Exception as e:
            db.update("exports", export_id, status="error", error=str(e))
            raise

    return tasks.start("export", f"Export {exporter.name}", run, id=f"x-{export_id}"), export_id


def download_path(export_id: str):
    doc = db.get("exports", export_id)
    if not doc or doc.get("status") != "done":
        return None
    name = (doc.get("result") or {}).get("download")
    if not name:
        return None
    p = util.safe_child(config.exports_dir(), name)
    return p if p.is_file() else None


def delete(export_id: str) -> None:
    audit.record("export.delete", export=export_id)
    shutil.rmtree(util.safe_child(config.exports_dir(), export_id), ignore_errors=True)
    z = util.safe_child(config.exports_dir(), f"{export_id}.zip")
    if z.exists():
        z.unlink()
    db.delete("exports", export_id)
