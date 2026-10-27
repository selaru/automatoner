"""HTTP API. Everything the web UI does goes through here."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import __version__, config, tasks
from ..catalogue import service as catalogue
from ..datasets import providers
from ..datasets import service as datasets
from ..datasets.synth import llm as synth_llm
from ..doctor import checks as doctor_checks
from ..doctor import installers
from ..export import service as exports
from .. import recipes as recipes_mod
from ..store import db
from ..training import plugins, runner

router = APIRouter(prefix="/api")


def _bad(e: Exception, code: int = 400) -> HTTPException:
    return HTTPException(code, str(e))


@router.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": __version__, "home": str(config.home())}


# --- catalogue ---------------------------------------------------------------------------------

@router.get("/categories")
def categories() -> list[dict[str, Any]]:
    out = []
    models = catalogue.list_models()
    for c in catalogue.categories():
        ms = [m for m in models if c["id"] in m["categories"]]
        out.append({**c, "models": len(ms), "trainable": sum(1 for m in ms if m["trainable"].get(c["id"]))})
    return out


@router.get("/models")
def models(category: str | None = None) -> list[dict[str, Any]]:
    return catalogue.list_models(category)


@router.post("/models/{model_id}/download")
def download(model_id: str) -> dict[str, Any]:
    try:
        return catalogue.start_download(model_id).public()
    except (KeyError, ValueError) as e:
        raise _bad(e, 404 if isinstance(e, KeyError) else 400)


@router.get("/models/{model_id}/download")
def download_status(model_id: str) -> dict[str, Any]:
    t = tasks.get(f"dl-{model_id}")
    return t.public() if t else {"status": "none"}


@router.delete("/models/{model_id}/download")
def cancel_download(model_id: str) -> dict[str, Any]:
    t = tasks.get(f"dl-{model_id}")
    if t:
        t.cancel_flag.set()
    return {"ok": True}


@router.delete("/models/{model_id}")
def remove_model(model_id: str) -> dict[str, Any]:
    try:
        catalogue.remove(model_id)
    except KeyError as e:
        raise _bad(e, 404)
    return {"ok": True}


# --- datasets ----------------------------------------------------------------------------------

@router.get("/datasets")
def list_datasets(category: str | None = None) -> list[dict[str, Any]]:
    return datasets.list_datasets(category)


@router.get("/datasets/{ds_id}")
def get_dataset(ds_id: str) -> dict[str, Any]:
    d = datasets.get(ds_id)
    if d is None:
        raise HTTPException(404, "unknown dataset")
    return d


@router.get("/datasets/{ds_id}/preview")
def preview_dataset(ds_id: str, limit: int = 20) -> dict[str, Any]:
    try:
        return datasets.preview(ds_id, min(limit, 200))
    except KeyError:
        raise HTTPException(404, "unknown dataset")


@router.delete("/datasets/{ds_id}")
def delete_dataset(ds_id: str) -> dict[str, Any]:
    datasets.delete(ds_id)
    return {"ok": True}


@router.post("/datasets/upload")
async def upload_dataset(name: str = Form(...), category: str = Form(...), options: str = Form("{}"), files: list[UploadFile] = File(...)) -> dict[str, Any]:
    try:
        blobs = [(f.filename or "data", await f.read()) for f in files]
        return await asyncio.to_thread(datasets.import_upload, name, category, blobs, json.loads(options or "{}"))
    except (ValueError, KeyError) as e:
        raise _bad(e)


class HFDataset(BaseModel):
    name: str
    category: str
    repo: str
    split: str = "train"
    subset: str | None = None
    max_rows: int = 50000
    options: dict[str, Any] = {}


@router.post("/datasets/hf")
async def hf_dataset(body: HFDataset) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(datasets.import_hf, body.name, body.category, body.repo, split=body.split, subset=body.subset,
                                       max_rows=body.max_rows, options=body.options)
    except Exception as e:  # noqa: BLE001
        raise _bad(e)


class Synthetic(BaseModel):
    name: str
    category: str
    spec: dict[str, Any]
    provider: str | None = None
    model: str | None = None
    options: dict[str, Any] = {}


@router.post("/datasets/synthetic")
def synthetic(body: Synthetic) -> dict[str, Any]:
    try:
        task, ds_id = datasets.start_synthetic(body.name, body.category, body.spec, provider=body.provider, model=body.model, options=body.options)
    except Exception as e:  # noqa: BLE001
        raise _bad(e)
    return {"task": task.public(), "dataset_id": ds_id}


@router.get("/providers")
def list_providers() -> dict[str, Any]:
    s = config.load_settings()
    return {"providers": [{"id": p, "model": s["provider_models"].get(p), "configured": p == "ollama" or bool(s.get(f"{p}_api_key"))} for p in providers.PROVIDERS],
            "default": s["default_provider"], "ollama_models": providers.ollama_models()}


@router.post("/providers/{name}/test")
def test_provider(name: str, model: str | None = None) -> dict[str, Any]:
    try:
        p = providers.get_provider(name, model)
        reply = providers.ask(p, "Reply with the single word: ok", "ping", max_tokens=20, temperature=0, retries=1)
        return {"ok": True, "model": p.model, "reply": reply.strip()[:80]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


# --- tasks (downloads, generation, installs, exports) ---------------------------------------------

@router.get("/tasks")
def list_tasks(kind: str | None = None) -> list[dict[str, Any]]:
    return [t.public(tail=20) for t in tasks.all_tasks(kind)]


@router.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    t = tasks.get(task_id)
    if t is None:
        raise HTTPException(404, "unknown task")
    return t.public()


@router.delete("/tasks/{task_id}")
def cancel_task(task_id: str) -> dict[str, Any]:
    t = tasks.get(task_id)
    if t is None:
        raise HTTPException(404, "unknown task")
    t.cancel_flag.set()
    return {"ok": True}


# --- training ----------------------------------------------------------------------------------

@router.get("/trainers")
def trainers() -> list[dict[str, Any]]:
    return [plugins.available(k) for k in plugins.INFO]


@router.get("/trainers/{trainer}/schema")
def trainer_schema(trainer: str, category: str, model_id: str) -> list[dict[str, Any]]:
    try:
        return runner.schema_for(trainer, category, model_id)
    except (KeyError, runner.JobError) as e:
        raise _bad(e, 404)


class JobIn(BaseModel):
    model_id: str
    dataset_id: str
    trainer: str | None = None
    category: str | None = None
    config: dict[str, Any] = {}


@router.post("/jobs")
def create_job(body: JobIn) -> dict[str, Any]:
    try:
        return runner.create(body.model_id, body.dataset_id, body.trainer, body.category, body.config)
    except runner.JobError as e:
        raise _bad(e)


@router.get("/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return db.list_("jobs")


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    j = db.get("jobs", job_id)
    if j is None:
        raise HTTPException(404, "unknown job")
    return {**j, "series": runner.series(job_id), "manifest": runner.manifest(job_id)}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    return {"cancelled": runner.cancel(job_id)}


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    runner.delete(job_id)
    return {"ok": True}


@router.get("/jobs/{job_id}/report")
def job_report(job_id: str) -> dict[str, Any]:
    f = runner.job_dir(job_id) / "model" / "report.md"
    return {"markdown": f.read_text(encoding="utf-8") if f.exists() else None}


@router.get("/jobs/{job_id}/log")
def job_log(job_id: str, offset: int = 0) -> dict[str, Any]:
    text, new = runner.read_log(job_id, offset)
    return {"text": text, "offset": new}


# --- exports -----------------------------------------------------------------------------------

@router.get("/jobs/{job_id}/exports")
def export_options(job_id: str) -> list[dict[str, Any]]:
    try:
        return exports.options_for(job_id)
    except KeyError:
        raise HTTPException(404, "unknown job")


class ExportIn(BaseModel):
    job_id: str
    exporter: str
    options: dict[str, Any] = {}


@router.post("/exports")
def create_export(body: ExportIn) -> dict[str, Any]:
    try:
        task, export_id = exports.start(body.job_id, body.exporter, body.options)
    except Exception as e:  # noqa: BLE001
        raise _bad(e)
    return {"task": task.public(), "export_id": export_id}


@router.get("/exports")
def list_exports() -> list[dict[str, Any]]:
    return db.list_("exports")


@router.get("/exports/{export_id}")
def get_export(export_id: str) -> dict[str, Any]:
    e = db.get("exports", export_id)
    if e is None:
        raise HTTPException(404, "unknown export")
    return e


@router.get("/exports/{export_id}/download")
def download_export(export_id: str):
    p = exports.download_path(export_id)
    if p is None:
        raise HTTPException(404, "nothing to download")
    return FileResponse(p, filename=p.name)


@router.delete("/exports/{export_id}")
def delete_export(export_id: str) -> dict[str, Any]:
    exports.delete(export_id)
    return {"ok": True}


# --- recipes -----------------------------------------------------------------------------------

@router.get("/recipes")
def list_recipes() -> list[dict[str, Any]]:
    out = []
    for r in recipes_mod.list_recipes():
        entry = catalogue.get_entry(r["model"])
        out.append({**{k: r.get(k) for k in ("id", "name", "blurb", "source", "category", "model", "trainer", "needs_llm", "params")},
                    "model_name": entry["name"] if entry else r["model"], "model_state": catalogue.entry_view(entry)["state"]["status"] if entry else "missing",
                    "export": (r.get("export") or {}).get("exporter")})
    return out


class RecipeRun(BaseModel):
    params: dict[str, Any] = {}
    provider: str | None = None
    model: str | None = None


@router.post("/recipes/{recipe_id}/run")
def run_recipe(recipe_id: str, body: RecipeRun) -> dict[str, Any]:
    try:
        return recipes_mod.run(recipe_id, body.params, body.provider, body.model).public()
    except KeyError:
        raise HTTPException(404, "unknown recipe")


# --- settings ----------------------------------------------------------------------------------

@router.get("/settings")
def get_settings() -> dict[str, Any]:
    return config.public_settings()


@router.put("/settings")
def put_settings(update: dict[str, Any]) -> dict[str, Any]:
    config.save_settings({k: v for k, v in update.items() if v is not None})
    return config.public_settings()


@router.post("/settings/hf-token/check")
def check_hf_token() -> dict[str, Any]:
    c = doctor_checks.hf_check()
    return {"ok": c["status"] == "ok", "detail": c["detail"]}


# --- doctor ------------------------------------------------------------------------------------

@router.get("/doctor")
async def doctor() -> dict[str, Any]:
    results = await asyncio.to_thread(doctor_checks.run_all)
    return {"checks": results, "summary": doctor_checks.summary(results)}


@router.post("/doctor/fix/extra/{extra}")
def fix_extra(extra: str) -> dict[str, Any]:
    try:
        return installers.install_extra(extra).public()
    except (KeyError, ValueError) as e:
        raise _bad(e)


@router.post("/doctor/fix/action/{action}")
def fix_action(action: str) -> dict[str, Any]:
    if action == "llamacpp":
        return installers.fetch_llamacpp().public()
    raise HTTPException(404, "unknown action")


# --- websockets --------------------------------------------------------------------------------

ws_router = APIRouter()


@ws_router.websocket("/ws/jobs/{job_id}")
async def ws_job(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    offset = 0
    try:
        while True:
            text, offset = await asyncio.to_thread(runner.read_log, job_id, offset)
            job = db.get("jobs", job_id)
            if text:
                await ws.send_json({"type": "log", "text": text})
            await ws.send_json({"type": "status", "status": job["status"] if job else "missing"})
            if job is None or (job["status"] not in ("queued", "running") and not text):
                break
            await asyncio.sleep(0.7)
    except WebSocketDisconnect:
        return
    await ws.close()


@ws_router.websocket("/ws/tasks/{task_id}")
async def ws_task(ws: WebSocket, task_id: str) -> None:
    await ws.accept()
    try:
        while True:
            t = tasks.get(task_id)
            if t is None:
                await ws.send_json({"type": "missing"})
                break
            await ws.send_json({"type": "task", "task": t.public(tail=100)})
            if t.status != "running":
                break
            await asyncio.sleep(0.6)
    except WebSocketDisconnect:
        return
    await ws.close()
