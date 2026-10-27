"""Recipes: saved combinations of model, data template, training config and export target.

A recipe runs as a background pipeline: model download, dataset, training, export.
"""
from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Any

import yaml

from . import config, tasks
from .catalogue import service as catalogue
from .datasets import service as datasets
from .export import service as exports
from .store import db
from .training import runner

CANDIDATE_DIRS = [Path(__file__).resolve().parent / "recipes", Path(__file__).resolve().parent.parent / "recipes"]


def _dir() -> Path:
    for d in CANDIDATE_DIRS:
        if d.exists() and any(d.glob("*.yaml")):
            return d
    return CANDIDATE_DIRS[-1]


def list_recipes() -> list[dict[str, Any]]:
    out = []
    for f in sorted(_dir().glob("*.yaml")):
        r = yaml.safe_load(f.read_text(encoding="utf-8"))
        r["file"] = f.name
        out.append(r)
    return out


def get(recipe_id: str) -> dict[str, Any] | None:
    return next((r for r in list_recipes() if r["id"] == recipe_id), None)


def _subst(obj: Any, params: dict[str, Any]) -> Any:
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: str(params.get(m.group(1), m.group(0))), obj)
    if isinstance(obj, list):
        return [_subst(x, params) for x in obj]
    if isinstance(obj, dict):
        return {k: _subst(v, params) for k, v in obj.items()}
    return obj


def _load_files(spec: dict[str, Any]) -> dict[str, Any]:
    """Keys ending in _file are read from disk and stored under the key without the suffix."""
    out = {}
    for k, v in spec.items():
        if k.endswith("_file") and isinstance(v, str):
            path = Path(v).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"{k}: {path} not found")
            text = path.read_text(encoding="utf-8")
            base = k[: -len("_file")]
            out[base] = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else text.strip() if base == "system_prompt" else text
        else:
            out[k] = v
    return out


def _wait(task: tasks.Task, parent: tasks.Task, label: str) -> None:
    seen = 0
    while task.status == "running":
        if parent.cancelled:
            task.cancel_flag.set()
            raise RuntimeError("cancelled")
        parent.set(message=f"{label}: {task.message}" if task.message else label)
        while seen < len(task.log):
            parent.say(task.log[seen]); seen += 1
        time.sleep(0.4)
    if task.status != "done":
        raise RuntimeError(f"{label} failed: {task.error}")


def run(recipe_id: str, params: dict[str, Any] | None = None, provider: str | None = None, model: str | None = None) -> tasks.Task:
    recipe = get(recipe_id)
    if recipe is None:
        raise KeyError(recipe_id)
    params = {**{p["key"]: p.get("default") for p in recipe.get("params", [])}, **(params or {})}

    def pipeline(task: tasks.Task) -> dict[str, Any]:
        entry = catalogue.get_entry(recipe["model"])
        if entry is None:
            raise RuntimeError(f"model {recipe['model']} is not in the catalogue")
        result: dict[str, Any] = {"recipe": recipe_id}
        if not catalogue.is_ready(entry["id"]):
            task.say(f"step 1/4: downloading {entry['name']}")
            _wait(catalogue.start_download(entry["id"]), task, "download")
        else:
            task.say("step 1/4: model ready")
        task.set(0.1)

        ds = recipe["dataset"]
        spec = _load_files(_subst(copy.deepcopy(ds["spec"]), params))
        task.say(f"step 2/4: dataset ({ds.get('name', recipe['name'])})")
        t, ds_id = datasets.start_synthetic(ds.get("name", recipe["name"]), recipe["category"], spec, provider=provider, model=model,
                                            options=_subst(ds.get("options", {}), params))
        _wait(t, task, "dataset")
        meta = datasets.get(ds_id) or {}
        if meta.get("status") != "ready":
            raise RuntimeError(f"dataset is {meta.get('status')}: {(meta.get('report') or {}).get('errors') or meta.get('error')}")
        result["dataset_id"] = ds_id
        task.set(0.3)

        task.say("step 3/4: training")
        job = runner.create(entry["id"], ds_id, recipe.get("trainer"), recipe["category"], _subst(recipe.get("train", {}), params))
        result["job_id"] = job["id"]
        while (db.get("jobs", job["id"]) or {}).get("status") in ("queued", "running"):
            if task.cancelled:
                runner.cancel(job["id"])
                raise RuntimeError("cancelled")
            task.set(0.3 + 0.5 * 0.5, "training")
            time.sleep(0.7)
        job = db.get("jobs", job["id"]) or {}
        if job.get("status") != "done":
            raise RuntimeError(f"training {job.get('status')}: {job.get('error')}")
        task.set(0.85)

        ex = recipe.get("export")
        if ex:
            task.say(f"step 4/4: export ({ex['exporter']})")
            et, export_id = exports.start(job["id"], ex["exporter"], _subst(ex.get("options", {}), params))
            _wait(et, task, "export")
            result["export_id"] = export_id
        return result

    return tasks.start("recipe", f"Recipe: {recipe['name']}", pipeline, id=f"rcp-{recipe_id}-{int(time.time())}")
