"""Trainer plugin contract."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class TrainContext:
    job_id: str
    job_dir: Path
    category: str
    model: dict[str, Any]            # catalogue entry
    model_path: str | None           # local weights for hf models, None for builtin algorithms
    dataset: dict[str, Any]          # dataset meta
    dataset_dir: Path
    config: dict[str, Any]
    out_dir: Path                    # artifacts go here
    is_cancelled: Callable[[], bool] = lambda: False

    def emit(self, **metric: Any) -> None:
        """One machine-readable line in the log; the UI plots numeric series from these."""
        print("@@ " + json.dumps({"t": round(time.time(), 2), **metric}), flush=True)

    def log(self, msg: str) -> None:
        print(msg, flush=True)


def field_(key: str, label: str, type_: str, default: Any, *, help: str = "", min: Any = None, max: Any = None,
           options: list[Any] | None = None, advanced: bool = False) -> dict[str, Any]:
    f = {"key": key, "label": label, "type": type_, "default": default, "help": help, "advanced": advanced}
    if min is not None: f["min"] = min
    if max is not None: f["max"] = max
    if options is not None: f["options"] = options
    return f


def write_manifest(ctx: TrainContext, manifest: dict[str, Any]) -> None:
    manifest = {"job_id": ctx.job_id, "category": ctx.category, "model_id": ctx.model["id"], **manifest}
    (ctx.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
