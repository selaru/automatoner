"""Exporter contract. An exporter turns a trained job's artifacts into something that runs somewhere."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FRIENDLY = {
    "onnx": "Needs an ONNX model.", "hf_model": "Needs full fine-tuned language-model weights.", "tokenizer": "Needs a tokenizer file.",
    "cact": "Needs a Cactus .cact model (train with Needle or Whistle).", "joblib": "Needs a scikit-learn bundle.",
    "features": "Needs a feature description.", "inference": "Needs an inference description.", "torchscript": "Needs a PyTorch model.",
}


class ExportError(RuntimeError):
    pass


@dataclass
class ExportContext:
    job: dict[str, Any]
    manifest: dict[str, Any]
    model_dir: Path              # jobs/<id>/model
    out_dir: Path                # exports/<id>
    options: dict[str, Any]
    log: Any = print

    @property
    def artifacts(self) -> dict[str, str]:
        return self.manifest.get("artifacts", {})

    def artifact_path(self, key: str) -> Path:
        rel = self.artifacts.get(key)
        if not rel:
            raise ExportError(f"this model has no '{key}' artifact")
        return self.model_dir / rel


class Exporter:
    id = ""
    name = ""
    blurb = ""
    needs_artifacts: tuple[str, ...] = ()       # all required in the manifest
    categories: tuple[str, ...] | None = None    # None = any
    trainers: tuple[str, ...] | None = None      # None = any
    host: tuple[str, ...] | None = None          # e.g. ("darwin",) ; None = any
    requires: tuple[str, ...] = ()               # python modules
    tools: tuple[str, ...] = ()                  # executables looked up on PATH (informational)
    options: tuple[dict[str, Any], ...] = ()     # option fields for the UI

    def applicable(self, manifest: dict[str, Any], category: str) -> tuple[bool, str]:
        arts = manifest.get("artifacts", {})
        for a in self.needs_artifacts:
            if a not in arts:
                return False, f"{FRIENDLY.get(a, a)} This model type does not produce it."
        if self.categories and category not in self.categories:
            return False, f"not available for {category}"
        if self.trainers and manifest.get("trainer") not in self.trainers:
            return False, f"not available for the {manifest.get('trainer')} trainer"
        return True, ""

    def host_ok(self) -> tuple[bool, str]:
        if self.host and not any(sys.platform.startswith(h) for h in self.host):
            return False, f"only runs on {', '.join(self.host)}"
        import importlib.util
        missing = [m for m in self.requires if importlib.util.find_spec(m) is None]
        if missing:
            return False, f"missing Python packages: {', '.join(missing)}"
        return True, ""

    def export(self, ctx: ExportContext) -> dict[str, Any]:
        raise NotImplementedError


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_dir(src: Path, dest: Path) -> Path:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(src).as_posix())
    return dest


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def describe_files(root: Path) -> list[dict[str, Any]]:
    return [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size} for p in sorted(root.rglob("*")) if p.is_file()]
