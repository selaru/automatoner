"""Exporters for Transformers and MLX outputs: text classifier Lambda, HF folder, GGUF, Apple Core AI."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ... import config
from ...util import slug
from .. import templates as T
from ..base import ExportContext, Exporter, ExportError, copy, describe_files, sha256, write, zip_dir

HF_TRAINERS = ("hf_text", "mlx_lm")


def _run(ctx: ExportContext, cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 7200) -> str:
    ctx.log("$ " + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, **(env or {})})
    assert p.stdout
    out = []
    for line in p.stdout:
        out.append(line)
        ctx.log(line.rstrip())
    code = p.wait(timeout=timeout)
    if code != 0:
        raise ExportError(f"command failed with code {code}: {' '.join(cmd[:4])}...\n" + "".join(out[-12:]))
    return "".join(out)


class LambdaText(Exporter):
    id = "lambda_text"
    name = "AWS Lambda (text classifier)"
    blurb = "Container image with onnxruntime and tokenizers, plus the handler. Text models need the tokenizer, so a container is the easy route."
    needs_artifacts = ("onnx", "tokenizer", "inference")
    categories = ("text_classification",)

    def export(self, ctx: ExportContext) -> dict[str, Any]:
        pkg = ctx.out_dir / "package"
        copy(ctx.artifact_path("onnx"), pkg / "model.onnx")
        copy(ctx.artifact_path("tokenizer"), pkg / "tokenizer.json")
        inf = json.loads(ctx.artifact_path("inference").read_text(encoding="utf-8"))
        tokcfg = {}
        cfg_file = ctx.model_dir / "hf" / "tokenizer_config.json"
        if cfg_file.exists():
            tc = json.loads(cfg_file.read_text(encoding="utf-8"))
            tokcfg = {"pad_token": tc.get("pad_token") if isinstance(tc.get("pad_token"), str) else "[PAD]"}
        tj = json.loads((pkg / "tokenizer.json").read_text(encoding="utf-8"))
        pad = tokcfg.get("pad_token", "[PAD]")
        vocab = (tj.get("model") or {}).get("vocab")
        inf["pad_token"] = pad
        inf["pad_id"] = vocab.get(pad, 0) if isinstance(vocab, dict) else 0
        write(pkg / "inference.json", json.dumps(inf, indent=2))
        write(pkg / "handler.py", T.TEXT_HANDLER)
        write(pkg / "requirements.txt", T.TEXT_REQUIREMENTS)
        write(pkg / "Dockerfile", T.TEXT_DOCKERFILE)
        write(pkg / "sample_event.json", json.dumps({"text": "Allow 0.0.0.0/0 on port 22 for the bastion security group"}))
        write(pkg / "test_local.py", T.TEST_LOCAL)
        write(pkg / "README.md", f"# {slug(ctx.job['dataset_name'])} text classifier\n\n```\ndocker buildx build --platform linux/arm64 -t classifier .\n```\n\nPush to ECR and create a Lambda function from the image (arm64, 512 MB is a good start).\n\nLocal test: `pip install -r requirements.txt && python test_local.py`\n")
        zip_dir(pkg, ctx.out_dir.parent / f"{ctx.out_dir.name}.zip")
        return {"kind": "folder+zip", "root": "package", "download": f"{ctx.out_dir.name}.zip", "files": describe_files(pkg),
                "smoke_test": "cd package && pip install -r requirements.txt && python test_local.py"}


class HFFolder(Exporter):
    id = "hf_folder"
    name = "Hugging Face folder"
    blurb = "Merged weights in safetensors with tokenizer and config. Loads with Transformers anywhere."
    needs_artifacts = ("hf_model",)

    def export(self, ctx: ExportContext) -> dict[str, Any]:
        pkg = ctx.out_dir / "model"
        copy(ctx.artifact_path("hf_model"), pkg)
        size = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file())
        return {"kind": "folder", "root": "model", "location": str(pkg), "files": describe_files(pkg), "bytes": size,
                "notes": f"Files are in {pkg}. Too large to offer as a download in the browser." if size > 400e6 else None,
                "download": None, "smoke_test": f"python -c \"from transformers import AutoTokenizer, AutoModelForCausalLM as M; p='{pkg}'; AutoTokenizer.from_pretrained(p); M.from_pretrained(p); print('ok')\""}


class GGUF(Exporter):
    id = "gguf"
    name = "GGUF (llama.cpp)"
    blurb = "Single-file model for llama.cpp, Ollama and LM Studio. F16 or Q8_0 straight from the converter."
    needs_artifacts = ("hf_model",)
    categories = ("text_to_json", "tool_calling", "extraction")
    tools = ("git",)
    options = ({"key": "outtype", "label": "Precision", "type": "select", "default": "q8_0", "options": ["q8_0", "f16", "bf16"]},)

    def export(self, ctx: ExportContext) -> dict[str, Any]:
        root = config.tools_dir() / "llama.cpp"
        script = root / "convert_hf_to_gguf.py"
        if not script.exists():
            raise ExportError("llama.cpp is not fetched yet. Open the Doctor page and press 'Fetch llama.cpp', or run: git clone --depth 1 https://github.com/ggml-org/llama.cpp " + str(root))
        outtype = ctx.options.get("outtype") or "q8_0"
        if outtype not in ("q8_0", "f16", "bf16"):
            raise ExportError(f"unsupported precision {outtype!r}")
        pkg = ctx.out_dir / "model"
        pkg.mkdir(parents=True)
        out_file = pkg / f"{slug(ctx.job['model_id'])}-{slug(ctx.job['dataset_name'])}-{outtype}.gguf"
        env = {"PYTHONPATH": str(root / "gguf-py") + os.pathsep + os.environ.get("PYTHONPATH", "")}
        try:
            _run(ctx, [sys.executable, str(script), str(ctx.artifact_path("hf_model")), "--outfile", str(out_file), "--outtype", outtype], cwd=root, env=env)
        except ExportError as e:
            if "No module named" in str(e):
                raise ExportError(str(e) + "\nInstall the converter's dependencies into this environment:  uv pip install sentencepiece protobuf") from e
            raise
        return {"kind": "file", "location": str(out_file), "download": None, "files": describe_files(pkg), "bytes": out_file.stat().st_size, "sha256": sha256(out_file),
                "notes": f"GGUF file: {out_file}",
                "smoke_test": f"llama-cli -m \"{out_file}\" -p \"hello\" -n 20   # or: ollama create mymodel -f Modelfile"}


class CoreAI(Exporter):
    id = "coreai"
    name = "Apple Core AI bundle"
    blurb = "iOS bundle through Apple's coreai-models recipe, for models Apple's recipe supports. Needs macOS 27 and Xcode 27."
    needs_artifacts = ("hf_model",)
    categories = ("text_to_json", "tool_calling", "extraction")
    host = ("darwin",)
    tools = ("git", "uv")

    def applicable(self, manifest: dict[str, Any], category: str) -> tuple[bool, str]:
        ok, why = super().applicable(manifest, category)
        if not ok:
            return ok, why
        from ...catalogue import service as catalogue
        entry = catalogue.get_entry(manifest.get("model_id", ""))
        if not entry or not entry.get("coreai_recipe"):
            return False, "Apple's Core AI recipes cover Qwen3 0.6B only for now"
        return True, ""

    def export(self, ctx: ExportContext) -> dict[str, Any]:
        from ...catalogue import service as catalogue
        entry = catalogue.get_entry(ctx.job["model_id"]) or {}
        recipe = entry.get("coreai_recipe")
        if int((platform.mac_ver()[0] or "0").split(".")[0] or 0) < 27:
            raise ExportError(f"Core AI export needs macOS 27 (this is {platform.mac_ver()[0]}).")
        uv, git = shutil.which("uv"), shutil.which("git")
        if not uv or not git:
            raise ExportError("uv and git are required: brew install uv git")
        repo = config.tools_dir() / "coreai-models"
        if not repo.exists():
            _run(ctx, [git, "clone", "--depth", "1", "https://github.com/apple/coreai-models", str(repo)])
        help_text = _run(ctx, [uv, "run", "coreai.llm.export", "--help"], cwd=repo)
        flag = next((f for f in ("--checkpoint", "--model-path", "--hf-model-id", "--model-id", "--weights", "--source") if f in help_text), None)
        if flag is None:
            raise ExportError("Could not find the option that takes a local checkpoint in the recipe's --help output:\n" + help_text[:1500])
        cmd = [uv, "run", "coreai.llm.export", recipe, "--platform", "iOS", flag, str(ctx.artifact_path("hf_model"))]
        _run(ctx, cmd, cwd=repo)
        exports = repo / "exports"
        bundles = sorted((p for p in exports.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True) if exports.exists() else []
        if not bundles:
            raise ExportError("The recipe finished but no bundle was found under " + str(exports))
        pkg = ctx.out_dir / "bundle"
        copy(bundles[0], pkg)
        return {"kind": "folder", "location": str(pkg), "download": None, "files": describe_files(pkg)[:50],
                "notes": (f"Bundle copied to {pkg}. Next: hf upload <you>/<repo> {pkg} ios, then point the app's model ID at that repo "
                          "(see the taskllm README, 'Ship it')."),
                "smoke_test": f"ls \"{pkg}\""}


EXPORTERS = [LambdaText(), HFFolder(), GGUF(), CoreAI()]
