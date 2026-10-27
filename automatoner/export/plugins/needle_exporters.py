"""Exporters for Cactus .cact models (Needle 3 and Whistle): the archive with or without the engine for a platform."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ...training import cactus_common as cc
from ..base import ExportContext, Exporter, ExportError, copy, describe_files, sha256, write, zip_dir

PLATFORMS = ["macos-arm64", "linux-x86_64", "linux-arm64", "linux-armv7", "linux-riscv64", "windows-x86_64", "windows-arm64",
             "android-arm64", "android-armv7", "ios-arm64", "ios-sim-arm64", "wasm", "wasm-component"]

README = '''# {title}

Built by Automatoner from job `{job_id}`.

Files
- `{cact}` the model ({size}).
{engine_note}

Run it
{run}

More: https://cactuscompute.com/blog/needle-supported-devices
'''


class Cact(Exporter):
    id = "cact"
    name = "Cactus .cact + engine"
    blurb = "The model archive with the Cactus engine for the device you choose: phones, Linux boards, Windows, browsers."
    needs_artifacts = ("cact",)
    trainers = ("needle", "whistle")
    requires = ("needle",)
    options = ({"key": "platform", "label": "Target platform", "type": "select", "default": "macos-arm64", "options": ["(model file only)"] + PLATFORMS},)

    def export(self, ctx: ExportContext) -> dict[str, Any]:
        trainer = ctx.manifest["trainer"]
        platform = ctx.options.get("platform") or "macos-arm64"
        if not platform.startswith("(") and platform not in PLATFORMS:
            raise ExportError(f"unknown platform {platform!r}")
        pkg = ctx.out_dir / "package"
        pkg.mkdir(parents=True)
        cact = ctx.artifact_path("cact")
        is_whistle = trainer == "whistle"
        final_name = "whistle.cact" if is_whistle else "needle3.cact"
        if platform.startswith("("):
            copy(cact, pkg / final_name)
            engine = "No engine included. Fetch one with `needle download <platform>`."
        else:
            adapter = ctx.artifact_path("adapter")
            layers = str(ctx.manifest.get("layers", 20 if not is_whistle else 8))
            cmd = [*cc.needle_exe(), *(["whistle"] if is_whistle else []), "build"]
            base = ctx.manifest.get("base_checkpoint")
            if base and Path(base).exists():
                cmd.append(base)
            cmd += ["--lora", str(adapter), "--layers", layers, "--platform", platform, "--out", str(pkg)]
            ctx.log("$ " + " ".join(cmd))
            r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "NEEDLE_TELEMETRY": "0", "DO_NOT_TRACK": "1", "NO_COLOR": "1"})
            for line in (r.stdout + r.stderr).splitlines():
                ctx.log(line)
            if r.returncode != 0:
                raise ExportError(f"needle build failed ({r.returncode}):\n" + (r.stdout + r.stderr)[-1200:])
            engine = f"- the Cactus engine for {platform}: the `needle` runner, `libneedle.a` and `needle.h`."
        size = (pkg / final_name).stat().st_size if (pkg / final_name).exists() else cact.stat().st_size
        if is_whistle:
            run = f"```\n./needle --model {final_name} --audio clip.wav\n./needle --model {final_name} --audio talk.wav --audio-stream\n```"
        else:
            run = f"```\n./needle --model {final_name} --tools tools.json --prompt \"dim the living room to 30\"\n./needle --model {final_name} --tools tools.json --serve\n```"
        write(pkg / "README.md", README.format(title=f"{ctx.job['model_name']} on {ctx.job['dataset_name']}", job_id=ctx.job["id"], cact=final_name,
                                              size=f"{size / 1e6:.1f} MB", engine_note=engine, run=run))
        zip_dir(pkg, ctx.out_dir.parent / f"{ctx.out_dir.name}.zip")
        return {"kind": "folder+zip", "root": "package", "download": f"{ctx.out_dir.name}.zip", "files": describe_files(pkg),
                "location": str(pkg), "sha256": sha256(pkg / final_name) if (pkg / final_name).exists() else None,
                "smoke_test": f"cd package && ./needle --model {final_name} " + ("--audio clip.wav" if is_whistle else "--tools tools.json --prompt \"...\"") if not platform.startswith("(") else None}


EXPORTERS = [Cact()]
