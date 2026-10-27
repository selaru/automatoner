"""Fixes the doctor can apply: install a Python extra, fetch llama.cpp."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from typing import Any

from .. import config, tasks
from .checks import EXTRAS, _host_matches


def installed_extras() -> list[str]:
    return [k for k, e in EXTRAS.items() if all(importlib.util.find_spec(m) for m in e["modules"])]


def _stream(cmd: list[str], task: tasks.Task, cwd: str | None = None) -> int:
    task.say("$ " + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdout
    for line in p.stdout:
        task.say(line)
        if task.cancelled:
            p.terminate()
            break
    return p.wait()


def install_extra(extra: str) -> tasks.Task:
    if extra not in EXTRAS:
        raise KeyError(extra)
    if not _host_matches(EXTRAS[extra].get("platforms")):
        raise ValueError(f"'{extra}' is not available on this computer")

    def run(task: tasks.Task) -> dict[str, Any]:
        wanted = sorted(set(installed_extras()) | {extra} | set(config.load_settings().get("enabled_extras", [])))
        wanted = [w for w in wanted if _host_matches(EXTRAS[w].get("platforms"))]
        project = config.PROJECT_DIR
        uv = shutil.which("uv")
        if uv and (project / "pyproject.toml").exists() and (project / "uv.lock").exists():
            cmd = [uv, "sync", "--inexact", *sum((["--extra", w] for w in wanted), [])]
            code = _stream(cmd, task, cwd=str(project))
        elif uv:
            code = _stream([uv, "pip", "install", "--python", sys.executable, f"automatoner[{','.join(wanted)}]"], task)
        else:
            code = _stream([sys.executable, "-m", "pip", "install", f"automatoner[{','.join(wanted)}]"], task)
        if code != 0:
            raise RuntimeError(f"install exited with code {code}")
        config.save_settings({"enabled_extras": wanted})
        importlib.invalidate_caches()
        task.say("done. Restart the server if the new packages are not picked up.")
        return {"extras": wanted}

    return tasks.start("install", f"Install '{extra}' extra", run, id=f"inst-{extra}")


def fetch_llamacpp() -> tasks.Task:
    def run(task: tasks.Task) -> dict[str, Any]:
        if not shutil.which("git"):
            raise RuntimeError("git is required")
        dest = config.tools_dir() / "llama.cpp"
        if dest.exists():
            task.say("already present; updating")
            code = _stream(["git", "-C", str(dest), "pull", "--ff-only"], task)
        else:
            code = _stream(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(dest)], task)
        if code != 0:
            raise RuntimeError(f"git exited with code {code}")
        req = dest / "requirements" / "requirements-convert_hf_to_gguf.txt"
        if req.exists():
            task.say("llama.cpp converter dependencies are installed on first export")
        return {"path": str(dest)}

    return tasks.start("install", "Fetch llama.cpp", run, id="inst-llamacpp")
