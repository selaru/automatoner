"""Helpers for wrapping the Cactus `needle` command line."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .base import TrainContext

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
LOSS_PATTERNS = [
    # "epoch     1/2  loss 0.5782  val 0.5162"
    (re.compile(r"epoch\s+(\d+)(?:/\d+)?\s+(?:train[_ ]?)?loss\s*[=:]?\s*" + NUM + r"\s+val(?:id)?(?:[_ ]loss)?\s*[=:]?\s*" + NUM, re.I), "epoch_both"),
    # "step      3/8  loss 0.5997"
    (re.compile(r"(?:step|iter)\s+(\d+)(?:/\d+)?\s+loss\s*[=:]?\s*" + NUM, re.I), "step_train"),
]
HELD_OUT = re.compile(r"accuracy\s+(\d+)/(\d+)\s+held-out", re.I)


def needle_exe() -> list[str]:
    exe = shutil.which("needle")
    if exe:
        return [exe]
    for name in ("needle", "needle.exe"):
        p = Path(sys.executable).parent / name
        if p.exists():
            return [str(p)]
    return [sys.executable, "-m", "needle"]


def run_cli(ctx: TrainContext, cmd: list[str], *, env_extra: dict[str, str] | None = None) -> list[str]:
    """Run a command, echo clean output to the job log, emit loss numbers it prints, return all lines."""
    ctx.log("$ " + " ".join(cmd))
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "NEEDLE_TELEMETRY": "0", "DO_NOT_TRACK": "1", **(env_extra or {})}
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    assert p.stdout
    lines: list[str] = []
    step = 0
    last_step = 0
    for raw in p.stdout:
        for line in ANSI.sub("", raw).replace("\r", "\n").split("\n"):
            if not line.strip():
                continue
            lines.append(line)
            print(line, flush=True)
            for pat, kind in LOSS_PATTERNS:
                m = pat.search(line)
                if not m:
                    continue
                g = m.groups()
                step += 1
                if kind == "epoch_both":
                    ctx.emit(step=last_step or int(g[0]), epoch=int(g[0]), valid_loss=float(g[2]))
                elif kind == "step_train":
                    last_step = int(g[0])
                    ctx.emit(step=last_step, train_loss=float(g[1]))
                elif kind == "val":
                    ctx.emit(step=step, valid_loss=float(g[0]))
                else:
                    ctx.emit(step=step, train_loss=float(g[0]))
                break
        if ctx.is_cancelled():
            p.terminate()
    code = p.wait()
    if code != 0:
        raise RuntimeError(f"command exited with code {code}: {' '.join(cmd[:3])} ...")
    return lines


def snapshot_file(root: str | None, rel: str) -> str:
    if not root:
        raise ValueError("this model is not downloaded. Download it from the Models page first.")
    p = Path(root) / rel
    if not p.exists():
        raise FileNotFoundError(f"{p} not found in the downloaded model; remove it and download again")
    return str(p)
