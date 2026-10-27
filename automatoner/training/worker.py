"""Runs one training job in its own process:  python -m automatoner.training.worker <job_dir>

Writes status.json when done. Everything printed goes to the job log, which the UI streams.
"""
from __future__ import annotations

import json
import signal
import sys
import traceback
from pathlib import Path

from . import plugins
from .base import TrainContext


def _kill_children(*_: object) -> None:
    try:
        import psutil
        for c in psutil.Process().children(recursive=True):
            c.kill()
    except Exception:  # noqa: BLE001
        pass
    sys.exit(143)


def main(job_dir: str) -> int:
    jd = Path(job_dir)
    job = json.loads((jd / "job.json").read_text(encoding="utf-8"))
    for sig in (signal.SIGTERM, getattr(signal, "SIGBREAK", None)):
        if sig is not None:
            try:
                signal.signal(sig, _kill_children)
            except (ValueError, OSError):
                pass
    out = jd / "model"
    out.mkdir(parents=True, exist_ok=True)
    ctx = TrainContext(job_id=job["id"], job_dir=jd, category=job["category"], model=job["model"], model_path=job.get("model_path"),
                       dataset=job["dataset"], dataset_dir=Path(job["dataset_dir"]), config=job["config"], out_dir=out)
    status = {"status": "done", "error": None}
    try:
        plugin = plugins.load(job["trainer"])
        print(f"== {plugin.id} · {job['model']['name']} · {job['dataset']['name']} ({job['category']})", flush=True)
        plugin.run(ctx)
        if not (out / "manifest.json").exists():
            raise RuntimeError("trainer finished without writing manifest.json")
        print("== done", flush=True)
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        status = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        print(f"== failed: {status['error']}", flush=True)
    (jd / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return 0 if status["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
