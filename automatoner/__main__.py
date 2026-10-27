"""Command line: automatoner serve | doctor | models | train | export"""
from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser


def _doctor(args: argparse.Namespace) -> int:
    from .doctor import checks
    results = checks.run_all()
    icons = {"ok": "\033[32m✓\033[0m", "warn": "\033[33m!\033[0m", "fail": "\033[31m✗\033[0m", "missing": "\033[33m+\033[0m", "na": "\033[90m-\033[0m"}
    if not sys.stdout.isatty() or sys.platform == "win32":
        icons = {"ok": "ok  ", "warn": "warn", "fail": "FAIL", "missing": "miss", "na": "n/a "}
    group = None
    for c in results:
        if c["group"] != group:
            group = c["group"]
            print(f"\n{group.upper()}")
        print(f"  {icons[c['status']]} {c['name']}: {c['detail']}")
        if c["fix"] and c["status"] in ("missing", "warn") and c["fix"].get("type") in ("extra", "command"):
            how = f"automatoner doctor --install {c['fix']['extra']}" if c["fix"]["type"] == "extra" else c["fix"]["command"]
            print(f"      fix: {how}")
    s = checks.summary(results)
    print(f"\n{s['ok']} ok, {s['warn']} warnings, {s['missing']} optional parts not installed, {s['fail']} problems")
    if args.install:
        from . import tasks
        from .doctor import installers
        t = installers.install_extra(args.install)
        shown = 0
        while t.status == "running" or shown < len(t.log):
            while shown < len(t.log):
                print(t.log[shown]); shown += 1
            time.sleep(0.2)
        return 0 if t.status == "done" else 1
    return 1 if s["fail"] else 0


def _models(args: argparse.Namespace) -> int:
    from .catalogue import service
    for cat in service.categories():
        if args.category and cat["id"] != args.category:
            continue
        print(f"\n{cat['name']}  ({cat['id']})")
        for m in service.list_models(cat["id"]):
            t = ", ".join(m["trainer_options"].get(cat["id"], [])) or "planned"
            st = m["state"]["status"]
            print(f"  {m['id']:<24} {m['params']:>10} {m['size_mb']:>6} MB  {st:<14} trainer: {t}")
    return 0


def _wait_job(job_id: str) -> dict:
    from .store import db
    from .training import runner
    off = 0
    while True:
        text, off = runner.read_log(job_id, off)
        if text:
            print(text, end="", flush=True)
        j = db.get("jobs", job_id)
        if j and j["status"] not in ("queued", "running"):
            text, off = runner.read_log(job_id, off)
            print(text, end="")
            return j
        time.sleep(0.5)


def _train(args: argparse.Namespace) -> int:
    from .training import runner
    cfg = json.loads(args.config) if args.config else {}
    job = runner.create(args.model, args.dataset, args.trainer, None, cfg)
    print(f"job {job['id']}")
    j = _wait_job(job["id"])
    print(f"status: {j['status']}" + (f" ({j['error']})" if j.get("error") else ""))
    return 0 if j["status"] == "done" else 1


def _export(args: argparse.Namespace) -> int:
    from .export import service
    t, eid = service.start(args.job, args.exporter, json.loads(args.options) if args.options else {})
    while t.status == "running":
        time.sleep(0.2)
    if t.status != "done":
        print(t.error); return 1
    from . import config
    print(f"exported to {config.exports_dir() / eid}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn
    from .app import create_app
    app = create_app(allow_any_host=args.host not in ("127.0.0.1", "localhost", "::1"))
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    if not args.no_browser:
        import threading
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Automatoner at {url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="automatoner")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="start the web interface")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8765); s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=_serve)
    d = sub.add_parser("doctor", help="check that everything needed is installed")
    d.add_argument("--install", metavar="EXTRA", help="install a Python extra: " + ", ".join(["tabular", "text", "mlx", "speech", "coreml"]))
    d.set_defaults(fn=_doctor)
    m = sub.add_parser("models", help="list the model catalogue"); m.add_argument("--category"); m.set_defaults(fn=_models)
    t = sub.add_parser("train", help="train from the command line")
    t.add_argument("--model", required=True); t.add_argument("--dataset", required=True); t.add_argument("--trainer"); t.add_argument("--config", help="JSON")
    t.set_defaults(fn=_train)
    e = sub.add_parser("export", help="export a finished job")
    e.add_argument("--job", required=True); e.add_argument("--exporter", required=True); e.add_argument("--options", help="JSON")
    e.set_defaults(fn=_export)
    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        args = p.parse_args(["serve"])
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
