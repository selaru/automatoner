"""Command line: everything the web UI does, for scripts and terminals.

  automatoner serve | doctor | models | download | datasets | train | jobs | export | recipes | mcp

Add --json to list and show commands for machine-readable output. Exit code 0 means success, 1 failure, 2 bad usage.
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
import webbrowser
from typing import Any

from . import audit, auth, ops


class CliError(Exception):
    pass


def out_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def table(rows: list[dict[str, Any]], cols: list[str], empty: str = "(none)") -> None:
    if not rows:
        print(empty)
        return
    cells = [[("" if r.get(c) is None else str(r.get(c))) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    widths = [min(w, 48) for w in widths]
    print("  ".join(c.upper().ljust(w) for c, w in zip(cols, widths)))
    for row in cells:
        print("  ".join(v[:w].ljust(w) for v, w in zip(row, widths)))


def log_line(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def parse_json_arg(text: str | None, what: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise CliError(f"{what} is not valid JSON: {e}")
    if not isinstance(value, dict):
        raise CliError(f"{what} must be a JSON object")
    return value


def read_json_file(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise CliError(f"cannot read {path}: {e}")


def metrics_lines(metrics: Any) -> list[str]:
    lines = []
    for split, vals in (metrics or {}).items():
        if isinstance(vals, dict):
            nums = {k: v for k, v in vals.items() if isinstance(v, (int, float)) and k not in ("n", "k")}
            lines.append(f"  {split}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in list(nums.items())[:8]))
    return lines


# --- commands ------------------------------------------------------------------------------------------

def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn
    from .app import create_app
    remote = a.host not in ("127.0.0.1", "localhost", "::1")
    app = create_app(allow_any_host=remote)  # refuses to start on a non-loopback address without an access token
    if auth.configured_token():
        print(f"Access token required (from {auth.source()}). The web interface asks for it once; scripts send 'Authorization: Bearer <token>'.")
    if remote:
        print("Listening beyond this computer. Put it behind TLS (VPN or reverse proxy): the token travels in clear text over plain HTTP.", file=sys.stderr)
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '::') else a.host}:{a.port}"
    if not a.no_browser:
        import threading
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Automatoner at {url}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


def cmd_token(a: argparse.Namespace) -> int:
    if a.action in ("create", "rotate"):
        try:
            t = auth.create_token(rotate=a.action == "rotate")
        except ValueError as e:
            raise CliError(str(e))
        audit.record(f"token.{a.action}")
        print(t)
        print(f"Saved to {auth.token_path()} (owner-only). Restart `automatoner serve` to use it. It is shown once here; to see it again, read that file.", file=sys.stderr)
    elif a.action == "remove":
        removed = auth.remove_token()
        audit.record("token.remove")
        print("token removed" if removed else "no token file")
    else:
        src = auth.source()
        print(f"access token: {'set, from ' + src if src else 'not set'}")
    return 0


def cmd_audit(a: argparse.Namespace) -> int:
    rows = audit.read(a.limit, a.action)
    if a.json:
        out_json(rows)
        return 0
    if not rows:
        print(f"no entries yet ({audit.path()})")
    for e in reversed(rows):
        extra = " ".join(f"{k}={v}" for k, v in e.items() if k not in ("time", "actor", "action"))
        print(f"{e['time']}  {e['actor']:<18} {e['action']:<18} {extra}")
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    if a.install:
        from .doctor import installers
        try:
            t = installers.install_extra(a.install)
        except (KeyError, ValueError) as e:
            raise CliError(str(e))
        res = ops.wait_task(t.id, on_log=lambda l: print(l.rstrip()))
        return 0 if res["status"] == "done" else 1
    d = ops.doctor()
    if a.json:
        out_json(d)
        return 1 if d["summary"]["fail"] else 0
    icons = {"ok": "\033[32m✓\033[0m", "warn": "\033[33m!\033[0m", "fail": "\033[31m✗\033[0m", "missing": "\033[33m+\033[0m", "na": "\033[90m-\033[0m"}
    if not sys.stdout.isatty() or sys.platform == "win32":
        icons = {"ok": "ok  ", "warn": "warn", "fail": "FAIL", "missing": "miss", "na": "n/a "}
    group = None
    for c in d["checks"]:
        if c["group"] != group:
            group = c["group"]
            print(f"\n{group.upper()}")
        print(f"  {icons[c['status']]} {c['name']}: {c['detail']}")
        f = c["fix"]
        if f and c["status"] in ("missing", "warn") and f.get("type") in ("extra", "command"):
            print("      fix: " + (f"automatoner doctor --install {f['extra']}" if f["type"] == "extra" else f["command"]))
    s = d["summary"]
    print(f"\n{s['ok']} ok, {s['warn']} warnings, {s['missing']} optional parts not installed, {s['fail']} problems")
    return 1 if s["fail"] else 0


def cmd_categories(a: argparse.Namespace) -> int:
    rows = ops.categories()
    if a.json:
        out_json(rows)
    else:
        table(rows, ["id", "models", "trainable_models", "what_it_does"])
    return 0


def cmd_models(a: argparse.Namespace) -> int:
    rows = ops.models(a.category)
    if a.json:
        out_json(rows)
        return 0
    for r in rows:
        r["train_with"] = "; ".join(f"{c}: {'/'.join(t) or 'planned'}" for c, t in r["trainers"].items())
    table(rows, ["id", "params", "size_mb", "status", "train_with"])
    return 0


def cmd_download(a: argparse.Namespace) -> int:
    t = ops.start_download(a.model)
    res = ops.wait_task(t["task_id"], on_log=log_line)
    print(res["status"] if res["status"] == "done" else f"failed: {res['error']}")
    return 0 if res["status"] == "done" else 1


def cmd_datasets(a: argparse.Namespace) -> int:
    from .datasets import service as ds
    if a.action == "list":
        rows = ops.datasets_list(a.category)
        out_json(rows) if a.json else table(rows, ["id", "name", "category", "status", "rows"])
        return 0
    if a.action == "show":
        d = ops.dataset(a.target)
        if a.json:
            out_json(d)
        else:
            print(f"{d['name']} ({d['category']}) {d['status']}, {d['rows']} rows, splits {d['splits']}")
            for m in d["errors"] + d["warnings"]:
                print("  - " + m)
        return 0
    if a.action == "delete":
        ds.delete(a.target)
        print("deleted")
        return 0
    if a.action == "add":
        if not a.category:
            raise CliError("--category is required")
        d = ops.add_dataset_file(a.target, a.name or a.target, a.category, a.label_column)
        out_json(d) if a.json else print(f"{d['id']}: {d['status']}, {d['rows']} rows, splits {d['splits']}" + "".join(f"\n  - {e}" for e in d["errors"]))
        return 0 if d["status"] == "ready" else 1
    if a.action == "generate":
        if not a.category:
            raise CliError("--category is required")
        if a.spec:
            spec = read_json_file(a.spec)
        elif a.describe:
            spec = {"describe": a.describe, "rows": a.rows}
        else:
            raise CliError("give --spec FILE (see the Datasets page for the format) or, for tabular data, --describe TEXT")
        g = ops.start_generation(a.name or "Generated dataset", a.category, spec, a.provider, a.model)
        res = ops.wait_task(g["task_id"], on_log=log_line)
        d = ops.dataset(g["dataset_id"], 0)
        out_json(d) if a.json else print(f"{d['id']}: {d['status']}, {d['rows']} rows" + (f"\n  {res['error']}" if res.get("error") else ""))
        return 0 if d["status"] == "ready" else 1
    return 2


def cmd_train(a: argparse.Namespace) -> int:
    cfg = parse_json_arg(a.config, "--config")
    j = ops.start_training(a.model, a.dataset, a.trainer, cfg)
    print(f"job {j['id']}", file=sys.stderr)
    if a.no_wait:
        out_json(j) if a.json else print(j["id"])
        return 0
    final = ops.wait_job(j["id"], on_log=log_line)
    out_json(final) if a.json else (print(f"{final['id']}: {final['status']}" + (f" ({final['error']})" if final.get("error") else "")), *map(print, metrics_lines(final.get("metrics"))))
    return 0 if final["status"] == "done" else 1


def cmd_jobs(a: argparse.Namespace) -> int:
    if a.action == "list":
        rows = ops.jobs(a.limit)
        out_json(rows) if a.json else table(rows, ["id", "status", "model", "category", "dataset_name"])
        return 0
    if not a.target:
        raise CliError(f"jobs {a.action} needs a job id")
    if a.action == "show":
        j = ops.job(a.target, log_lines=a.log_lines)
        if a.json:
            out_json(j)
        else:
            print(f"{j['id']}: {j['status']}  {j['model']} on {j['dataset_name']}" + (f"\n  error: {j['error']}" if j.get("error") else ""))
            print("\n".join(metrics_lines(j.get("metrics"))))
            if j.get("log_tail"):
                print("\n".join(j["log_tail"]))
        return 0
    if a.action == "log":
        if a.follow:
            ops.wait_job(a.target, on_log=print)
        else:
            print("\n".join(ops.log_tail(a.target, a.lines, 2000)))
        return 0
    if a.action == "cancel":
        r = ops.cancel_job(a.target)
        print("cancelled" if r["cancelled"] else "nothing to cancel (not running)")
        return 0 if r["cancelled"] else 1
    return 2


def cmd_export(a: argparse.Namespace) -> int:
    if a.list or not a.exporter:
        rows = ops.export_options(a.job)
        if a.json:
            out_json(rows)
        else:
            for r in rows:
                print(f"{'*' if r['available'] else ' '} {r['id']:<18} {r['what_it_makes'] if r['available'] else r['reason']}")
            print("\n* = available. Run: automatoner export --job JOB --exporter ID [--options JSON]")
        return 0
    e = ops.start_export(a.job, a.exporter, parse_json_arg(a.options, "--options"))
    res = ops.wait_task(e["task_id"], on_log=log_line)
    if res["status"] != "done":
        print(f"failed: {res['error']}", file=sys.stderr)
        return 1
    r = ops.export_result(e["export_id"])
    out_json(r) if a.json else print(f"exported to {r['location']}" + (f"\nzip: {r['download_zip']}" if r["download_zip"] else "") + (f"\ntry: {r['smoke_test']}" if r["smoke_test"] else ""))
    return 0


def cmd_recipes(a: argparse.Namespace) -> int:
    if a.action == "list":
        rows = ops.recipes_list()
        out_json(rows) if a.json else table(rows, ["id", "category", "model", "needs_llm", "export"])
        return 0
    if not a.target:
        raise CliError("recipes run needs a recipe id")
    r = ops.start_recipe(a.target, a.provider, a.model)
    res = ops.wait_task(r["task_id"], on_log=log_line)
    out_json(res) if a.json else print(res["status"] if res["status"] == "done" else f"failed: {res['error']}")
    if res["status"] == "done" and not a.json:
        print(json.dumps(res["result"]))
    return 0 if res["status"] == "done" else 1


def cmd_mcp(a: argparse.Namespace) -> int:
    try:
        from .mcp_server import Policy, build_server
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] == "mcp":
            raise CliError("the MCP SDK is not installed. Run: automatoner doctor --install mcp")
        raise
    policy = Policy(read_only=a.read_only, allow_llm=a.allow_llm, allow_dirs=[__import__("pathlib").Path(d) for d in a.allow_dir],
                    max_jobs=a.max_jobs, max_running=a.max_running, max_examples=a.max_examples, max_download_mb=a.max_download_mb)
    audit.set_actor(f"mcp:{getpass.getuser()}")
    audit.record("mcp.start", read_only=a.read_only, allow_llm=a.allow_llm, allow_dirs=a.allow_dir)
    build_server(policy).run("stdio")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="automatoner", description="Train tiny task-specific models. Run with no command to open the web UI.")
    sub = p.add_subparsers(dest="cmd")

    def add(name: str, help: str, fn, json_flag: bool = True) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help)
        sp.set_defaults(fn=fn)
        if json_flag:
            sp.add_argument("--json", action="store_true", help="machine-readable output")
        return sp

    s = add("serve", "start the web interface", cmd_serve, False)
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8765); s.add_argument("--no-browser", action="store_true")
    d = add("doctor", "check that everything needed is installed", cmd_doctor)
    d.add_argument("--install", metavar="EXTRA", help="install a Python extra: tabular, text, mlx, speech, coreml, mcp")
    tk = add("token", "manage the access token for the web interface (needed to listen beyond this computer)", cmd_token, False)
    tk.add_argument("action", choices=["create", "rotate", "remove", "status"], nargs="?", default="status")
    au = add("audit", "show the audit log: who did what", cmd_audit)
    au.add_argument("--limit", type=int, default=50); au.add_argument("--action", help="only entries whose action starts with this, e.g. llm or job")
    add("categories", "what models can be trained for", cmd_categories)
    m = add("models", "list the model catalogue", cmd_models); m.add_argument("--category")
    dl = add("download", "download a model from Hugging Face", cmd_download, False); dl.add_argument("model")

    ds = add("datasets", "list, add, generate, show or delete datasets", cmd_datasets)
    ds.add_argument("action", choices=["list", "show", "add", "generate", "delete"])
    ds.add_argument("target", nargs="?", help="dataset id (show, delete) or file path (add)")
    ds.add_argument("--category"); ds.add_argument("--name"); ds.add_argument("--label-column")
    ds.add_argument("--spec", help="JSON file describing what to generate"); ds.add_argument("--describe", help="tabular: describe the system in words and let an LLM design the simulator")
    ds.add_argument("--rows", type=int, default=5000); ds.add_argument("--provider"); ds.add_argument("--model")

    t = add("train", "train a model", cmd_train)
    t.add_argument("--model", required=True); t.add_argument("--dataset", required=True); t.add_argument("--trainer")
    t.add_argument("--config", help="JSON of trainer settings"); t.add_argument("--no-wait", action="store_true", help="start the job and return its id")

    j = add("jobs", "list, show, follow or cancel training jobs", cmd_jobs)
    j.add_argument("action", choices=["list", "show", "log", "cancel"]); j.add_argument("target", nargs="?", help="job id")
    j.add_argument("--limit", type=int, default=20); j.add_argument("--log-lines", type=int, default=0); j.add_argument("--lines", type=int, default=60)
    j.add_argument("--follow", "-f", action="store_true")

    e = add("export", "export a finished job for where it will run", cmd_export)
    e.add_argument("--job", required=True); e.add_argument("--exporter"); e.add_argument("--options", help="JSON"); e.add_argument("--list", action="store_true", help="show the exports that fit this job")

    r = add("recipes", "list or run ready-made pipelines", cmd_recipes)
    r.add_argument("action", choices=["list", "run"]); r.add_argument("target", nargs="?", help="recipe id"); r.add_argument("--provider"); r.add_argument("--model")

    mc = add("mcp", "serve Automatoner to AI agents over MCP (stdio)", cmd_mcp, False)
    mc.add_argument("--read-only", action="store_true", help="expose only tools that read")
    mc.add_argument("--allow-llm", action="store_true", help="let the agent generate data with paid LLM providers")
    mc.add_argument("--allow-dir", action="append", default=[], help="folder the agent may import dataset files from (repeatable)")
    mc.add_argument("--max-jobs", type=int, default=5, help="training jobs the agent may start per session")
    mc.add_argument("--max-running", type=int, default=2, help="jobs that may run at the same time")
    mc.add_argument("--max-examples", type=int, default=5000, help="largest synthetic dataset the agent may request")
    mc.add_argument("--max-download-mb", type=int, default=5000, help="largest model the agent may download")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        args = parser.parse_args(["serve"])
    actor = audit.set_actor(f"cli:{getpass.getuser()}")
    if args.fn not in (cmd_audit, cmd_token, cmd_serve):
        audit.record("cli", command=" ".join(str(x) for x in (argv if argv is not None else sys.argv[1:])[:2]))
    try:
        return args.fn(args)
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
