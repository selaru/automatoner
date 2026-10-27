"""MCP server: lets an AI agent drive Automatoner (pick a model, make data, train, export).

Run with `automatoner mcp` (stdio). Safe by default:
- Read tools are always available. Write tools can be switched off with --read-only.
- Datasets can only be imported from the inbox folder and folders passed with --allow-dir.
- Generating data with a paid LLM provider is off unless --allow-llm is given.
- Limits on jobs started, jobs running at once, synthetic dataset size and model download size.
- Stored API keys and tokens are never returned by any tool.
Long operations (downloads, data generation, exports) return a task id; poll get_task. Training returns a job id; poll get_job.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import functools

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import __version__, config, ops
from .catalogue import service as catalogue
from .store import db


@dataclass
class Policy:
    read_only: bool = False
    allow_llm: bool = False
    allow_dirs: list[Path] = field(default_factory=list)
    max_jobs: int = 5
    max_running: int = 2
    max_examples: int = 5000
    max_download_mb: int = 5000
    max_spec_bytes: int = 200_000

    def inbox(self) -> Path:
        p = config.home() / "inbox"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def roots(self) -> list[Path]:
        return [self.inbox().resolve(), *(Path(d).expanduser().resolve() for d in self.allow_dirs)]


class Session:
    def __init__(self) -> None:
        self.jobs_started = 0
        self.lock = threading.Lock()


def check_path(policy: Policy, path: str) -> Path:
    """Resolve symlinks and `..`, then require the file to sit inside an allowed folder."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = policy.inbox() / p
    p = p.resolve()
    if not any(p == r or r in p.parents for r in policy.roots()):
        raise ValueError(f"{path} is outside the folders this server may read. Put the file in {policy.inbox()} "
                         "or ask the user to start the server with --allow-dir <folder>.")
    if not p.is_file():
        raise ValueError(f"{p} is not a file")
    return p


RO = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
NET = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)

INSTRUCTIONS = """Automatoner trains tiny single-purpose models. Typical flow:
1. doctor, then list_categories and list_models to see what can be trained and what is installed.
2. Get a dataset: add_dataset_from_file (file in the inbox) or generate_dataset. Check it with get_dataset (status must be 'ready').
3. download_model if the model's status is 'not_downloaded', then start_training. Poll get_job until status is done or error.
4. list_exports for the job, then start_export, and read the result with get_task.
Downloads, data generation and exports return a task_id: poll get_task until status is not 'running'.
Never guess ids; copy them from earlier results. Training can take minutes."""


def build_server(policy: Policy | None = None) -> MCPServer:
    policy = policy or Policy()
    session = Session()
    server = MCPServer("automatoner", instructions=INSTRUCTIONS, version=__version__)
    raw_tool = server.tool

    def tool(**kw: Any):
        """Register a tool. Expected problems (bad id, limit reached, file not allowed) become ToolError so the agent reads the message;
        anything else stays an internal error and is not described to the agent."""
        register = raw_tool(**kw)

        def wrap(fn):
            @functools.wraps(fn)
            def inner(*a: Any, **k: Any):
                try:
                    return fn(*a, **k)
                except ValueError as e:
                    raise ToolError(str(e)) from None
            return register(inner)
        return wrap

    # ---- read tools ---------------------------------------------------------------------------------

    @tool(annotations=RO)
    def get_limits() -> dict[str, Any]:
        """What this server allows: read-only mode, folders files can be imported from, LLM use, limits and how many jobs were started."""
        return {"read_only": policy.read_only, "llm_data_generation_allowed": policy.allow_llm, "import_folders": [str(r) for r in policy.roots()],
                "max_training_jobs_per_session": policy.max_jobs, "training_jobs_started": session.jobs_started, "max_jobs_running_at_once": policy.max_running,
                "max_synthetic_examples": policy.max_examples, "max_model_download_mb": policy.max_download_mb}

    @tool(annotations=RO)
    def doctor() -> dict[str, Any]:
        """Check the environment: which Python extras, tools and accelerators are installed. Only items that are not ok are listed."""
        d = ops.doctor()
        return {"summary": d["summary"], "needs_attention": [{k: c[k] for k in ("id", "name", "status", "detail")} for c in d["checks"] if c["status"] != "ok"]}

    @tool(annotations=RO)
    def list_categories() -> list[dict[str, Any]]:
        """The kinds of task a model can be trained for, with the dataset format each expects."""
        return ops.categories()

    @tool(annotations=RO)
    def list_models(category: str | None = None) -> list[dict[str, Any]]:
        """Base models, optionally for one category id. 'status' is builtin, downloaded or not_downloaded. 'trainers' lists what can train it ('planned' means not yet)."""
        return ops.models(category)

    @tool(annotations=RO)
    def list_datasets(category: str | None = None) -> list[dict[str, Any]]:
        """Datasets that exist, with status (ready, generating, invalid, error), row counts and validation problems."""
        return ops.datasets_list(category)

    @tool(annotations=RO)
    def get_dataset(dataset_id: str, preview_rows: int = 5) -> dict[str, Any]:
        """One dataset: status, splits, validation errors and a few preview rows (at most 10)."""
        return ops.dataset(dataset_id, min(max(preview_rows, 0), 10))

    @tool(annotations=RO)
    def list_recipes() -> list[dict[str, Any]]:
        """Ready-made pipelines (data, model, training, export) for common small jobs."""
        return ops.recipes_list()

    @tool(annotations=RO)
    def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
        """Training jobs, newest first, with status and metrics."""
        return ops.jobs(min(max(limit, 1), 50))

    @tool(annotations=RO)
    def get_job(job_id: str, log_lines: int = 0) -> dict[str, Any]:
        """One job: status (queued, running, done, error, cancelled), test metrics, artifacts and optionally the last log lines (at most 60)."""
        return ops.job(job_id, min(max(log_lines, 0), 60))

    @tool(annotations=RO)
    def list_exports(job_id: str) -> list[dict[str, Any]]:
        """The ways a finished job can be exported, with 'available' and the reason when not, plus each option's default."""
        return ops.export_options(job_id)

    @tool(annotations=RO)
    def get_task(task_id: str) -> dict[str, Any]:
        """Progress of a background task (download, data generation, export, recipe). When done, 'result' holds ids such as export_id or dataset_id."""
        return ops.task(task_id)

    @tool(annotations=RO)
    def get_export(export_id: str) -> dict[str, Any]:
        """Where a finished export is on disk, its files and a command to try it."""
        return ops.export_result(export_id)

    if policy.read_only:
        return server

    # ---- write tools --------------------------------------------------------------------------------

    def running_jobs() -> int:
        return sum(1 for j in db.list_("jobs") if j["status"] in ("queued", "running"))

    def reserve_job() -> None:
        with session.lock:
            if session.jobs_started >= policy.max_jobs:
                raise ValueError(f"limit reached: {policy.max_jobs} training jobs per session. Ask the user to restart the server with a higher --max-jobs.")
            if running_jobs() >= policy.max_running:
                raise ValueError(f"{policy.max_running} jobs are already running. Wait for one to finish (get_job) before starting another.")
            session.jobs_started += 1

    def release_job() -> None:
        with session.lock:
            session.jobs_started = max(0, session.jobs_started - 1)

    def need_llm(provider: str | None) -> None:
        if not policy.allow_llm:
            raise ValueError("generating data with an LLM provider is turned off. The user can enable it with --allow-llm. "
                             "Tabular data from a simulator spec works without it.")

    @tool(annotations=NET)
    def download_model(model_id: str) -> dict[str, Any]:
        """Download a model from Hugging Face into the local cache. Returns a task_id to poll with get_task. Refused if larger than the size limit."""
        entry = catalogue.get_entry(model_id)
        if entry is None:
            raise ValueError(f"unknown model {model_id}")
        if entry["source"] == "builtin":
            raise ValueError("this is a built-in algorithm; nothing to download")
        if entry["size_mb"] > policy.max_download_mb:
            raise ValueError(f"{entry['name']} is about {entry['size_mb']} MB, over the {policy.max_download_mb} MB limit")
        return ops.start_download(model_id)

    @tool(annotations=WRITE)
    def add_dataset_from_file(path: str, name: str, category: str, label_column: str | None = None) -> dict[str, Any]:
        """Import a CSV, TSV, Parquet, JSON or JSONL file (or a zip with audio and a JSONL) as a dataset. The path must be inside the inbox or an allowed folder; a bare file name means the inbox. Returns the dataset summary: check status and errors."""
        return ops.add_dataset_file(check_path(policy, path), name, category, label_column)

    @tool(annotations=NET)
    def generate_dataset(name: str, category: str, spec: dict[str, Any], provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        """Generate a synthetic dataset. For tabular categories, spec is a simulator description (columns with kind and anomaly settings, rows, anomaly_rate) and needs no LLM. Other categories need an LLM (provider ollama, anthropic, gemini, openai or openrouter) and spec such as {labels, per_label} for text_classification or {tools, total} for tool_calling. Returns task_id and dataset_id."""
        if len(json.dumps(spec)) > policy.max_spec_bytes:
            raise ValueError("spec is too large")
        n = ops.generation_size(category, spec)
        if n > policy.max_examples:
            raise ValueError(f"{n} examples requested; the limit is {policy.max_examples}")
        if ops.uses_llm(category, spec):
            need_llm(provider)
        return ops.start_generation(name, category, spec, provider, model)

    @tool(annotations=WRITE)
    def start_training(model_id: str, dataset_id: str, trainer: str | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Start training. The model must be downloaded (or built in) and the dataset ready and of a category the model supports. 'config' overrides trainer settings, for example {"epochs": 3}. Returns the job; poll with get_job."""
        reserve_job()
        try:
            return ops.start_training(model_id, dataset_id, trainer, config)
        except Exception:
            release_job()
            raise

    @tool(annotations=WRITE)
    def cancel_job(job_id: str) -> dict[str, Any]:
        """Stop a running training job."""
        return ops.cancel_job(job_id)

    @tool(annotations=WRITE)
    def start_export(job_id: str, exporter: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Export a finished job (ids from list_exports). Returns task_id and export_id; read the outcome with get_task, then get_export."""
        return ops.start_export(job_id, exporter, options)

    @tool(annotations=NET)
    def run_recipe(recipe_id: str, provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        """Run a ready-made pipeline end to end: model download, dataset, training, export. Counts as one training job. Recipes that need an LLM are refused unless the server allows it. Returns a task_id to poll."""
        recipe = next((r for r in ops.recipes_list() if r["id"] == recipe_id), None)
        if recipe is None:
            raise ValueError(f"unknown recipe {recipe_id}")
        if recipe["needs_llm"]:
            need_llm(provider)
        entry = catalogue.get_entry(recipe["model"])
        if entry and entry["source"] == "hf" and not catalogue.is_ready(entry["id"]) and entry["size_mb"] > policy.max_download_mb:
            raise ValueError(f"the recipe needs {entry['name']} (about {entry['size_mb']} MB), over the download limit")
        reserve_job()
        try:
            return ops.start_recipe(recipe_id, provider, model)
        except Exception:
            release_job()
            raise

    return server
