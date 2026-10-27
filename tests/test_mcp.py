import asyncio
import json
import os
import time

import pytest

pytest.importorskip("mcp")

from mcp.client import Client  # noqa: E402

from automatoner import config  # noqa: E402
from automatoner.mcp_server import Policy, build_server  # noqa: E402
from conftest import RHYTHM_SPEC  # noqa: E402

WRITE_TOOLS = {"download_model", "add_dataset_from_file", "generate_dataset", "start_training", "cancel_job", "start_export", "run_recipe"}


class Agent:
    """A synchronous wrapper around a real in-memory MCP client, so tests see what an agent sees."""

    def __init__(self, policy=None):
        self.policy = policy or Policy()
        self.server = build_server(self.policy)  # one server per session, as with the real stdio server

    def call(self, name, args=None):
        async def go():
            async with Client(self.server) as c:
                return await c.call_tool(name, args or {})
        r = asyncio.run(go())
        text = "".join(getattr(x, "text", "") for x in r.content)
        return r.is_error, (r.structured_content if not r.is_error else text)

    def ok(self, name, args=None):
        err, out = self.call(name, args)
        assert not err, out
        return out.get("result", out) if isinstance(out, dict) and set(out) == {"result"} else out

    def fails(self, name, args=None):
        err, out = self.call(name, args)
        assert err, f"{name} should have been refused, got {out}"
        return out

    def poll(self, task_id, timeout=120):
        t0 = time.time()
        while True:
            t = self.ok("get_task", {"task_id": task_id})
            if t["status"] != "running":
                return t
            assert time.time() - t0 < timeout
            time.sleep(0.2)

    def wait_job(self, job_id, timeout=300):
        t0 = time.time()
        while True:
            j = self.ok("get_job", {"job_id": job_id})
            if j["status"] not in ("queued", "running"):
                return j
            assert time.time() - t0 < timeout
            time.sleep(0.3)


def tool_names(policy):
    async def go():
        async with Client(build_server(policy)) as c:
            return {t.name for t in (await c.list_tools()).tools}
    return asyncio.run(go())


def test_read_only_mode_exposes_no_write_tool():
    full, ro = tool_names(Policy()), tool_names(Policy(read_only=True))
    assert WRITE_TOOLS <= full and not (WRITE_TOOLS & ro)
    assert {"list_models", "get_job", "doctor", "get_limits"} <= ro


def test_agents_get_the_real_error_message():
    out = Agent().fails("get_job", {"job_id": "nope"})
    assert "unknown job nope" in out
    assert "unknown dataset" in Agent().fails("start_training", {"model_id": "isolation_forest", "dataset_id": "x"})


def test_files_can_only_be_read_from_allowed_folders(tmp_path):
    inbox = Policy().inbox()
    (inbox / "ok.csv").write_text("a,b\n1,2\n")
    secret = tmp_path / "secret.csv"
    secret.write_text("a,b\n1,2\n")
    agent = Agent()
    args = {"name": "x", "category": "tabular_anomaly"}
    assert "outside the folders" in agent.fails("add_dataset_from_file", {**args, "path": str(secret)})
    assert "outside the folders" in agent.fails("add_dataset_from_file", {**args, "path": "/etc/hosts"})
    assert "outside the folders" in agent.fails("add_dataset_from_file", {**args, "path": str(inbox / ".." / ".." / "etc" / "hosts")})
    assert "outside the folders" in agent.fails("add_dataset_from_file", {**args, "path": "../../../etc/hosts"})
    link = inbox / "link.csv"
    if not link.exists():
        os.symlink(secret, link)
    assert "outside the folders" in agent.fails("add_dataset_from_file", {**args, "path": "link.csv"})
    # a bare name means the inbox; the data is too small to be valid but the file was read
    assert agent.ok("add_dataset_from_file", {**args, "path": "ok.csv"})["status"] == "invalid"
    # a folder the user allowed on the command line
    assert Agent(Policy(allow_dirs=[tmp_path])).ok("add_dataset_from_file", {**args, "path": str(secret)})["rows"] == 1


def test_llm_generation_is_off_by_default_but_the_simulator_is_not():
    agent = Agent()
    spec = {"labels": [{"name": "a"}, {"name": "b"}], "per_label": 10}
    assert "turned off" in agent.fails("generate_dataset", {"name": "t", "category": "text_classification", "spec": spec, "provider": "ollama"})
    assert "turned off" in agent.fails("generate_dataset", {"name": "t", "category": "tabular_anomaly", "spec": {"describe": "a web shop"}})
    assert "turned off" in agent.fails("run_recipe", {"recipe_id": "drift-sniffer"})
    started = agent.ok("generate_dataset", {"name": "sim", "category": "tabular_anomaly", "spec": RHYTHM_SPEC})
    assert agent.poll(started["task_id"])["status"] == "done"


def test_size_limits():
    agent = Agent(Policy(max_examples=100, max_download_mb=10))
    assert "limit is 100" in agent.fails("generate_dataset", {"name": "big", "category": "tabular_anomaly", "spec": {**RHYTHM_SPEC, "rows": 500}})
    assert "limit is 100" in agent.fails("generate_dataset", {"name": "big", "category": "text_classification", "provider": "ollama",
                                                               "spec": {"labels": [{"name": "a"}, {"name": "b"}], "per_label": 60}})
    assert "limit" in agent.fails("download_model", {"model_id": "minilm-l6"})
    assert "built-in" in agent.fails("download_model", {"model_id": "isolation_forest"})
    assert "too large" in Agent(Policy()).fails("generate_dataset", {"name": "x", "category": "tabular_anomaly", "spec": {"columns": "x" * 300_000}})


def test_full_workflow_through_mcp_and_job_limit():
    agent = Agent(Policy(max_jobs=1))
    g = agent.ok("generate_dataset", {"name": "mcp demo", "category": "tabular_anomaly", "spec": RHYTHM_SPEC})
    assert agent.poll(g["task_id"])["status"] == "done"
    ds = agent.ok("get_dataset", {"dataset_id": g["dataset_id"], "preview_rows": 50})
    assert ds["status"] == "ready" and len(ds["preview"]["rows"]) <= 10

    # a refused start must not use up the budget
    agent.fails("start_training", {"model_id": "lightgbm", "dataset_id": g["dataset_id"]})
    job = agent.ok("start_training", {"model_id": "isolation_forest", "dataset_id": g["dataset_id"], "config": {"n_estimators": 30}})
    done = agent.wait_job(job["id"])
    assert done["status"] == "done" and done["metrics"]["test"]["auroc"] > 0.8 and "onnx" in done["artifacts"]
    assert len(agent.ok("get_job", {"job_id": job["id"], "log_lines": 5000})["log_tail"]) <= 60

    assert "limit reached" in agent.fails("start_training", {"model_id": "isolation_forest", "dataset_id": g["dataset_id"]})

    options = {o["id"]: o for o in agent.ok("list_exports", {"job_id": job["id"]})}
    assert options["lambda_zip"]["available"] and not options["gguf"]["available"] and options["gguf"]["reason"]
    x = agent.ok("start_export", {"job_id": job["id"], "exporter": "lambda_zip", "options": {"function_name": "mcp-fn"}})
    assert agent.poll(x["task_id"])["status"] == "done"
    res = agent.ok("get_export", {"export_id": x["export_id"]})
    assert res["status"] == "done" and "handler.py" in res["files"] and os.path.isdir(res["location"])
    assert agent.ok("get_limits")["training_jobs_started"] == 1


def test_a_running_job_blocks_the_next_one_when_max_running_is_reached():
    agent = Agent(Policy(max_jobs=10, max_running=1, max_examples=30000))
    g = agent.ok("generate_dataset", {"name": "busy", "category": "tabular_anomaly", "spec": {**RHYTHM_SPEC, "rows": 20000}})
    agent.poll(g["task_id"])
    first = agent.ok("start_training", {"model_id": "one_class_svm", "dataset_id": g["dataset_id"], "config": {"max_train_rows": 20000}})
    out = agent.fails("start_training", {"model_id": "isolation_forest", "dataset_id": g["dataset_id"]})
    assert "already running" in out
    agent.ok("cancel_job", {"job_id": first["id"]})


def test_no_tool_ever_returns_a_stored_secret():
    config.save_settings({"hf_token": "hf_TOPSECRET", "anthropic_api_key": "sk-ant-TOPSECRET", "openai_api_key": "sk-TOPSECRET"})
    agent = Agent()
    seen = []
    for name, args in (("doctor", {}), ("get_limits", {}), ("list_models", {}), ("list_categories", {}), ("list_recipes", {}), ("list_datasets", {}),
                       ("list_jobs", {}), ("get_task", {"task_id": "dl-minilm-l6"})):
        err, out = agent.call(name, args)
        seen.append(json.dumps(out, default=str))
    assert "TOPSECRET" not in "".join(seen)
    config.save_settings({"hf_token": "", "anthropic_api_key": "", "openai_api_key": ""})


def test_tool_descriptions_tell_the_agent_how_to_use_them():
    async def go():
        async with Client(build_server(Policy())) as c:
            return (await c.list_tools()).tools, c.instructions
    tools, instructions = asyncio.run(go())
    assert all(t.description and len(t.description) > 25 for t in tools)
    assert "get_task" in instructions and "inbox" in next(t.description for t in tools if t.name == "add_dataset_from_file")
    ro = {t.name: t.annotations.read_only_hint for t in tools}
    assert ro["list_models"] is True and ro["start_training"] is False


def test_real_stdio_server_as_an_agent_host_would_launch_it(home):
    """Start `automatoner mcp` as a subprocess and talk to it over stdio."""
    import sys
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "automatoner", "mcp", "--read-only"], env={**os.environ, "AUTOMATONER_HOME": home})

    async def go():
        async with Client(params) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            models = await c.call_tool("list_models", {"category": "tabular_anomaly"})
            limits = await c.call_tool("get_limits", {})
            return names, models, limits

    names, models, limits = asyncio.run(go())
    assert "list_models" in names and not (WRITE_TOOLS & names)
    assert {m["id"] for m in models.structured_content["result"]} == {"isolation_forest", "one_class_svm", "autoencoder"}
    assert limits.structured_content["read_only"] is True
