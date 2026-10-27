"""Opt-in: real downloads and real training. Run with:  AUTOMATONER_SLOW=1 uv run pytest tests/test_slow_real_models.py -s

Needs the text, mlx and speech extras and network access. Uses tiny datasets written by a scripted LLM, so it checks that every stage
runs and produces artifacts, not that the models are good.
"""
import json
import os
import random
import shutil
import subprocess
import sys

import pytest

from automatoner import config, tasks
from automatoner.catalogue import service as catalogue
from automatoner.datasets import providers, service as ds
from automatoner.export import service as exports
from automatoner.training import plugins, runner
from conftest import wait_job, wait_task

slow = pytest.mark.skipif(not os.environ.get("AUTOMATONER_SLOW"), reason="set AUTOMATONER_SLOW=1 to run")


def download(model_id):
    catalogue.start_download(model_id)
    t = tasks.get(f"dl-{model_id}")
    wait_task(t, timeout=1800)
    assert t.status == "done", t.error


def need(trainer):
    if not plugins.available(trainer)["ready"]:
        pytest.skip(f"{trainer} is not installed")


@slow
def test_text_classifier_to_lambda():
    need("hf_text")
    words = {"widened": ["0.0.0.0/0", "wildcard *"], "narrowed": ["removed 0.0.0.0/0", "closed port 22"], "unchanged": ["added tag", "renamed description"]}
    providers.register_mock("cls", lambda s, u, m, t: json.dumps([f"sg-{random.randint(1, 9999)} {random.choice(words[next(l for l in words if f'Label: {l}' in u)])} {i}" for i in range(20)]))
    task, ds_id = ds.start_synthetic("drift", "text_classification", {"labels": [{"name": l} for l in words], "per_label": 60}, provider="cls")
    wait_task(task)
    download("minilm-l6")
    j = wait_job(runner.create("minilm-l6", ds_id, None, None, {"epochs": 3})["id"], timeout=900)
    assert j["status"] == "done", j["error"]
    m = runner.manifest(j["id"])
    assert m["metrics"]["test"]["accuracy"] > 0.6 and m["parity"]["max_abs_diff"] < 1e-3
    t, _ = exports.start(j["id"], "lambda_text", {})
    wait_task(t)
    assert t.status == "done", t.error


@slow
@pytest.mark.parametrize("trainer", ["mlx_lm", "hf_text"])
def test_small_llm_lora_then_gguf(trainer):
    need(trainer)
    if trainer == "mlx_lm" and not plugins.available("mlx_lm")["host_ok"]:
        pytest.skip("Apple Silicon only")

    def llm(system, user, mt, temp):
        if "array of strings" in system:
            return json.dumps([f"build a {random.choice(['garden', 'robot', 'bakery'])} thing {random.random()}" for _ in range(8)])
        return json.dumps({"title": "T", "todos": ["a", "b"]})
    providers.register_mock("llm", llm)
    spec = {"total": 60, "system_prompt": "s", "rubric": "r", "output_keys": ["title", "todos"], "cells": {"k": {"a": {"desc": "a", "share": 1}}}}
    task, ds_id = ds.start_synthetic("cards", "text_to_json", spec, provider="llm")
    wait_task(task)
    download("qwen3-0.6b")
    j = wait_job(runner.create("qwen3-0.6b", ds_id, trainer, None, {"epochs": 0.5, "batch_size": 2, "eval_samples": 4, "max_new_tokens": 60})["id"], timeout=1800)
    assert j["status"] == "done", j["error"]
    arts = runner.manifest(j["id"])["artifacts"]
    assert {"adapter", "hf_model"} <= set(arts) and (runner.job_dir(j["id"]) / "model" / "hf" / "config.json").exists()


@slow
def test_needle_tool_calling_to_cact():
    need("needle")
    tools = [{"name": "set_light", "description": "Turn a light on or off.", "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "on": {"type": "boolean"}}, "required": ["room"]}}]

    def llm(system, user, mt, temp):
        rows = []
        for _ in range(10):
            room, on = random.choice(["kitchen", "garage", "office"]), random.random() < 0.5
            rows.append({"query": f"turn {'on' if on else 'off'} the {room} light {random.randint(1, 9999)}", "answers": [{"name": "set_light", "arguments": {"room": room, "on": on}}], "reasoning": room})
        rows.append({"query": f"capital of france {random.randint(1, 9999)}", "answers": [], "reasoning": ""})
        return json.dumps(rows)
    providers.register_mock("tc", llm)
    task, ds_id = ds.start_synthetic("tools", "tool_calling", {"tools": tools, "total": 70}, provider="tc")
    wait_task(task)
    download("needle3")
    j = wait_job(runner.create("needle3", ds_id, None, None, {"epochs": 6, "layers": 20, "eval_samples": 8})["id"], timeout=1800)
    assert j["status"] == "done", j["error"]
    assert runner.manifest(j["id"])["metrics"]["test"]["exact_match"] >= 0.5
    t, eid = exports.start(j["id"], "cact", {"platform": "(model file only)"})
    wait_task(t)
    assert t.status == "done", t.error
    assert (config.exports_dir() / eid / "package" / "needle3.cact").exists()


@slow
@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("say") or not shutil.which("ffmpeg"), reason="needs macOS say and ffmpeg to make audio")
def test_whistle_new_vocabulary(tmp_path):
    need("whistle")
    import io
    import zipfile
    rows, buf = [], io.BytesIO()
    phrases = ["turn on the kitchen light", "turn off the garage door", "open the office fan", "close the front gate", "dim the bedroom lamp"]
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(40):
            text = phrases[i % 5]
            aiff, wav = tmp_path / f"{i}.aiff", tmp_path / f"{i}.wav"
            subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), text], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(wav)], check=True)
            z.write(wav, f"clips/{i}.wav")
            rows.append(json.dumps({"audio": f"clips/{i}.wav", "text": text, "language": "en"}))
        z.writestr("train.jsonl", "\n".join(rows))
    meta = ds.import_upload("say", "speech", [("d.zip", buf.getvalue())])
    assert meta["status"] == "ready", meta["report"]
    download("whistle")
    j = wait_job(runner.create("whistle", meta["id"], None, None, {"epochs": 1, "batch_size": 8})["id"], timeout=1800)
    assert j["status"] == "done", j["error"]
    assert runner.manifest(j["id"])["metrics"]["test"]["wer"] < 0.5
