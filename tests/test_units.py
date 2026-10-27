import json

from automatoner.training import cactus_common as cc
from automatoner.training import evaluators, plugins
from automatoner.training.hf_common import TOOL_SYSTEM, to_chat
from automatoner.training.plugins import mlx_plugin
from automatoner.training.tabular_common import calibrate, threshold_by_f1

import numpy as np


def test_calibration_puts_threshold_at_half_and_saturates():
    raw = np.random.default_rng(0).normal(0, 1, 1000)
    lo, hi = calibrate(raw, threshold_raw=3.0)
    score = lambda r: float(np.clip((r - lo) / (hi - lo), 0, 1))  # noqa: E731
    assert abs(score(3.0) - 0.5) < 1e-9 and score(-5) == 0.0 and score(100) == 1.0
    assert score(1.5) < 0.5 < score(4.5)


def test_threshold_by_f1_needs_both_classes():
    assert threshold_by_f1(np.array([0.1, 0.2]), np.array([0, 0])) is None
    raw = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    t = threshold_by_f1(raw, np.array([0, 0, 0, 1, 1]))
    assert 0.3 < t <= 0.9


def test_to_chat_converts_needle_rows_and_keeps_meta():
    row = {"query": "lights on", "tools": [{"name": "f", "description": "d", "parameters": {}}], "answers": [{"name": "f", "arguments": {"on": True}}]}
    chat = to_chat(row)
    assert [m["role"] for m in chat["messages"]] == ["system", "user", "assistant"]
    assert chat["messages"][0]["content"].startswith(TOOL_SYSTEM.split("{tools}")[0][:30])
    assert json.loads(chat["messages"][2]["content"]) == row["answers"]
    assert to_chat({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], "meta": {"k": 1}})["meta"] == {"k": 1}


def test_tool_call_scoring():
    ref = json.dumps([{"name": "f", "arguments": {"a": 1, "b": 2}}])
    same_other_order = json.dumps([{"name": "f", "arguments": {"b": 2, "a": 1}}])
    assert evaluators.tool_calls({"reference": ref, "output": same_other_order})["exact_match"] == 1.0
    wrong_arg = json.dumps([{"name": "f", "arguments": {"a": 9, "b": 2}}])
    s = evaluators.tool_calls({"reference": ref, "output": "sure: " + wrong_arg})
    assert s["names_match"] == 1.0 and s["exact_match"] == 0.0 and s["valid_json"] == 1.0
    assert evaluators.tool_calls({"reference": "[]", "output": "[]"})["exact_match"] == 1.0
    assert evaluators.tool_calls({"reference": ref, "output": "I cannot"})["valid_json"] == 0.0


def test_generic_json_scoring_requires_same_keys_in_order():
    ref = '{"a": 1, "b": "x"}'
    assert evaluators.json_generic({"reference": ref, "output": '{"a": 2, "b": "y"}'})["exact_keys"] == 1.0
    assert evaluators.json_generic({"reference": ref, "output": '{"b": "y", "a": 2}'})["exact_keys"] == 0.0


def test_evaluate_groups_by_meta():
    rows = [{"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": '{"k": 1}'}], "meta": {"lang": "en" if i % 2 else "ro"}} for i in range(6)]
    overall, md = evaluators.evaluate("json_generic", rows, lambda s, u: '{"k": 5}', limit=6, title="t")
    assert overall["valid_json"] == 1.0 and "## By lang" in md and "| ro | 3 |" in md


def test_mlx_log_parsing_handles_old_and_new_formats():
    for line in ("  70   1.020 ▼   1,667   10.3k", "Iter 20: Train loss 1.2, Learning Rate 1e-4"):
        assert mlx_plugin.ITER_TRAIN.search(line) and not mlx_plugin.ITER_VAL.search(line)
    for line in ("  72  val 0.732  0.21s", "Iter 200: Val loss 1.331, Val took 1s"):
        assert mlx_plugin.ITER_VAL.search(line)
    assert mlx_plugin.TEST_LOSS.search("Test loss 0.490, Test ppl 1.633.").group(1) == "0.490"


def test_cactus_log_parsing():
    epoch = cc.LOSS_PATTERNS[0][0].search("  epoch     1/2  loss 0.5782  val 0.5162")
    assert epoch.groups() == ("1", "0.5782", "0.5162")
    assert cc.LOSS_PATTERNS[1][0].search("  step      3/8  loss 0.5997").groups() == ("3", "0.5997")
    assert cc.HELD_OUT.search("  accuracy  4/9 held-out calls exact, scored on").groups() == ("4", "9")


def test_plugin_availability_reports_missing_parts():
    for pid in plugins.INFO:
        a = plugins.available(pid)
        assert set(a) >= {"ready", "missing", "host_ok", "extra"}
        assert a["ready"] == (not a["missing"] and a["host_ok"])
