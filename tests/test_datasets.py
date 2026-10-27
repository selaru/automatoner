import io
import json
import random
import zipfile

import pytest

from automatoner.datasets import providers, schemas, service as ds, validators
from automatoner.datasets.importers import files as ifiles
from automatoner.datasets.synth import tabular
from conftest import RHYTHM_SPEC, wait_task


def test_simulator_balances_and_labels():
    df = tabular.simulate(RHYTHM_SPEC)
    assert len(df) == 4000 and set(df["label"]) == {0, 1}
    assert abs(df["label"].mean() - 0.05) < 0.005
    # anomalies really differ from normal rows
    assert df[df.label == 1]["error_rate"].mean() > 3 * df[df.label == 0]["error_rate"].mean()


def test_simulator_is_deterministic_and_names_labels():
    a, b = tabular.simulate(RHYTHM_SPEC), tabular.simulate(RHYTHM_SPEC)
    assert a.equals(b)
    named = tabular.simulate({**RHYTHM_SPEC, "label_names": ["ok", "bad"]})
    assert set(named["label"]) == {"ok", "bad"}


@pytest.mark.parametrize("patch,msg", [
    ({"columns": []}, "non-empty"),
    ({"anomaly_rate": 0.9}, "anomaly_rate"),
    ({"rows": 5}, "rows"),
    ({"columns": [{"name": "x", "kind": "bogus"}]}, "kind"),
    ({"columns": [{"name": "x", "kind": "normal"}, {"name": "x", "kind": "normal"}]}, "unique"),
])
def test_simulator_rejects_bad_specs(patch, msg):
    with pytest.raises(ValueError, match=msg):
        tabular.simulate({**RHYTHM_SPEC, **patch})


def test_split_is_stratified_and_disjoint():
    strata = ["a"] * 90 + ["b"] * 10
    tr, va, te = validators.split_indices(strata, 0.1, 0.1, seed=1)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) + len(va) + len(te) == 100
    assert any(strata[i] == "b" for i in te) and any(strata[i] == "b" for i in va)


def test_clean_rows_drops_duplicates_and_bad_json():
    good = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": '{"a": 1}'}]}
    dup = json.loads(json.dumps(good))
    bad = {"messages": [{"role": "user", "content": "yo"}, {"role": "assistant", "content": "not json"}]}
    rows, rep = validators.clean_rows("text_to_json", [good, dup, bad] + [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "{}"}]} for i in range(12)])
    assert len(rows) == 13 and rep.dropped == 2 and not rep.errors


def test_needle_rows_must_call_known_tools():
    tools = [{"name": "f", "description": "x", "parameters": {}}]
    assert schemas.row_problem("tool_calling", {"query": "q", "tools": tools, "answers": [{"name": "f", "arguments": {}}]}) is None
    assert "unknown tool" in schemas.row_problem("tool_calling", {"query": "q", "tools": tools, "answers": [{"name": "g", "arguments": {}}]})
    assert schemas.row_problem("tool_calling", {"query": "q", "tools": tools, "answers": []}) is None


def test_zip_slip_is_rejected(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="unsafe path"):
        ifiles.safe_extract_zip(buf.getvalue(), tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_upload_csv_detects_label_and_columns():
    df = tabular.simulate(RHYTHM_SPEC)
    meta = ds.import_upload("mine", "tabular_anomaly", [("data.csv", df.to_csv(index=False).encode())])
    assert meta["status"] == "ready" and meta["label_column"] == "label"
    assert meta["numeric_columns"] == ["hour", "requests", "error_rate"] and meta["categorical_columns"] == ["kind"]
    assert sum(meta["splits"].values()) == 4000


def test_classification_without_label_is_invalid():
    df = tabular.simulate(RHYTHM_SPEC).drop(columns=["label"])
    meta = ds.import_upload("nolabel", "tabular_classification", [("d.csv", df.to_csv(index=False).encode())])
    assert meta["status"] == "invalid" and any("label" in e for e in meta["report"]["errors"])


def test_upload_without_a_table_leaves_nothing_behind():
    from automatoner import config
    before = set(p.name for p in config.datasets_dir().iterdir())
    with pytest.raises(ValueError):
        ds.import_upload("bad", "tabular_anomaly", [("notes.pdf", b"%PDF")])
    assert set(p.name for p in config.datasets_dir().iterdir()) == before


def test_provider_json_extraction():
    assert providers.extract_json('<think>hmm</think>```json\n["a","b"]\n```') == ["a", "b"]
    assert providers.extract_json('Sure! {"x": 1} done') == {"x": 1}
    with pytest.raises(ValueError):
        providers.extract_json("no json here")


def test_text_classification_generation_with_scripted_llm():
    def llm(system, user, max_tokens, temp):
        label = "A" if "Label: A" in user else "B"
        return json.dumps([f"{label} example {random.random()}" for _ in range(20)] + [f"{label} repeated"] * 3)
    providers.register_mock("m1", llm)
    task, ds_id = ds.start_synthetic("tc", "text_classification", {"labels": [{"name": "A"}, {"name": "B"}], "per_label": 40}, provider="m1")
    wait_task(task)
    meta = ds.get(ds_id)
    assert meta["status"] == "ready" and meta["rows"] == 80 and meta["labels"] == ["A", "B"]


def test_text_to_json_two_stage_generation_and_resume():
    calls = {"n": 0}

    def llm(system, user, max_tokens, temp):
        calls["n"] += 1
        if "JSON array of strings" in system:
            return json.dumps([f"note {random.random()}" for _ in range(8)])
        return json.dumps({"title": "T", "todos": ["a", "b"]})
    providers.register_mock("m2", llm)
    spec = {"total": 24, "system_prompt": "sys", "rubric": "r", "output_keys": ["title", "todos"], "cells": {"lang": {"en": {"desc": "English", "share": 1}}}}
    task, ds_id = ds.start_synthetic("cards", "text_to_json", spec, provider="m2")
    wait_task(task)
    meta = ds.get(ds_id)
    assert meta["status"] == "ready" and meta["rows"] >= 20
    row = json.loads((ds.dataset_dir(ds_id) / "train.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"] and row["messages"][0]["content"] == "sys"
    assert list(json.loads(row["messages"][2]["content"])) == ["title", "todos"]


def test_wrong_key_order_from_llm_is_rejected():
    providers.register_mock("m3", lambda s, u, m, t: json.dumps(["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]) if "array of strings" in s else '{"todos": [], "title": "T"}')
    task, ds_id = ds.start_synthetic("bad", "text_to_json", {"total": 10, "system_prompt": "s", "rubric": "r", "output_keys": ["title", "todos"]}, provider="m3")
    wait_task(task)
    assert ds.get(ds_id)["status"] in ("invalid", "error")
