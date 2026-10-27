import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from automatoner import config
from automatoner.export import service as exports
from automatoner.store import db
from automatoner.training import runner
from conftest import needs_text, wait_job, wait_task


def train(model, dataset, cfg=None):
    job = runner.create(model, dataset, None, None, cfg or {})
    return wait_job(job["id"])


@pytest.mark.parametrize("model,floor", [("isolation_forest", 0.9), ("one_class_svm", 0.8)])
def test_anomaly_models_separate_anomalies(model, floor, anomaly_dataset):
    j = train(model, anomaly_dataset)
    assert j["status"] == "done", j["error"]
    m = runner.manifest(j["id"])
    assert m["metrics"]["test"]["auroc"] > floor
    assert m["parity"]["max_abs_diff"] < 1e-3 and m["inference"]["threshold"] == 0.5
    assert m["onnx_bytes"] < 10_000_000


@pytest.mark.parametrize("model", ["lightgbm", "random_forest", "logistic_regression"])
def test_classifiers_learn(model, classification_dataset):
    j = train(model, classification_dataset)
    assert j["status"] == "done", j["error"]
    m = runner.manifest(j["id"])
    assert m["metrics"]["test"]["accuracy"] > 0.9 and m["parity"]["max_abs_diff"] < 1e-3
    assert m["inference"]["classes"] == ["normal", "strange"]


def test_lambda_package_flags_an_outlier_and_passes_a_normal_row(anomaly_dataset, tmp_path):
    """The generated handler, run exactly as Lambda would, must agree with the model."""
    j = train("isolation_forest", anomaly_dataset)
    task, export_id = exports.start(j["id"], "lambda_zip", {"function_name": "t", "memory": 128})
    wait_task(task)
    assert task.status == "done", task.error
    pkg = config.exports_dir() / export_id / "package"
    assert {"handler.py", "model.onnx", "features.json", "inference.json", "template.yaml", "main.tf", "t.zip"} <= {p.name for p in pkg.iterdir()}
    code = ("import json,handler;"
            "n=handler.handler({'records':[{'hour':12,'requests':200,'error_rate':0.01,'kind':'a'},{'hour':12,'requests':900,'error_rate':0.4,'kind':'b'}]});"
            "print(json.dumps(n))")
    out = json.loads(subprocess.run([sys.executable, "-c", code], cwd=pkg, capture_output=True, text=True, check=True).stdout)["results"]
    assert out[0]["is_anomaly"] is False and out[0]["score"] < 0.5
    assert out[1]["is_anomaly"] is True and out[1]["score"] > 0.5
    assert any(name in out[1]["reason"] for name in ("error_rate", "requests"))


def test_lambda_zip_contains_only_what_lambda_needs(anomaly_dataset):
    import zipfile
    j = train("isolation_forest", anomaly_dataset)
    task, export_id = exports.start(j["id"], "lambda_zip", {"function_name": "z"})
    wait_task(task)
    with zipfile.ZipFile(exports.download_path(export_id)) as z:
        names = set(z.namelist())
    assert {"handler.py", "z.zip", "model.onnx"} <= names
    with zipfile.ZipFile(config.exports_dir() / export_id / "package" / "z.zip") as inner:
        assert set(inner.namelist()) == {"handler.py", "model.onnx", "features.json", "inference.json"}


def test_joblib_bundle_predicts(classification_dataset):
    j = train("logistic_regression", classification_dataset)
    task, export_id = exports.start(j["id"], "joblib", {})
    wait_task(task)
    pkg = config.exports_dir() / export_id / "package"
    rec = {"records": [{"hour": 12, "requests": 900, "error_rate": 0.4, "kind": "b"}]}
    out = subprocess.run([sys.executable, "predict.py"], cwd=pkg, input=json.dumps(rec), capture_output=True, text=True, check=True).stdout
    assert json.loads(out)[0]["label"] == "strange"


def test_export_options_explain_why_something_is_unavailable(classification_dataset):
    j = train("lightgbm", classification_dataset)
    opts = {o["id"]: o for o in exports.options_for(j["id"])}
    assert opts["lambda_zip"]["available"] and opts["joblib"]["available"]
    assert not opts["gguf"]["available"] and opts["gguf"]["reason"]
    assert not opts["cact"]["available"]


def test_job_refuses_wrong_category_and_missing_dataset(anomaly_dataset):
    with pytest.raises(runner.JobError, match="cannot be trained for"):
        runner.create("lightgbm", anomaly_dataset, None, None, {})
    with pytest.raises(runner.JobError, match="unknown dataset"):
        runner.create("isolation_forest", "nope", None, None, {})
    from automatoner.datasets import service as ds
    rows = "\n".join(json.dumps({"text": f"sample {i}", "label": "ab"[i % 2]}) for i in range(40)).encode()
    text_ds = ds.import_upload("tiny text", "text_classification", [("d.jsonl", rows)])["id"]
    with pytest.raises(runner.JobError, match="not available yet"):
        runner.create("gliner2-base", text_ds, None, None, {})


def test_cancel_stops_the_process_tree(anomaly_dataset):
    j = runner.create("one_class_svm", anomaly_dataset, None, None, {"max_train_rows": 50000, "nu": 0.2})
    assert runner.cancel(j["id"]) is True
    assert wait_job(j["id"])["status"] == "cancelled"


@needs_text
def test_autoencoder_flags_outlier_in_its_handler(anomaly_dataset):
    j = train("autoencoder", anomaly_dataset, {"epochs": 15})
    assert j["status"] == "done", j["error"]
    m = runner.manifest(j["id"])
    assert m["metrics"]["test"]["auroc"] > 0.9 and m["parity"]["max_abs_diff"] < 1e-3 and "torchscript" in m["artifacts"]
