import importlib.util
import os
import tempfile
import time

import pytest

# One data folder for the whole test run, so no test touches the real ~/.automatoner.
_HOME = tempfile.mkdtemp(prefix="automatoner-tests-")
os.environ["AUTOMATONER_HOME"] = _HOME

HAS_TEXT = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
needs_text = pytest.mark.skipif(not HAS_TEXT, reason="needs the 'text' extra")

RHYTHM_SPEC = {
    "rows": 4000, "anomaly_rate": 0.05, "seed": 3, "label_column": "label",
    "columns": [
        {"name": "hour", "kind": "uniform_int", "low": 0, "high": 23},
        {"name": "requests", "kind": "seasonal", "base": 200, "amplitude": 100, "period_column": "hour", "period": 24, "noise": 10, "anomaly": {"mult": 3}},
        {"name": "error_rate", "kind": "normal", "mean": 0.01, "std": 0.004, "min": 0, "max": 1, "anomaly": {"shift_std": 9}},
        {"name": "kind", "kind": "category", "values": ["a", "b"], "probs": [0.9, 0.1], "anomaly": {"probs": [0.2, 0.8]}},
    ],
}


def wait_task(task, timeout=120):
    t0 = time.time()
    while task.status == "running":
        assert time.time() - t0 < timeout, "task timed out"
        time.sleep(0.05)
    return task


def wait_job(job_id, timeout=300):
    from automatoner.store import db
    t0 = time.time()
    while db.get("jobs", job_id)["status"] in ("queued", "running"):
        assert time.time() - t0 < timeout, "job timed out"
        time.sleep(0.2)
    return db.get("jobs", job_id)


@pytest.fixture(scope="session")
def home():
    return _HOME


@pytest.fixture(scope="session")
def anomaly_dataset():
    from automatoner.datasets import service as ds
    task, ds_id = ds.start_synthetic("rhythm", "tabular_anomaly", RHYTHM_SPEC)
    wait_task(task)
    assert ds.get(ds_id)["status"] == "ready"
    return ds_id


@pytest.fixture(scope="session")
def classification_dataset():
    from automatoner.datasets import service as ds
    spec = {**RHYTHM_SPEC, "label_names": ["normal", "strange"]}
    task, ds_id = ds.start_synthetic("events", "tabular_classification", spec)
    wait_task(task)
    assert ds.get(ds_id)["status"] == "ready"
    return ds_id
