import json

import pytest

from automatoner import __main__ as cli
from automatoner import config, ops
from automatoner.store import db
from automatoner.training import runner
from conftest import RHYTHM_SPEC


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_listing_commands_emit_json(capsys):
    code, out, _ = run(capsys, "models", "--category", "tabular_anomaly", "--json")
    assert code == 0 and {m["id"] for m in json.loads(out)} == {"isolation_forest", "one_class_svm", "autoencoder"}
    code, out, _ = run(capsys, "categories", "--json")
    assert code == 0 and len(json.loads(out)) == 9
    code, out, _ = run(capsys, "recipes", "list", "--json")
    assert code == 0 and "rhythm-watcher" in {r["id"] for r in json.loads(out)}
    code, out, _ = run(capsys, "doctor", "--json")
    assert code == 0 and json.loads(out)["summary"]["fail"] == 0


def test_text_tables_are_readable(capsys):
    code, out, _ = run(capsys, "models", "--category", "tabular_anomaly")
    assert code == 0 and out.splitlines()[0].startswith("ID") and "isolation_forest" in out
    code, out, _ = run(capsys, "jobs", "list")
    assert code == 0


def test_whole_workflow_from_the_command_line(capsys, tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(RHYTHM_SPEC))
    code, out, _ = run(capsys, "datasets", "generate", "--category", "tabular_anomaly", "--name", "cli demo", "--spec", str(spec), "--json")
    ds = json.loads(out)
    assert code == 0 and ds["status"] == "ready"

    code, out, _ = run(capsys, "datasets", "list", "--category", "tabular_anomaly", "--json")
    assert ds["id"] in {d["id"] for d in json.loads(out)}
    code, out, _ = run(capsys, "datasets", "show", ds["id"])
    assert code == 0 and "cli demo" in out

    code, out, err = run(capsys, "train", "--model", "isolation_forest", "--dataset", ds["id"], "--config", '{"n_estimators": 30}')
    assert code == 0 and ": done" in out and "auroc" in out
    job_id = next(line.split()[1] for line in err.splitlines() if line.startswith("job "))

    code, out, _ = run(capsys, "jobs", "show", job_id, "--json", "--log-lines", "3")
    j = json.loads(out)
    assert j["status"] == "done" and j["config"]["n_estimators"] == 30 and len(j["log_tail"]) <= 3 and "onnx" in j["artifacts"]

    code, out, _ = run(capsys, "export", "--job", job_id, "--list")
    assert code == 0 and "lambda_zip" in out and "*" in out
    code, out, _ = run(capsys, "export", "--job", job_id, "--exporter", "lambda_zip", "--options", '{"function_name": "cli-fn"}', "--json")
    r = json.loads(out)
    assert code == 0 and r["status"] == "done" and any(f.endswith("handler.py") for f in r["files"])
    assert (config.exports_dir() / r["download_zip"].split("/")[-1]).exists()


def test_train_no_wait_returns_an_id_and_status_catches_up(capsys, tmp_path):
    csv = tmp_path / "d.csv"
    import automatoner.datasets.synth.tabular as tab
    tab.simulate(RHYTHM_SPEC).to_csv(csv, index=False)
    code, out, _ = run(capsys, "datasets", "add", str(csv), "--category", "tabular_anomaly", "--name", "from csv", "--json")
    ds = json.loads(out)
    assert code == 0 and ds["label_column"] == "label"
    code, out, _ = run(capsys, "train", "--model", "isolation_forest", "--dataset", ds["id"], "--no-wait")
    job_id = out.strip()
    assert code == 0 and job_id.startswith("job-")
    assert ops.wait_job(job_id)["status"] == "done"


def test_errors_use_exit_codes_and_clear_messages(capsys):
    code, _, err = run(capsys, "train", "--model", "isolation_forest", "--dataset", "nope")
    assert code == 1 and "unknown dataset" in err
    code, _, err = run(capsys, "train", "--model", "isolation_forest", "--dataset", "x", "--config", "{not json")
    assert code == 2 and "--config is not valid JSON" in err
    code, _, err = run(capsys, "datasets", "generate", "--category", "tabular_anomaly")
    assert code == 2 and "--spec" in err
    code, _, err = run(capsys, "jobs", "show")
    assert code == 2
    code, _, err = run(capsys, "jobs", "cancel", "job-none")
    assert code == 1
    with pytest.raises(SystemExit) as e:
        cli.main(["train", "--model", "x"])
    assert e.value.code == 2


def test_status_is_resolved_for_jobs_started_by_another_process():
    """A worker that finished while no process was watching must not stay 'running' forever."""
    jd = runner.job_dir("job-20260101-000000-aaaaaa")
    (jd / "model").mkdir(parents=True)
    (jd / "status.json").write_text(json.dumps({"status": "done", "error": None}))
    (jd / "model" / "manifest.json").write_text(json.dumps({"metrics": {"test": {"auroc": 0.9}}, "artifacts": {"onnx": "model.onnx"}, "size_bytes": 5}))
    base = {"model_id": "m", "model_name": "m", "dataset_id": "d", "dataset_name": "d", "trainer": "sklearn", "category": "tabular_anomaly", "config": {}, "created": "x", "pid": 99999999}
    db.put("jobs", "job-20260101-000000-aaaaaa", {"id": "job-20260101-000000-aaaaaa", "status": "running", **base})
    done = runner.refresh("job-20260101-000000-aaaaaa")
    assert done["status"] == "done" and done["summary"]["metrics"]["test"]["auroc"] == 0.9

    db.put("jobs", "job-20260101-000001-bbbbbb", {"id": "job-20260101-000001-bbbbbb", "status": "running", **base})
    assert runner.refresh("job-20260101-000001-bbbbbb")["status"] == "interrupted"
    # a second process starting up settles dead jobs and leaves live ones alone
    runner.mark_interrupted()
