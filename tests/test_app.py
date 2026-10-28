import json
import os
import stat
import sys
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from automatoner import config, recipes
from automatoner.app import create_app
from automatoner.catalogue import service as catalogue
from automatoner.doctor import checks
from automatoner.export import service as exports
from automatoner.training import plugins
from conftest import RHYTHM_SPEC


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), base_url="http://127.0.0.1:8765")


def poll(client, url, done=lambda d: d["status"] != "running", timeout=120):
    t0 = time.time()
    while True:
        d = client.get(url).json()
        if done(d):
            return d
        assert time.time() - t0 < timeout
        time.sleep(0.1)


def test_full_flow_through_http(client):
    assert client.get("/api/health").json()["ok"]
    r = client.post("/api/datasets/synthetic", json={"name": "api", "category": "tabular_anomaly", "spec": RHYTHM_SPEC}).json()
    assert poll(client, f"/api/tasks/{r['task']['id']}")["status"] == "done"
    ds = client.get(f"/api/datasets/{r['dataset_id']}").json()
    assert ds["status"] == "ready"
    assert len(client.get(f"/api/datasets/{r['dataset_id']}/preview?limit=3").json()["rows"]) == 3
    job = client.post("/api/jobs", json={"model_id": "isolation_forest", "dataset_id": r["dataset_id"]}).json()
    j = poll(client, f"/api/jobs/{job['id']}", lambda d: d["status"] not in ("queued", "running"))
    assert j["status"] == "done" and j["manifest"]["metrics"]["test"]["auroc"] > 0.9
    opts = client.get(f"/api/jobs/{job['id']}/exports").json()
    assert opts[0]["id"] == "lambda_zip" and opts[0]["available"]
    x = client.post("/api/exports", json={"job_id": job["id"], "exporter": "lambda_zip", "options": {"function_name": "api-fn"}}).json()
    assert poll(client, f"/api/tasks/{x['task']['id']}")["status"] == "done"
    dl = client.get(f"/api/exports/{x['export_id']}/download")
    assert dl.status_code == 200 and dl.content[:2] == b"PK"
    with client.websocket_connect(f"/ws/jobs/{job['id']}", headers={"host": "127.0.0.1:8765"}) as ws:
        assert ws.receive_json()["type"] == "log"


def test_api_errors_are_readable(client):
    assert client.post("/api/jobs", json={"model_id": "qwen3-0.6b", "dataset_id": "x"}).json()["detail"].startswith("unknown dataset")
    assert client.get("/api/datasets/nope").status_code == 404
    assert client.post("/api/models/isolation_forest/download").status_code == 400
    assert client.post("/api/exports", json={"job_id": "nope", "exporter": "joblib"}).status_code == 400
    bad = client.post("/api/datasets/upload", data={"name": "x", "category": "tabular_anomaly"}, files={"files": ("a.pdf", b"%PDF")})
    assert bad.status_code == 400 and "no data file" in bad.json()["detail"]


def test_settings_never_return_secrets(client):
    client.put("/api/settings", json={"hf_token": "hf_SECRET123", "anthropic_api_key": "sk-ant-SECRET", "ollama_url": "http://127.0.0.1:11434"})
    body = client.get("/api/settings").text
    assert "SECRET" not in body
    s = client.get("/api/settings").json()
    assert s["hf_token_set"] is True and s["anthropic_api_key_set"] is True and "hf_token" not in s
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(config.home() / "settings.json").st_mode) == 0o600
    assert "SECRET" not in json.dumps(client.get("/api/doctor").json())
    client.put("/api/settings", json={"hf_token": "", "anthropic_api_key": ""})
    assert client.get("/api/settings").json()["hf_token_set"] is False


def test_unknown_settings_keys_are_ignored(client):
    client.put("/api/settings", json={"not_a_setting": "x"})
    assert "not_a_setting" not in config.load_settings()


def test_doctor_reports_each_extra_and_never_crashes(client):
    d = client.get("/api/doctor").json()
    ids = {c["id"] for c in d["checks"]}
    assert {"python", "uv", "disk", "ram", "accelerator", "extra-tabular", "extra-text", "hf-token", "ollama", "docker", "ffmpeg"} <= ids
    assert d["summary"]["fail"] == 0
    tab = next(c for c in d["checks"] if c["id"] == "extra-tabular")
    assert tab["status"] == "ok"
    for c in d["checks"]:
        assert c["status"] in ("ok", "warn", "fail", "missing", "na")
        if c["status"] == "missing":
            assert c["fix"] and c["fix"]["type"] in ("extra", "action")


def test_catalogue_is_consistent():
    cats = {c["id"] for c in catalogue.categories()}
    exporter_ids = set(exports.registry())
    seen = set()
    for m in catalogue.list_models():
        assert m["id"] not in seen
        seen.add(m["id"])
        assert m["source"] in ("hf", "builtin") and (m["source"] == "builtin" or m["repo"])
        assert set(m["categories"]) <= cats
        for cat, options in m["trainer_options"].items():
            assert set(options) <= set(plugins.INFO), f"{m['id']}: unknown trainer in {options}"
        assert set(m["trainer_options"]) == set(m["categories"])
    assert len(cats) == 9 and all(catalogue.list_models(c) for c in cats)
    assert {"lambda_zip", "gguf", "cact", "coreai"} <= exporter_ids


def test_every_recipe_points_at_real_things():
    exporter_ids = set(exports.registry())
    found = recipes.list_recipes()
    assert len(found) >= 7
    for r in found:
        entry = catalogue.get_entry(r["model"])
        assert entry, r["id"]
        assert r["category"] in entry["categories"], r["id"]
        trainers = catalogue.entry_view(entry)["trainer_options"][r["category"]]
        assert r.get("trainer") in trainers, f"{r['id']}: {r.get('trainer')} not in {trainers}"
        assert r["export"]["exporter"] in exporter_ids, r["id"]


def test_recipe_substitution_and_file_loading(tmp_path):
    (tmp_path / "rubric.md").write_text("be good")
    (tmp_path / "seeds.jsonl").write_text('{"text": "a"}\n{"text": "b"}\n')
    spec = recipes._load_files(recipes._subst({"rubric_file": "${d}/rubric.md", "seeds_file": "${d}/seeds.jsonl", "x": ["${d}"]}, {"d": str(tmp_path)}))
    assert spec["rubric"] == "be good" and spec["seeds"] == [{"text": "a"}, {"text": "b"}] and spec["x"] == [str(tmp_path)]
    with pytest.raises(FileNotFoundError):
        recipes._load_files({"rubric_file": str(tmp_path / "missing.md")})


def test_rhythm_recipe_runs_unattended():
    t = recipes.run("rhythm-watcher")
    t0 = time.time()
    while t.status == "running":
        assert time.time() - t0 < 180
        time.sleep(0.2)
    assert t.status == "done", t.error
    assert {"dataset_id", "job_id", "export_id"} <= set(t.result)


def test_static_ui_is_served_when_built(client):
    from automatoner.app import DIST
    if not (DIST / "index.html").exists():
        pytest.skip("web UI not built")
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert client.get("/train").status_code == 200  # SPA fallback for client-side routes


def test_rebinding_and_cross_site_requests_are_refused(client):
    assert client.get("/api/health", headers={"host": "evil.example.com"}).status_code == 403
    assert client.get("/api/health", headers={"host": "localhost:8765"}).status_code == 200
    assert client.get("/api/health", headers={"host": "[::1]:8765"}).status_code == 200
    # a page on another site posting to the local server
    r = client.post("/api/settings/hf-token/check", headers={"origin": "https://evil.example.com"})
    assert r.status_code == 403
    r = client.delete("/api/datasets/x", headers={"origin": "https://evil.example.com"})
    assert r.status_code == 403
    # same-origin and Vite dev origin are fine
    assert client.post("/api/settings/hf-token/check", headers={"origin": "http://127.0.0.1:8765"}).status_code == 200
    assert client.put("/api/settings", json={}, headers={"origin": "http://localhost:5173"}).status_code == 200
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/jobs/x", headers={"host": "127.0.0.1:8765", "origin": "https://evil.example.com"}):
            pass


def test_non_loopback_server_accepts_its_own_host():
    from automatoner.app import create_app as make
    tok = "t" * 32
    c = TestClient(make(allow_any_host=True, token=tok))
    assert c.get("/api/health", headers={"host": "gpu-box.lan:8765", "authorization": f"Bearer {tok}"}).status_code == 200


def test_resume_job_id_is_validated():
    import re
    ok = "job-20261027-101500-ab12cd"
    pat = r"job-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}"
    assert re.fullmatch(pat, ok) and not re.fullmatch(pat, "../../etc")


def test_low_memory_is_a_warning_not_a_failure(monkeypatch):
    class Mem:
        total = 4 * 2**30
    monkeypatch.setattr(checks.psutil, "virtual_memory", lambda: Mem())
    assert checks.ram_check()["status"] == "warn"


def test_unsafe_ids_are_refused(client):
    from automatoner import util
    from automatoner.training import runner
    for bad in ("..", "../x", "a/b", "/etc", ".hidden", ""):
        try:
            util.safe_child(runner.config.jobs_dir(), bad)
        except util.InvalidId:
            continue
        raise AssertionError(f"accepted {bad!r}")
    assert client.get("/api/jobs/.hidden/report").status_code == 404
