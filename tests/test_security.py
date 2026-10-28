"""Access token, local-only mode and the audit log."""
import json
import sys

import pytest
from fastapi.testclient import TestClient

from automatoner import audit, auth, config
from automatoner.app import create_app
from automatoner.datasets import providers

TOKEN = "s3cret-" + "x" * 30
LOCAL = "http://127.0.0.1:8765"


def entries(action=None):
    return audit.read(500, action)


@pytest.fixture
def secured():
    return TestClient(create_app(token=TOKEN), base_url=LOCAL)


# --- access token ------------------------------------------------------------------------------------------

def test_remote_listening_needs_a_token(monkeypatch):
    monkeypatch.delenv("AUTOMATONER_TOKEN", raising=False)
    with pytest.raises(auth.TokenRequired, match="refusing to listen"):
        create_app(allow_any_host=True)
    create_app(allow_any_host=True, token=TOKEN)  # fine with one


def test_loopback_without_token_stays_open():
    c = TestClient(create_app(), base_url=LOCAL)
    assert c.get("/api/categories").status_code == 200
    assert c.get("/api/auth/status").json() == {"required": False, "authenticated": True}


def test_api_refuses_requests_without_the_token(secured):
    assert secured.get("/api/categories").status_code == 401
    assert secured.get("/api/settings").status_code == 401
    assert secured.put("/api/settings", json={"local_only": True}).status_code == 401
    assert secured.get("/api/auth/status").json() == {"required": True, "authenticated": False}
    assert secured.get("/").status_code == 200  # the sign-in page itself must load


def test_bearer_token_works_and_wrong_one_does_not(secured):
    assert secured.get("/api/categories", headers={"authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert secured.get("/api/categories", headers={"authorization": "Bearer nope"}).status_code == 401
    assert secured.get("/api/categories", headers={"authorization": TOKEN}).status_code == 401


def test_browser_login_sets_an_httponly_cookie_that_is_not_the_token(secured, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", lambda *_: _noop())
    bad = secured.post("/api/auth/login", json={"token": "wrong"})
    assert bad.status_code == 401
    ok = secured.post("/api/auth/login", json={"token": TOKEN})
    assert ok.status_code == 200
    cookie = ok.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert TOKEN not in cookie
    assert secured.get("/api/categories").status_code == 200  # the client keeps the cookie
    secured.post("/api/auth/logout")
    assert secured.get("/api/categories").status_code == 401


async def _noop():
    return None


def test_websockets_need_the_token(secured):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with secured.websocket_connect("/ws/tasks/none"):
            pass
    with secured.websocket_connect("/ws/tasks/none", headers={"authorization": f"Bearer {TOKEN}", "host": "127.0.0.1:8765"}) as ws:
        assert ws.receive_json()["type"] == "missing"


def test_login_is_throttled_and_audited(secured, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", lambda *_: _noop())
    for _ in range(10):
        assert secured.post("/api/auth/login", json={"token": "bad"}).status_code == 401
    assert secured.post("/api/auth/login", json={"token": TOKEN}).status_code == 429
    assert any(e["action"] == "auth.failed" for e in entries("auth"))


def test_token_file_is_private_and_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOMATONER_HOME", str(tmp_path))
    monkeypatch.delenv("AUTOMATONER_TOKEN", raising=False)
    assert auth.configured_token() is None
    t = auth.create_token()
    assert len(t) >= auth.MIN_LENGTH and auth.configured_token() == t
    if sys.platform != "win32":  # Windows has no owner-only file modes; the data folder sits in the user profile
        assert (auth.token_path().stat().st_mode & 0o077) == 0
    with pytest.raises(ValueError):
        auth.create_token()
    assert auth.create_token(rotate=True) != t
    monkeypatch.setenv("AUTOMATONER_TOKEN", "e" * 24)
    assert auth.configured_token() == "e" * 24
    monkeypatch.setenv("AUTOMATONER_TOKEN", "short")
    with pytest.raises(auth.TokenRequired):
        auth.configured_token()


def test_token_cli_never_leaks_via_status(monkeypatch, tmp_path, capsys):
    from automatoner.__main__ import main
    monkeypatch.setenv("AUTOMATONER_HOME", str(tmp_path))
    monkeypatch.delenv("AUTOMATONER_TOKEN", raising=False)
    assert main(["token", "create"]) == 0
    created = capsys.readouterr().out.strip()
    assert main(["token"]) == 0
    assert created not in capsys.readouterr().out
    assert main(["token", "create"]) == 2  # already exists
    assert main(["token", "remove"]) == 0 and auth.configured_token() is None


def test_serve_refuses_remote_host_without_token(monkeypatch, tmp_path, capsys):
    from automatoner.__main__ import main
    monkeypatch.setenv("AUTOMATONER_HOME", str(tmp_path))
    monkeypatch.delenv("AUTOMATONER_TOKEN", raising=False)
    assert main(["serve", "--host", "0.0.0.0", "--no-browser"]) == 1
    assert "refusing to listen" in capsys.readouterr().err


# --- local-only mode ---------------------------------------------------------------------------------------

@pytest.fixture
def local_only_on(monkeypatch):
    monkeypatch.delenv("AUTOMATONER_LOCAL_ONLY", raising=False)
    old = config.load_settings().get("local_only")
    config.save_settings({"local_only": True})
    yield
    config.save_settings({"local_only": bool(old)})


def test_local_only_blocks_cloud_providers_but_not_ollama(local_only_on):
    config.save_settings({"anthropic_api_key": "sk-test-key", "ollama_url": "http://127.0.0.1:11434"})
    with pytest.raises(providers.ProviderError, match="Local-only mode is on"):
        providers.get_provider("anthropic")
    providers.get_provider("ollama")
    assert any(e["action"] == "llm.blocked" and e["provider"] == "anthropic" for e in entries("llm"))


def test_local_only_refuses_an_ollama_url_outside_the_network(local_only_on):
    old = config.load_settings()["ollama_url"]
    try:
        config.save_settings({"ollama_url": "https://ollama.example.com"})
        with pytest.raises(providers.ProviderError, match="not on this computer or a private network"):
            providers.get_provider("ollama")
        for ok in ("http://10.0.0.5:11434", "http://gpu-box:11434", "http://gpu.corp.internal:11434", "http://localhost:11434"):
            assert providers.internal_url(ok), ok
        assert not providers.internal_url("https://api.openai.com")
    finally:
        config.save_settings({"ollama_url": old})


def test_local_only_flag_via_http_and_providers_list(monkeypatch):
    monkeypatch.delenv("AUTOMATONER_LOCAL_ONLY", raising=False)
    c = TestClient(create_app(), base_url=LOCAL)
    try:
        assert c.put("/api/settings", json={"local_only": True}).json()["local_only"] is True
        listing = c.get("/api/providers").json()
        assert listing["local_only"] and {p["id"]: p["allowed"] for p in listing["providers"]}["anthropic"] is False
    finally:
        c.put("/api/settings", json={"local_only": False})


def test_environment_pins_the_policy(monkeypatch):
    monkeypatch.setenv("AUTOMATONER_LOCAL_ONLY", "1")
    c = TestClient(create_app(), base_url=LOCAL)
    s = c.get("/api/settings").json()
    assert s["local_only"] is True and s["local_only_locked"] is True
    assert c.put("/api/settings", json={"local_only": False}).status_code == 409
    with pytest.raises(providers.ProviderError):
        providers.get_provider("openai")
    monkeypatch.setenv("AUTOMATONER_LOCAL_ONLY", "0")
    assert config.local_only() is False and config.local_only_locked()


def test_mocks_still_work_in_local_only_mode(local_only_on):
    providers.register_mock("fake", lambda s, u, m, t: "hi")
    assert providers.ask(providers.get_provider("fake"), "s", "u") == "hi"


# --- audit log ---------------------------------------------------------------------------------------------

def test_llm_calls_are_logged_by_size_never_by_content():
    providers.register_mock("audited", lambda s, u, m, t: "the reply SECRET-REPLY")
    providers.ask(providers.get_provider("audited"), "system text", "SECRET-PROMPT")
    e = entries("llm.call")[0]
    assert e["provider"] == "audited" and e["outcome"] == "ok" and e["sent_chars"] == len("system text") + len("SECRET-PROMPT")
    assert "SECRET" not in json.dumps(e)


def test_settings_changes_name_the_key_but_not_the_value():
    config.save_settings({"openai_api_key": "sk-very-secret-value", "hf_token": "hf_abcdef"})
    config.save_settings({"hf_token": ""})
    text = audit.path().read_text()
    assert "sk-very-secret-value" not in text and "hf_abcdef" not in text
    changes = [e["changes"] for e in entries("settings.change")]
    flat = [c for ch in changes for c in ch]
    assert "openai_api_key: set" in flat and "hf_token: set" in flat and "hf_token: cleared" in flat


def test_web_requests_are_attributed_and_background_work_keeps_the_actor(secured):
    r = secured.post("/api/datasets/synthetic", headers={"authorization": f"Bearer {TOKEN}"},
                     json={"name": "audit-me", "category": "tabular_anomaly",
                           "spec": {"rows": 300, "anomaly_rate": 0.05, "seed": 1, "label_column": "label",
                                    "columns": [{"name": "x", "kind": "normal", "mean": 0, "std": 1, "anomaly": {"shift_std": 8}}]}})
    assert r.status_code == 200
    e = entries("dataset.generate")[0]
    assert e["actor"].startswith("web:") and e["name"] == "audit-me"


def test_denied_requests_are_logged(secured):
    secured.get("/api/categories")
    assert any(e["action"] == "auth.denied" for e in entries("auth"))


def test_audit_endpoint_and_cli(secured, capsys):
    from automatoner.__main__ import main
    got = secured.get("/api/audit?limit=5", headers={"authorization": f"Bearer {TOKEN}"})
    assert got.status_code == 200 and isinstance(got.json(), list)
    assert secured.get("/api/audit").status_code == 401
    assert main(["datasets", "list"]) == 0
    capsys.readouterr()
    assert main(["audit", "--limit", "5", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows and all({"time", "actor", "action"} <= set(r) for r in rows)
    assert any(r["actor"].startswith("cli:") for r in entries("cli"))


def test_audit_file_is_private_rotates_and_survives_bad_input(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOMATONER_AUDIT_FILE", str(tmp_path / "a" / "audit.log"))
    audit.record("test.one", api_key="sk-nope", note="x" * 500, nested={"password": "p", "ok": 1})
    p = audit.path()
    if sys.platform != "win32":
        assert (p.stat().st_mode & 0o077) == 0
    e = audit.read(1)[0]
    assert e["api_key"] == "<hidden>" and e["nested"]["password"] == "<hidden>" and len(e["note"]) < 210
    monkeypatch.setattr(audit, "MAX_BYTES", 200)
    for i in range(10):
        audit.record("test.fill", i=i)
    assert p.with_name("audit.log.1").exists()
    p.write_text("not json\n" + p.read_text())
    assert audit.read(3)


def test_mcp_tool_calls_are_audited():
    pytest.importorskip("mcp")
    import anyio
    from mcp.client import Client
    from automatoner.mcp_server import Policy, build_server

    async def go():
        async with Client(build_server(Policy(read_only=True))) as c:
            await c.call_tool("get_limits", {})
    anyio.run(go)
    assert any(e["action"] == "mcp.tool" and e["tool"] == "get_limits" and e["outcome"] == "ok" for e in entries("mcp"))
