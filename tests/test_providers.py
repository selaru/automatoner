import httpx
import pytest

from automatoner import config
from automatoner.datasets import providers


class FakeResponse:
    def __init__(self, data, status=200):
        self._d, self.status_code, self.text = data, status, str(data)

    def json(self):
        return self._d


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers or {}, "json": json})
        return FakeResponse(RESPONSES[url.split("?")[0].split("/")[2]])
    monkeypatch.setattr(httpx, "post", fake_post)
    config.save_settings({"anthropic_api_key": "sk-ant-x", "gemini_api_key": "g-key", "openai_api_key": "sk-o", "openrouter_api_key": "or-key"})
    return calls


RESPONSES = {
    "api.anthropic.com": {"content": [{"type": "text", "text": "hello from claude"}]},
    "generativelanguage.googleapis.com": {"candidates": [{"content": {"parts": [{"text": "hello from gemini"}]}}]},
    "api.openai.com": {"choices": [{"message": {"content": "hello from openai"}}]},
    "openrouter.ai": {"choices": [{"message": {"content": "hello from router"}}]},
    "127.0.0.1:11434": {"message": {"content": "hello from ollama"}},
}


def test_anthropic_request_shape(captured):
    out = providers.get_provider("anthropic", "claude-x").complete("sys", "hi", 100, 0.5)
    c = captured[0]
    assert out == "hello from claude" and c["url"] == "https://api.anthropic.com/v1/messages"
    assert c["headers"]["x-api-key"] == "sk-ant-x" and c["headers"]["anthropic-version"]
    assert c["json"]["system"] == "sys" and c["json"]["messages"] == [{"role": "user", "content": "hi"}] and c["json"]["model"] == "claude-x"


def test_gemini_sends_key_in_header_not_url(captured):
    out = providers.get_provider("gemini", "gem-x").complete("sys", "hi", 100, 0.5)
    c = captured[0]
    assert out == "hello from gemini" and "g-key" not in c["url"] and c["headers"]["x-goog-api-key"] == "g-key"
    assert c["json"]["systemInstruction"]["parts"][0]["text"] == "sys"


def test_openai_uses_max_completion_tokens_and_no_temperature(captured):
    providers.get_provider("openai", "gpt-x").complete("sys", "hi", 100, 0.5)
    body = captured[0]["json"]
    assert body["max_completion_tokens"] == 100 and "max_tokens" not in body and "temperature" not in body
    assert captured[0]["headers"]["Authorization"] == "Bearer sk-o"


def test_openrouter_and_ollama(captured):
    assert providers.get_provider("openrouter", "m").complete("s", "u", 10, 0.2) == "hello from router"
    assert captured[0]["json"]["max_tokens"] == 10 and captured[0]["json"]["temperature"] == 0.2
    assert providers.get_provider("ollama", "qwen3:8b").complete("s", "u", 10, 0.2) == "hello from ollama"
    assert captured[1]["json"]["stream"] is False and captured[1]["json"]["think"] is False


def test_missing_key_gives_a_clear_error():
    config.save_settings({"anthropic_api_key": ""})
    with pytest.raises(providers.ProviderError, match="No API key for anthropic"):
        providers.get_provider("anthropic", "m")


def test_fatal_http_errors_are_not_retried(monkeypatch):
    n = {"calls": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        n["calls"] += 1
        return FakeResponse({"error": "bad key"}, 401)
    monkeypatch.setattr(httpx, "post", fake_post)
    config.save_settings({"openai_api_key": "k"})
    with pytest.raises(providers.ProviderError, match="401"):
        providers.ask(providers.get_provider("openai", "m"), "s", "u", retries=4)
    assert n["calls"] == 1


def test_anthropic_retries_without_temperature_when_model_rejects_it(monkeypatch):
    bodies = []

    def fake_post(url, headers=None, json=None, timeout=None):
        bodies.append(dict(json))
        if "temperature" in json:
            return FakeResponse({"error": {"message": "`temperature` is deprecated for this model."}}, 400)
        return FakeResponse(RESPONSES["api.anthropic.com"])
    monkeypatch.setattr(httpx, "post", fake_post)
    config.save_settings({"anthropic_api_key": "k"})
    out = providers.ask(providers.get_provider("anthropic", "claude-new"), "s", "u", temperature=0.8)
    assert out == "hello from claude" and len(bodies) == 2 and "temperature" not in bodies[1]
