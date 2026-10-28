"""One interface for every LLM that can write synthetic data. Plain HTTP, no vendor SDKs."""
from __future__ import annotations

import ipaddress
import json
import re
import time
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from ... import audit, config

PROVIDERS = ["ollama", "anthropic", "gemini", "openai", "openrouter"]


class ProviderError(RuntimeError):
    pass


@dataclass
class Provider:
    name: str
    model: str
    _complete: Callable[[str, str, int, float], str]

    def complete(self, system: str, user: str, max_tokens: int = 4000, temperature: float = 0.8) -> str:
        return self._complete(system, user, max_tokens, temperature)


def _post(url: str, *, headers: dict[str, str] | None = None, json_body: dict[str, Any], timeout: float = 180) -> dict[str, Any]:
    try:
        r = httpx.post(url, headers=headers, json=json_body, timeout=timeout)
    except httpx.HTTPError as e:
        raise ProviderError(f"cannot reach {url.split('?')[0]}: {e}") from e
    if r.status_code >= 400:
        raise ProviderError(f"HTTP {r.status_code} from {url.split('?')[0]}: {r.text[:300]}")
    return r.json()


def _ollama(url: str, model: str) -> Callable[[str, str, int, float], str]:
    def call(system: str, user: str, max_tokens: int, temperature: float) -> str:
        body = {"model": model, "stream": False, "think": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temperature, "num_predict": max_tokens}}
        data = _post(url.rstrip("/") + "/api/chat", json_body=body, timeout=600)
        return data["message"]["content"]
    return call


def _anthropic(key: str, model: str) -> Callable[[str, str, int, float], str]:
    def call(system: str, user: str, max_tokens: int, temperature: float) -> str:
        body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "system": system, "messages": [{"role": "user", "content": user}]}
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        try:
            data = _post("https://api.anthropic.com/v1/messages", headers=headers, json_body={**body, "temperature": min(temperature, 1.0)})
        except ProviderError as e:
            # Newer Claude models reject sampling parameters. Retry once without it.
            if "temperature" not in str(e).lower():
                raise
            data = _post("https://api.anthropic.com/v1/messages", headers=headers, json_body=body)
        return "".join(b.get("text", "") for b in data.get("content", []))
    return call


def _gemini(key: str, model: str) -> Callable[[str, str, int, float], str]:
    def call(system: str, user: str, max_tokens: int, temperature: float) -> str:
        data = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                     headers={"x-goog-api-key": key},
                     json_body={"systemInstruction": {"parts": [{"text": system}]},
                                "contents": [{"role": "user", "parts": [{"text": user}]}],
                                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}})
        cands = data.get("candidates") or []
        if not cands:
            raise ProviderError(f"Gemini returned no candidates: {json.dumps(data)[:300]}")
        return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
    return call


def _openai_like(base: str, key: str, model: str, *, token_param: str, send_temperature: bool,
                 extra_headers: dict[str, str] | None = None) -> Callable[[str, str, int, float], str]:
    def call(system: str, user: str, max_tokens: int, temperature: float) -> str:
        body: dict[str, Any] = {"model": model, token_param: max_tokens,
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if send_temperature:
            body["temperature"] = temperature
        data = _post(base.rstrip("/") + "/chat/completions",
                     headers={"Authorization": f"Bearer {key}", **(extra_headers or {})}, json_body=body)
        return data["choices"][0]["message"]["content"] or ""
    return call


_MOCKS: dict[str, Callable[[str, str, int, float], str]] = {}


def register_mock(name: str, fn: Callable[[str, str, int, float], str]) -> None:
    """Tests register scripted providers here."""
    _MOCKS[name] = fn


def internal_url(url: str) -> bool:
    """Is this server on this computer or the company network (loopback, private address, or a single-label or .local/.internal name)?"""
    host = (urlparse(url if "://" in url else f"http://{url}").hostname or "").lower()
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private or ip.is_link_local
    except ValueError:
        return host == "localhost" or "." not in host or host.endswith((".local", ".internal", ".lan", ".home.arpa"))


def check_allowed(name: str) -> None:
    """Local-only mode: refuse anything that would send data to a cloud LLM."""
    if not config.local_only() or name in _MOCKS:
        return
    why = None
    if name != "ollama":
        why = f"{name} is a cloud service"
    elif not internal_url(config.load_settings()["ollama_url"]):
        why = "the Ollama URL is not on this computer or a private network"
    if why:
        audit.record("llm.blocked", provider=name, reason=why)
        raise ProviderError(f"Local-only mode is on, so data is not sent to {name} ({why}). Use Ollama on this computer or your private network, "
                            "or turn local-only mode off in Settings (an administrator may have fixed it with AUTOMATONER_LOCAL_ONLY).")


def get_provider(name: str | None = None, model: str | None = None) -> Provider:
    s = config.load_settings()
    name = name or s["default_provider"]
    check_allowed(name)
    model = model or s["provider_models"].get(name, "")
    if name in _MOCKS:
        return Provider(name, model or "mock", _MOCKS[name])
    if name == "ollama":
        return Provider(name, model, _ollama(s["ollama_url"], model))
    key = s.get(f"{name}_api_key")
    if name in PROVIDERS and not key:
        raise ProviderError(f"No API key for {name}. Add it in Settings.")
    if name == "anthropic":
        return Provider(name, model, _anthropic(key, model))
    if name == "gemini":
        return Provider(name, model, _gemini(key, model))
    if name == "openai":
        return Provider(name, model, _openai_like("https://api.openai.com/v1", key, model, token_param="max_completion_tokens", send_temperature=False))
    if name == "openrouter":
        return Provider(name, model, _openai_like("https://openrouter.ai/api/v1", key, model, token_param="max_tokens", send_temperature=True))
    raise ProviderError(f"unknown provider {name!r}")


def ollama_models() -> list[str]:
    s = config.load_settings()
    try:
        r = httpx.get(s["ollama_url"].rstrip("/") + "/api/tags", timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def ask(provider: Provider, system: str, user: str, *, max_tokens: int = 4000, temperature: float = 0.8, retries: int = 4) -> str:
    """One logical request to a model, with retries. The audit log gets sizes and outcome, never the text."""
    check_allowed(provider.name)
    started = time.time()
    attempts = 0
    outcome = "ok"
    reply_chars = 0
    try:
        delay = 2.0
        for attempt in range(retries):
            attempts += 1
            try:
                reply = provider.complete(system, user, max_tokens, temperature)
                reply_chars = len(reply)
                return reply
            except ProviderError as e:
                msg = str(e)
                fatal = any(code in msg for code in ("HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404", "No API key"))
                if fatal or attempt == retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        raise ProviderError("unreachable")
    except ProviderError:
        outcome = "error"
        raise
    finally:
        audit.record("llm.call", provider=provider.name, model=provider.model, sent_chars=len(system) + len(user), received_chars=reply_chars,
                     attempts=attempts, seconds=round(time.time() - started, 2), outcome=outcome)


def extract_json(text: str) -> Any:
    """First JSON array or object in a model reply, tolerant of fences, think tags and chatter."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in reply")
    start = min(starts)
    closer = "]" if text[start] == "[" else "}"
    end = text.rfind(closer)
    if end < start:
        raise ValueError("unterminated JSON in reply")
    return json.loads(text[start:end + 1])
