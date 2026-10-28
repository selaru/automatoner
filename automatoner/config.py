"""Paths and settings. User data lives outside the repo.

Default home: ~/.automatoner (Linux, macOS) or %LOCALAPPDATA%\\automatoner (Windows).
Override with AUTOMATONER_HOME.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import threading
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent


def home() -> Path:
    env = os.environ.get("AUTOMATONER_HOME")
    if env:
        p = Path(env).expanduser()
    elif sys.platform == "win32":
        p = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "automatoner"
    else:
        p = Path.home() / ".automatoner"
    p.mkdir(parents=True, exist_ok=True)
    return p


def subdir(name: str) -> Path:
    p = home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path: return subdir("models")
def datasets_dir() -> Path: return subdir("datasets")
def jobs_dir() -> Path: return subdir("jobs")
def exports_dir() -> Path: return subdir("exports")
def tools_dir() -> Path: return subdir("tools")
def tasks_dir() -> Path: return subdir("tasks")


# --- settings -----------------------------------------------------------------------------------

SECRET_KEYS = {"hf_token", "anthropic_api_key", "gemini_api_key", "openai_api_key", "openrouter_api_key"}
DEFAULTS: dict[str, Any] = {
    "hf_token": "",
    "anthropic_api_key": "",
    "gemini_api_key": "",
    "openai_api_key": "",
    "openrouter_api_key": "",
    "ollama_url": "http://127.0.0.1:11434",
    "default_provider": "ollama",
    "provider_models": {
        "ollama": "qwen3:8b",
        "anthropic": "claude-sonnet-5-5",
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-5-mini",
        "openrouter": "anthropic/claude-sonnet-5-5",
    },
    "enabled_extras": [],
    "local_only": False,
}
_lock = threading.Lock()


def _settings_path() -> Path:
    return home() / "settings.json"


def load_settings() -> dict[str, Any]:
    with _lock:
        data: dict[str, Any] = {}
        p = _settings_path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
    merged = {**DEFAULTS, **data}
    merged["provider_models"] = {**DEFAULTS["provider_models"], **data.get("provider_models", {})}
    return merged


def local_only_locked() -> bool:
    """True when an administrator pinned the policy with AUTOMATONER_LOCAL_ONLY, so the UI cannot change it."""
    return os.environ.get("AUTOMATONER_LOCAL_ONLY", "").strip() != ""


def local_only() -> bool:
    """Local-only mode: synthetic data may only be written by a local model (Ollama on this network). Nothing goes to cloud LLMs."""
    env = os.environ.get("AUTOMATONER_LOCAL_ONLY", "").strip().lower()
    if env:
        return env in ("1", "true", "yes", "on")
    return bool(load_settings().get("local_only"))


def save_settings(update: dict[str, Any]) -> dict[str, Any]:
    current = load_settings()
    if "local_only" in update and local_only_locked() and bool(update["local_only"]) != local_only():
        raise ValueError("local-only mode is fixed by the AUTOMATONER_LOCAL_ONLY environment variable on this server")
    changed: list[str] = []
    for k, v in update.items():
        if k not in DEFAULTS:
            continue
        if k in SECRET_KEYS and v is None:
            continue
        if k in SECRET_KEYS:
            changed.append(f"{k}: {'set' if v else 'cleared'}")
        elif current.get(k) != v:
            changed.append(f"{k}: changed")
        if k == "provider_models" and isinstance(v, dict):
            current["provider_models"] = {**current["provider_models"], **v}
        else:
            current[k] = v
    with _lock:
        p = _settings_path()
        p.write_text(json.dumps(current, indent=2), encoding="utf-8")
        if sys.platform != "win32":
            os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    if changed:
        from . import audit
        audit.record("settings.change", changes=changed)
    return current


def public_settings() -> dict[str, Any]:
    """Settings safe to send to the browser: secrets become booleans."""
    s = load_settings()
    out = {k: v for k, v in s.items() if k not in SECRET_KEYS}
    out["local_only"], out["local_only_locked"] = local_only(), local_only_locked()
    for k in SECRET_KEYS:
        out[f"{k}_set"] = bool(s.get(k))
    return out


def hf_token() -> str | None:
    """Token from settings, else the standard environment/CLI login. Passed per call, never exported."""
    t = load_settings().get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return t or None
