"""Environment checks. Each returns a dict the UI renders: status ok | warn | fail | missing | na."""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

import httpx
import psutil

from .. import config

EXTRAS: dict[str, dict[str, Any]] = {
    "tabular": {"name": "Tabular models", "modules": ["sklearn", "lightgbm", "skl2onnx", "onnxmltools", "onnx", "onnxruntime"],
                "why": "scikit-learn, LightGBM and ONNX export. Needed for anomaly detection and tabular classification."},
    "text": {"name": "Transformers and PyTorch", "modules": ["torch", "transformers", "peft", "datasets", "accelerate", "onnx", "onnxruntime"],
             "why": "Fine-tune text classifiers and small language models, tiny autoencoders."},
    "mlx": {"name": "MLX LM (Apple Silicon)", "modules": ["mlx_lm"], "why": "Fastest LoRA training for 0.5B to 2B models on a Mac.", "platforms": ["darwin-arm64"]},
    "speech": {"name": "Cactus Needle and Whistle", "modules": ["needle"], "why": "Train Needle 3 (tool calls, extraction) and Whistle (speech to text)."},
    "coreml": {"name": "Core ML tools", "modules": ["coremltools"], "why": "Export to Core ML for iOS and macOS apps.", "platforms": ["darwin"]},
}


def _c(id: str, name: str, status: str, detail: str = "", fix: dict[str, Any] | None = None, group: str = "system") -> dict[str, Any]:
    return {"id": id, "name": name, "status": status, "detail": detail, "fix": fix, "group": group}


def _host_matches(platforms: list[str] | None) -> bool:
    if not platforms:
        return True
    tag = f"{sys.platform}-{platform.machine().lower()}"
    return any(tag.startswith(p) or p.startswith(sys.platform) and (len(p.split("-")) == 1 or p.split("-")[1] in tag) for p in platforms)


def _run(cmd: list[str], timeout: float = 8) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def python_check() -> dict[str, Any]:
    v = sys.version_info
    ok = (3, 11) <= (v.major, v.minor) < (3, 14)
    return _c("python", "Python", "ok" if ok else "warn", f"{platform.python_version()} at {sys.executable}",
              None if ok else {"type": "info", "label": "Use Python 3.11 to 3.13. With uv: uv python install 3.12"})


def uv_check() -> dict[str, Any]:
    p = shutil.which("uv")
    if p:
        return _c("uv", "uv (installs extras)", "ok", p)
    cmd = 'powershell -c "irm https://astral.sh/uv/install.ps1 | iex"' if sys.platform == "win32" else "curl -LsSf https://astral.sh/uv/install.sh | sh"
    return _c("uv", "uv (installs extras)", "warn", "not found. pip is used as a fallback.", {"type": "command", "command": cmd, "label": "Install uv"})


def disk_check() -> dict[str, Any]:
    free = shutil.disk_usage(config.home()).free / 1e9
    st = "ok" if free >= 15 else "warn" if free >= 3 else "fail"
    return _c("disk", "Free disk space", st, f"{free:.0f} GB free where models and jobs are stored ({config.home()})")


def ram_check() -> dict[str, Any]:
    gb = psutil.virtual_memory().total / 2**30
    st = "ok" if gb >= 16 else "warn" if gb >= 8 else "fail"
    return _c("ram", "Memory", st, f"{gb:.0f} GB. Training a 0.6B model with LoRA needs about 6 to 8 GB; 1.7B needs 16 GB or more.")


def accelerator_check() -> dict[str, Any]:
    if sys.platform == "darwin" and platform.machine().lower() == "arm64":
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])[1] or "Apple Silicon"
        return _c("accelerator", "Accelerator", "ok", f"{chip}: Metal (PyTorch MPS, MLX)")
    code, out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if code == 0 and out:
        extra = ""
        if sys.platform == "win32":
            extra = " On native Windows, Needle and Whistle (JAX) train on CPU only; use WSL2 to train them on this GPU."
        return _c("accelerator", "Accelerator", "ok", f"NVIDIA {out.splitlines()[0]}.{extra}")
    return _c("accelerator", "Accelerator", "warn", "no GPU found. Training runs on CPU, which is fine for tabular models and models under 200M parameters.")


def extra_checks() -> list[dict[str, Any]]:
    out = []
    for key, e in EXTRAS.items():
        if not _host_matches(e.get("platforms")):
            out.append(_c(f"extra-{key}", e["name"], "na", f"not available on this computer ({platform.system()} {platform.machine()})", group="python"))
            continue
        missing = [m for m in e["modules"] if importlib.util.find_spec(m) is None]
        if not missing:
            out.append(_c(f"extra-{key}", e["name"], "ok", "installed", group="python"))
        else:
            out.append(_c(f"extra-{key}", e["name"], "missing", f"{e['why']} Missing: {', '.join(missing)}.",
                          {"type": "extra", "extra": key, "label": f"Install '{key}'"}, group="python"))
    return out


def needle_cli_check() -> dict[str, Any] | None:
    if importlib.util.find_spec("needle") is None:
        return None
    p = shutil.which("needle") or os.path.join(os.path.dirname(sys.executable), "needle")
    ok = os.path.exists(p) or os.path.exists(p + ".exe")
    return _c("needle-cli", "needle command", "ok" if ok else "warn", p if ok else "cactus-needle is installed but the `needle` command was not found next to Python", group="python")


def ollama_check() -> dict[str, Any]:
    url = config.load_settings()["ollama_url"].rstrip("/")
    try:
        r = httpx.get(url + "/api/tags", timeout=2)
        names = [m["name"] for m in r.json().get("models", [])]
        return _c("ollama", "Ollama (local LLM for synthetic data)", "ok", f"{url} · {len(names)} model(s): {', '.join(names[:6]) or 'none pulled yet'}", group="tools")
    except Exception:  # noqa: BLE001
        has = shutil.which("ollama")
        detail = "installed but not running. Start it with: ollama serve" if has else "not installed. Optional: you can use Claude, Gemini or OpenAI keys instead."
        return _c("ollama", "Ollama (local LLM for synthetic data)", "warn", detail,
                  None if has else {"type": "link", "url": "https://ollama.com/download", "label": "Get Ollama"}, group="tools")


def docker_check() -> dict[str, Any]:
    if not shutil.which("docker"):
        return _c("docker", "Docker (optional)", "warn", "not installed. Only needed to build container exports locally.",
                  {"type": "link", "url": "https://docs.docker.com/get-started/get-docker/", "label": "Get Docker"}, group="tools")
    code, out = _run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=6)
    return _c("docker", "Docker (optional)", "ok" if code == 0 else "warn", f"server {out}" if code == 0 else "installed, but the daemon is not running", group="tools")


def ffmpeg_check() -> dict[str, Any]:
    p = shutil.which("ffmpeg")
    cmd = {"win32": "winget install Gyan.FFmpeg", "darwin": "brew install ffmpeg"}.get(sys.platform, "sudo apt install ffmpeg")
    return _c("ffmpeg", "ffmpeg (audio datasets)", "ok" if p else "warn", p or "not found. Only needed to convert audio for speech datasets.",
              None if p else {"type": "command", "command": cmd, "label": "Install ffmpeg"}, group="tools")


def git_check() -> dict[str, Any]:
    p = shutil.which("git")
    return _c("git", "git", "ok" if p else "warn", p or "not found. Needed to fetch llama.cpp and Apple's Core AI recipes.", group="tools")


def llamacpp_check() -> dict[str, Any]:
    script = config.tools_dir() / "llama.cpp" / "convert_hf_to_gguf.py"
    if script.exists():
        return _c("llamacpp", "llama.cpp converter (GGUF export)", "ok", str(script), group="tools")
    return _c("llamacpp", "llama.cpp converter (GGUF export)", "missing", "Used to export fine-tuned language models as GGUF.",
              {"type": "action", "action": "llamacpp", "label": "Fetch llama.cpp"}, group="tools")


def coreai_check() -> dict[str, Any] | None:
    if sys.platform != "darwin":
        return None
    major = int((platform.mac_ver()[0] or "0").split(".")[0] or 0)
    xcode = _run(["xcode-select", "-p"])[0] == 0
    uv = bool(shutil.which("uv"))
    parts = [f"macOS {platform.mac_ver()[0]}" + ("" if major >= 27 else " (needs 27)"), "Xcode tools " + ("found" if xcode else "missing"), "uv " + ("found" if uv else "missing")]
    ok = major >= 27 and xcode and uv
    return _c("coreai", "Apple Core AI export", "ok" if ok else "warn", ", ".join(parts), group="tools")


def windows_checks() -> list[dict[str, Any]]:
    if sys.platform != "win32":
        return []
    out = []
    try:
        import winreg  # type: ignore
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            v, _ = winreg.QueryValueEx(k, "LongPathsEnabled")
        out.append(_c("win-longpaths", "Windows long paths", "ok" if v == 1 else "warn",
                      "enabled" if v == 1 else "disabled. Some model files have long names; enable LongPathsEnabled in the registry."))
    except OSError:
        pass
    out.append(_c("win-wsl", "WSL2", "ok" if shutil.which("wsl") else "warn",
                  "available. Recommended for training Needle and Whistle on an NVIDIA GPU." if shutil.which("wsl") else "not found. Optional, recommended for GPU training of Needle and Whistle."))
    return out


def hf_check() -> dict[str, Any]:
    token = config.hf_token()
    if not token:
        return _c("hf-token", "Hugging Face token", "warn", "not set. Public models download without it; gated models (Gemma) need one.",
                  {"type": "link", "url": "https://huggingface.co/settings/tokens", "label": "Create a token"}, group="accounts")
    try:
        from huggingface_hub import HfApi
        who = HfApi().whoami(token=token)
        return _c("hf-token", "Hugging Face token", "ok", f"signed in as {who.get('name')}", group="accounts")
    except Exception as e:  # noqa: BLE001
        return _c("hf-token", "Hugging Face token", "fail", f"token rejected or Hugging Face unreachable: {str(e)[:120]}", group="accounts")


def provider_checks() -> list[dict[str, Any]]:
    s = config.load_settings()
    out = []
    labels = {"anthropic": "Anthropic (Claude)", "gemini": "Google (Gemini)", "openai": "OpenAI", "openrouter": "OpenRouter"}
    for name in labels:
        have = bool(s.get(f"{name}_api_key"))
        out.append(_c(f"key-{name}", f"{labels[name]} API key", "ok" if have else "na",
                      "saved (use Test in Settings to verify)" if have else "not set (optional)", group="accounts"))
    return out


def run_all() -> list[dict[str, Any]]:
    checks = [python_check(), uv_check(), disk_check(), ram_check(), accelerator_check(), *extra_checks()]
    for c in (needle_cli_check(),):
        if c:
            checks.append(c)
    checks += [git_check(), ollama_check(), docker_check(), ffmpeg_check(), llamacpp_check()]
    c = coreai_check()
    if c:
        checks.append(c)
    checks += windows_checks()
    checks += [hf_check(), *provider_checks()]
    return checks


def summary(checks: list[dict[str, Any]]) -> dict[str, int]:
    out = {"ok": 0, "warn": 0, "fail": 0, "missing": 0, "na": 0}
    for c in checks:
        out[c["status"]] = out.get(c["status"], 0) + 1
    return out
