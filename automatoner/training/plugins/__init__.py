"""Trainer registry. Plugins are imported lazily so the API process never loads heavy libraries."""
from __future__ import annotations

import importlib
import importlib.util
from typing import Any

_MODULES = {
    "sklearn": "sklearn_plugin",
    "lightgbm": "lightgbm_plugin",
    "torch_tiny": "torch_tiny_plugin",
    "hf_text": "hf_text_plugin",
    "mlx_lm": "mlx_plugin",
    "needle": "needle_plugin",
    "whistle": "whistle_plugin",
}

# Static facts that must not need an import of the heavy module.
INFO: dict[str, dict[str, Any]] = {
    "sklearn": {"name": "scikit-learn", "extra": "tabular", "requires": ["sklearn", "skl2onnx", "onnx", "onnxruntime"], "hardware": "CPU"},
    "lightgbm": {"name": "LightGBM", "extra": "tabular", "requires": ["lightgbm", "onnxmltools", "onnx", "onnxruntime", "sklearn"], "hardware": "CPU"},
    "torch_tiny": {"name": "Tiny PyTorch models", "extra": "text", "requires": ["torch", "onnx", "onnxruntime", "sklearn"], "hardware": "CPU, CUDA or Apple MPS"},
    "hf_text": {"name": "Transformers + PEFT", "extra": "text", "requires": ["torch", "transformers", "peft"], "hardware": "CPU, CUDA or Apple MPS"},
    "mlx_lm": {"name": "MLX LM (Apple Silicon)", "extra": "mlx", "requires": ["mlx_lm"], "hardware": "Apple Silicon only", "platforms": ["darwin-arm64"]},
    "needle": {"name": "Cactus Needle", "extra": "speech", "requires": ["needle"], "hardware": "CPU, CUDA (Linux) or Metal (macOS)"},
    "whistle": {"name": "Cactus Whistle", "extra": "speech", "requires": ["needle"], "hardware": "CPU, CUDA (Linux) or Metal (macOS)"},
}


def host_tag() -> str:
    import platform
    import sys
    return f"{sys.platform}-{platform.machine().lower()}".replace("win32", "windows").replace("amd64", "x86_64")


def available(plugin_id: str) -> dict[str, Any]:
    info = INFO[plugin_id]
    missing = [m for m in info["requires"] if importlib.util.find_spec(m) is None]
    plats = info.get("platforms")
    host_ok = True
    if plats:
        h = host_tag()
        host_ok = any(h.startswith(p.split("-")[0]) and (len(p.split("-")) < 2 or p.split("-")[1] in h) for p in plats)
    return {"id": plugin_id, **info, "missing": missing, "host_ok": host_ok, "ready": not missing and host_ok}


def load(plugin_id: str):
    if plugin_id not in _MODULES:
        raise KeyError(f"unknown trainer {plugin_id!r}")
    mod = importlib.import_module(f"{__name__}.{_MODULES[plugin_id]}")
    return mod.PLUGIN
