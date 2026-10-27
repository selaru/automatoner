"""Exporter registry."""
from __future__ import annotations

import importlib

_MODULES = ["tabular_exporters", "llm_exporters", "needle_exporters", "coreml_exporter"]


def all_exporters():
    out = []
    for m in _MODULES:
        try:
            mod = importlib.import_module(f"{__name__}.{m}")
        except ModuleNotFoundError as e:
            if e.name and e.name.startswith(__name__):
                continue  # module not written yet / optional
            raise
        out.extend(mod.EXPORTERS)
    return out
