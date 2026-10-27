"""Tabular simulator: sample a described normal regime and inject labelled anomalies.

Spec example:
{
  "rows": 5000, "anomaly_rate": 0.03, "seed": 7, "label_column": "label",
  "columns": [
    {"name": "hour", "kind": "uniform_int", "low": 0, "high": 23},
    {"name": "requests", "kind": "seasonal", "base": 200, "amplitude": 120, "period_column": "hour", "period": 24, "noise": 15,
       "anomaly": {"mult": 3.0}},
    {"name": "error_rate", "kind": "normal", "mean": 0.01, "std": 0.004, "min": 0, "max": 1, "anomaly": {"shift_std": 8}},
    {"name": "p95_ms", "kind": "lognormal", "mean_log": 4.6, "sigma": 0.2, "anomaly": {"mult": 2.5}},
    {"name": "event", "kind": "category", "values": ["a","b"], "probs": [0.9, 0.1], "anomaly": {"probs": [0.2, 0.8]}}
  ]
}
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

KINDS = {"normal", "lognormal", "poisson", "uniform", "uniform_int", "seasonal", "category"}


def validate_spec(spec: dict[str, Any]) -> None:
    cols = spec.get("columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError("spec needs a non-empty columns list")
    names = set()
    for c in cols:
        if c.get("kind") not in KINDS:
            raise ValueError(f"column {c.get('name')!r}: kind must be one of {sorted(KINDS)}")
        if not c.get("name") or c["name"] in names:
            raise ValueError("column names must be present and unique")
        names.add(c["name"])
        if c["kind"] == "category":
            n = len(c.get("values", []))
            if n < 2:
                raise ValueError(f"{c['name']}: category needs at least two values")
            for key in ("probs",):
                if c.get(key) and len(c[key]) != n:
                    raise ValueError(f"{c['name']}: probs length must match values")
                an = (c.get("anomaly") or {}).get(key)
                if an and len(an) != n:
                    raise ValueError(f"{c['name']}: anomaly probs length must match values")
    if not 0 < float(spec.get("anomaly_rate", 0.03)) < 0.5:
        raise ValueError("anomaly_rate must be between 0 and 0.5")
    if not 50 <= int(spec.get("rows", 5000)) <= 2_000_000:
        raise ValueError("rows must be between 50 and 2,000,000")


def _base(c: dict[str, Any], n: int, rng: np.random.Generator, frame: dict[str, np.ndarray]) -> np.ndarray:
    k = c["kind"]
    if k == "normal":
        x = rng.normal(c.get("mean", 0.0), c.get("std", 1.0), n)
    elif k == "lognormal":
        x = rng.lognormal(c.get("mean_log", 0.0), c.get("sigma", 0.5), n)
    elif k == "poisson":
        x = rng.poisson(c.get("lam", 5.0), n).astype(float)
    elif k == "uniform":
        x = rng.uniform(c.get("low", 0.0), c.get("high", 1.0), n)
    elif k == "uniform_int":
        x = rng.integers(c.get("low", 0), c.get("high", 10) + 1, n).astype(float)
    elif k == "seasonal":
        pc = c.get("period_column")
        phase = frame[pc].astype(float) if pc in frame else rng.uniform(0, c.get("period", 24), n)
        period = float(c.get("period", 24))
        x = c.get("base", 100.0) + c.get("amplitude", 50.0) * np.sin(2 * np.pi * phase / period) + rng.normal(0, c.get("noise", 5.0), n)
    else:
        raise ValueError(k)
    if "min" in c:
        x = np.maximum(x, c["min"])
    if "max" in c:
        x = np.minimum(x, c["max"])
    return x


def _spread(c: dict[str, Any], x: np.ndarray) -> float:
    k = c["kind"]
    if k == "normal":
        return float(c.get("std", 1.0))
    if k == "poisson":
        return float(np.sqrt(c.get("lam", 5.0)))
    return float(np.std(x)) or 1.0


def simulate(spec: dict[str, Any]) -> pd.DataFrame:
    validate_spec(spec)
    n = int(spec.get("rows", 5000))
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    rate = float(spec.get("anomaly_rate", 0.03))
    label_col = spec.get("label_column", "label")
    cols = spec["columns"]

    frame: dict[str, np.ndarray] = {}
    for c in cols:
        if c["kind"] == "category":
            frame[c["name"]] = rng.choice(c["values"], n, p=c.get("probs"))
        else:
            frame[c["name"]] = _base(c, n, rng, frame)

    is_anom = np.zeros(n, dtype=int)
    anom_idx = rng.choice(n, size=max(1, int(n * rate)), replace=False)
    is_anom[anom_idx] = 1
    affectable = [c for c in cols if c.get("anomaly")]
    if affectable:
        for i in anom_idx:
            k = int(rng.integers(1, len(affectable) + 1))
            for c in rng.choice(np.array(affectable, dtype=object), size=k, replace=False):
                a = c["anomaly"]
                name = c["name"]
                if c["kind"] == "category":
                    if a.get("probs"):
                        frame[name][i] = rng.choice(c["values"], p=a["probs"])
                    continue
                spread = _spread(c, frame[name])
                v = float(frame[name][i])
                if "mult" in a:
                    v *= float(a["mult"])
                if "shift_std" in a:
                    v += float(a["shift_std"]) * spread * float(rng.choice([-1, 1]) if a.get("two_sided") else 1)
                if "min" in c:
                    v = max(v, c["min"])
                if "max" in c:
                    v = min(v, c["max"])
                frame[name][i] = v
    df = pd.DataFrame(frame)
    for c in cols:
        if c["kind"] in ("poisson", "uniform_int"):
            df[c["name"]] = df[c["name"]].round().astype(int)
        elif c["kind"] != "category":
            df[c["name"]] = df[c["name"]].round(6)
    names = spec.get("label_names")
    if names:
        if len(names) != 2:
            raise ValueError("label_names must be [normal_name, anomaly_name]")
        df[label_col] = [names[int(v)] for v in is_anom]
    else:
        df[label_col] = is_anom
    return df
