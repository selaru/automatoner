"""Rubric evaluators for generative models: score model outputs on the test split and write a Markdown report."""
from __future__ import annotations

import json
import re
import statistics
from typing import Any, Callable

REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def get(name: str | None):
    return REGISTRY.get(name or "json_generic", REGISTRY["json_generic"])


def parse_json(text: str) -> Any | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_array(text: str) -> Any | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


@register("tool_calls")
def tool_calls(sample: dict[str, Any]) -> dict[str, float]:
    """Exact match of the called tool names and arguments, plus the weaker name-only match."""
    ref = parse_array(sample["reference"]) or []
    got = parse_array(sample["output"])
    s = {"valid_json": 0.0, "names_match": 0.0, "exact_match": 0.0}
    if not isinstance(got, list):
        return s
    s["valid_json"] = 1.0
    norm = lambda calls: [(c.get("name"), json.dumps(c.get("arguments", {}), sort_keys=True)) for c in calls if isinstance(c, dict)]  # noqa: E731
    s["names_match"] = float([n for n, _ in norm(got)] == [n for n, _ in norm(ref)])
    s["exact_match"] = float(norm(got) == norm(ref))
    return s


# --- generic: valid JSON and the same keys as the reference -------------------------------------------

@register("json_generic")
def json_generic(sample: dict[str, Any]) -> dict[str, float]:
    ref = parse_json(sample["reference"]) or {}
    got = parse_json(sample["output"])
    s = {"valid_json": 0.0, "exact_keys": 0.0, "key_types": 0.0}
    if got is None:
        return s
    s["valid_json"] = 1.0
    s["exact_keys"] = float(list(got) == list(ref))
    s["key_types"] = float(all(isinstance(got.get(k), type(v)) for k, v in ref.items()))
    return s


def aggregate(scores: list[dict[str, float]]) -> dict[str, float]:
    keys = scores[0].keys() if scores else []
    return {k: statistics.mean(s[k] for s in scores) for k in keys}


def report_markdown(title: str, n: int, seconds_per: float, overall: dict[str, float], by_group: dict[str, dict[str, dict[str, float]]],
                    samples: list[dict[str, Any]]) -> str:
    lines = [f"# Evaluation: {title}", "", f"{n} test examples, {seconds_per:.1f} s per example.", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k.replace('_', ' ')} | {v * 100:.1f}% |" for k, v in overall.items()]
    for dim, groups in by_group.items():
        lines += ["", f"## By {dim}", "", "| " + dim + " | n | " + " | ".join(overall) + " |", "|---|---|" + "---|" * len(overall)]
        for g, agg in sorted(groups.items()):
            lines.append(f"| {g} | {int(agg.pop('_n'))} | " + " | ".join(f"{agg[k] * 100:.0f}%" for k in overall) + " |")
    lines += ["", "## Samples", ""]
    for s in samples:
        lines += [f"**Input:** {s['input']}", "", "```json", s["output"].strip(), "```", ""]
    return "\n".join(lines)


def evaluate(name: str | None, rows: list[dict[str, Any]], generate: Callable[[str, str], str], *, limit: int, title: str,
             log: Callable[[str], None] = print) -> tuple[dict[str, float], str]:
    """rows: chat examples (with optional meta). generate(system, user) -> model output."""
    import time
    fn = get(name)
    rows = rows[:limit]
    scores, samples, metas = [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        msgs = r["messages"]
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next(m["content"] for m in msgs if m["role"] == "user")
        out = generate(system, user)
        s = fn({"input": user, "output": out, "reference": msgs[-1]["content"], "meta": r.get("meta")})
        scores.append(s); metas.append(r.get("meta") or {})
        if len(samples) < 4:
            samples.append({"input": user, "output": out})
        if (i + 1) % 10 == 0:
            log(f"evaluated {i + 1}/{len(rows)}")
    overall = aggregate(scores)
    by_group: dict[str, dict[str, dict[str, float]]] = {}
    dims = sorted({k for m in metas for k in m})
    for dim in dims:
        groups: dict[str, list[dict[str, float]]] = {}
        for s, m in zip(scores, metas):
            groups.setdefault(str(m.get(dim)), []).append(s)
        by_group[dim] = {g: {**aggregate(v), "_n": float(len(v))} for g, v in groups.items()}
    return overall, report_markdown(title, len(rows), (time.time() - t0) / max(1, len(rows)), overall, by_group, samples)
