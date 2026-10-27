"""Validate rows, drop duplicates, split train/valid/test, write files."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import schemas
from .schemas import Report


def clean_rows(category: str, rows: list[dict[str, Any]], *, base_dir: Path | None = None, require_json: bool = True) -> tuple[list[dict[str, Any]], Report]:
    rep = Report()
    seen: set[str] = set()
    out = []
    reasons: dict[str, int] = defaultdict(int)
    for r in rows:
        problem = schemas.row_problem(category, r, require_json=require_json, base_dir=base_dir)
        if problem:
            reasons[problem] += 1
            rep.dropped += 1
            continue
        key = schemas.dedupe_key(category, r)
        if key in seen:
            reasons["duplicate"] += 1
            rep.dropped += 1
            continue
        seen.add(key)
        out.append(r)
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        rep.warnings.append(f"dropped {n} row(s): {reason}")
    if category == "text_classification":
        labels = {str(r["label"]) for r in out}
        if len(labels) < 2:
            rep.errors.append("text classification needs at least two distinct labels")
    if len(out) < 10:
        rep.errors.append(f"only {len(out)} usable rows; need at least 10")
    return out, rep


def split_indices(strata: list[str], valid: float, test: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Stratified split. Small strata still put at least one row in valid and test when possible."""
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(strata):
        groups[s].append(i)
    tr: list[int] = []
    va: list[int] = []
    te: list[int] = []
    for _, idx in sorted(groups.items()):
        rng.shuffle(idx)
        n = len(idx)
        n_te = max(1, round(n * test)) if n >= 3 and test > 0 else 0
        n_va = max(1, round(n * valid)) if n >= 3 and valid > 0 else 0
        te += idx[:n_te]
        va += idx[n_te:n_te + n_va]
        tr += idx[n_te + n_va:]
    rng.shuffle(tr)
    return tr, va, te


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
