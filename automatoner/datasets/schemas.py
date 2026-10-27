"""Per-category dataset shape and validation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

TABULAR = {"tabular_anomaly", "tabular_classification"}
CHAT = {"text_to_json"}
NEEDLE = {"tool_calling", "extraction"}
TEXTCLS = {"text_classification"}

LABEL_NAMES = ("label", "is_anomaly", "anomaly", "target", "y", "class")


def file_format(category: str) -> str:
    return "csv" if category in TABULAR else "jsonl"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dropped: int = 0

    def public(self) -> dict[str, Any]:
        return {"errors": self.errors, "warnings": self.warnings, "dropped": self.dropped}


def _chat_problem(row: dict[str, Any], require_json: bool) -> str | None:
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return "needs a messages list"
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
        return f"roles must be [system,] user, assistant, got {roles}"
    if any(not str(m.get("content", "")).strip() for m in msgs):
        return "empty message"
    if require_json:
        try:
            json.loads(msgs[-1]["content"])
        except (json.JSONDecodeError, TypeError):
            return "assistant message is not valid JSON"
    return None


def _needle_problem(row: dict[str, Any]) -> str | None:
    if "messages" in row:
        return _chat_problem(row, require_json=False)
    if not str(row.get("query", "")).strip():
        return "missing query"
    tools = row.get("tools")
    if not isinstance(tools, list) or not tools:
        return "missing tools"
    answers = row.get("answers")
    if not isinstance(answers, list):
        return "answers must be a list (empty for off-topic examples)"
    names = {t.get("name") for t in tools if isinstance(t, dict)}
    for a in answers:
        if not isinstance(a, dict) or a.get("name") not in names:
            return f"answer calls unknown tool {a.get('name') if isinstance(a, dict) else a!r}"
        if not isinstance(a.get("arguments", {}), dict):
            return "arguments must be an object"
    return None


def row_problem(category: str, row: dict[str, Any], *, require_json: bool = True, base_dir: Any = None) -> str | None:
    if category in TEXTCLS:
        if not str(row.get("text", "")).strip():
            return "missing text"
        if row.get("label") in (None, ""):
            return "missing label"
    elif category in CHAT:
        return _chat_problem(row, require_json)
    elif category in NEEDLE:
        return _needle_problem(row)
    elif category == "speech":
        if not str(row.get("text", "")).strip():
            return "missing transcript"
        a = str(row.get("audio", ""))
        if not a:
            return "missing audio path"
        if base_dir is not None and not (base_dir / a).exists():
            return f"audio file not found: {a}"
    return None


def dedupe_key(category: str, row: dict[str, Any]) -> str:
    if category in TEXTCLS:
        return normalize_text(row.get("text", ""))
    if category in CHAT or ("messages" in row):
        msgs = row.get("messages", [])
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        return normalize_text(user)
    if category in NEEDLE:
        return normalize_text(row.get("query", ""))
    if category == "speech":
        return str(row.get("audio", ""))
    return json.dumps(row, sort_keys=True)


def stratum(category: str, row: dict[str, Any]) -> str:
    if category in TEXTCLS:
        return str(row.get("label"))
    meta = row.get("meta")
    if isinstance(meta, dict) and meta:
        return "|".join(str(meta[k]) for k in sorted(meta))
    if category in NEEDLE:
        return "offtopic" if not row.get("answers") else "call"
    if category == "speech":
        return str(row.get("language", ""))
    return ""
