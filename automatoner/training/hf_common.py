"""Helpers shared by the Transformers trainer and the MLX trainer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TOOL_SYSTEM = ("You turn user requests into tool calls. Reply with a JSON array of calls, each {{\"name\": ..., \"arguments\": {{...}}}}. "
               "Reply with [] when no tool fits.\nTools: {tools}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_chat(row: dict[str, Any]) -> dict[str, Any]:
    """Normalise any supported row to {"messages": [...], "meta": ...}."""
    if "messages" in row:
        msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
        return {"messages": msgs, "meta": row.get("meta")}
    tools = json.dumps(row["tools"], ensure_ascii=False, separators=(",", ":"))
    return {"messages": [{"role": "system", "content": TOOL_SYSTEM.format(tools=tools)}, {"role": "user", "content": row["query"]},
                         {"role": "assistant", "content": json.dumps(row["answers"], ensure_ascii=False)}], "meta": row.get("meta")}


def load_chat_splits(dataset_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {p: [to_chat(r) for r in read_jsonl(dataset_dir / f"{p}.jsonl")] for p in ("train", "valid", "test")}


def evaluator_for(category: str, cfg: dict[str, Any]) -> str:
    if cfg.get("evaluator"):
        return cfg["evaluator"]
    return "tool_calls" if category in ("tool_calling", "extraction") else "json_generic"
