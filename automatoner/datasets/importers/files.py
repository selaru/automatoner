"""Read user files into row dicts / DataFrames."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported table format: {ext}")


def read_records(path: Path) -> list[dict[str, Any]]:
    ext = path.suffix.lower()
    if ext in (".jsonl", ".ndjson"):
        rows = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name} line {i}: invalid JSON ({e.msg})") from e
        return rows
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data") or data.get("rows") or [data]
        return list(data)
    df = read_table(path)
    return json.loads(df.to_json(orient="records", force_ascii=False))


def safe_extract_zip(data: bytes, dest: Path) -> list[Path]:
    """Extract a zip, refusing paths that escape dest (zip slip)."""
    dest = dest.resolve()
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"unsafe path in zip: {info.filename}")
            if info.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())
            out.append(target)
    return out
