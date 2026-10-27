"""Download a Hugging Face dataset's data files and read them with pandas (no `datasets` dependency)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ... import config

EXTS = (".parquet", ".csv", ".jsonl", ".json", ".tsv")


def fetch(repo: str, *, split: str = "train", subset: str | None = None, max_rows: int | None = 50000) -> pd.DataFrame:
    from huggingface_hub import HfApi, hf_hub_download

    token = config.hf_token()
    api = HfApi()
    files = [f for f in api.list_repo_files(repo, repo_type="dataset", token=token) if f.lower().endswith(EXTS)]
    if not files:
        raise ValueError(f"{repo} has no parquet, csv or json files that can be read directly")
    pick = [f for f in files if split in Path(f).stem.lower() or f"/{split}" in f.lower() or f.lower().startswith(split)]
    if subset:
        sub = [f for f in pick if subset.lower() in f.lower()]
        pick = sub or pick
    pick = pick or files
    frames = []
    total = 0
    for f in sorted(pick):
        local = hf_hub_download(repo, f, repo_type="dataset", token=token)
        ext = Path(local).suffix.lower()
        if ext == ".parquet":
            df = pd.read_parquet(local)
        elif ext == ".csv":
            df = pd.read_csv(local)
        elif ext == ".tsv":
            df = pd.read_csv(local, sep="\t")
        else:
            df = pd.read_json(local, lines=ext == ".jsonl")
        frames.append(df)
        total += len(df)
        if max_rows and total >= max_rows:
            break
    out = pd.concat(frames, ignore_index=True)
    return out.head(max_rows) if max_rows else out
