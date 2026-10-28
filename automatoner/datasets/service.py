"""Dataset lifecycle: import, synthesize, validate, split, preview, delete."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .. import config, tasks, util
from ..store import db
from . import schemas, validators
from .importers import files as ifiles
from .importers import hf as ihf
from .providers import get_provider
from .synth import llm as synth_llm
from .synth import tabular as synth_tabular


def dataset_dir(ds_id: str) -> Path:
    return util.safe_child(config.datasets_dir(), ds_id)


# --- core: rows or DataFrame -> validated, split dataset ------------------------------------------

def _finish(ds_id: str, *, name: str, category: str, source: str, rows: list[dict[str, Any]] | None = None,
            df: pd.DataFrame | None = None, options: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    options = options or {}
    d = dataset_dir(ds_id)
    d.mkdir(parents=True, exist_ok=True)
    valid_frac = float(options.get("valid", 0.1))
    test_frac = float(options.get("test", 0.1))
    seed = int(options.get("seed", 42))
    meta: dict[str, Any] = {"id": ds_id, "name": name, "category": category, "source": source, "status": "ready",
                            "created": util.now_iso(), "options": options, **(extra or {})}

    if category in schemas.TABULAR:
        assert df is not None
        rep = schemas.Report()
        label = options.get("label_column")
        if not label:
            label = next((c for c in df.columns if str(c).lower() in schemas.LABEL_NAMES), None)
        if category == "tabular_classification" and (not label or label not in df.columns):
            rep.errors.append("a label column is required; set label_column")
        if label and label not in df.columns:
            rep.errors.append(f"label column {label!r} not found")
            label = None
        before = len(df)
        df = df.dropna(axis=0, how="all").reset_index(drop=True)
        features = [c for c in df.columns if c != label]
        numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
        categorical = [c for c in features if c not in numeric and df[c].nunique() <= 50]
        ignored = [c for c in features if c not in numeric and c not in categorical]
        if ignored:
            rep.warnings.append(f"ignored high-cardinality text columns: {ignored}")
        if not numeric and not categorical:
            rep.errors.append("no usable feature columns")
        if len(df) < 50:
            rep.errors.append(f"only {len(df)} rows; need at least 50")
        if before != len(df):
            rep.dropped = before - len(df)
        if numeric:
            nan_rows = int(df[numeric].isna().any(axis=1).sum())
            if nan_rows:
                rep.warnings.append(f"{nan_rows} rows have missing numeric values; they are filled with the median at training time")
        if not rep.errors:
            strata = [str(v) for v in df[label]] if label else ["x"] * len(df)
            tr, va, te = validators.split_indices(strata, valid_frac, test_frac, seed)
            for part, idx in (("train", tr), ("valid", va), ("test", te)):
                df.iloc[idx].to_csv(d / f"{part}.csv", index=False)
            df.to_csv(d / "all.csv", index=False)
            meta["splits"] = {"train": len(tr), "valid": len(va), "test": len(te)}
        meta.update({"rows": len(df), "columns": list(df.columns), "label_column": label,
                     "numeric_columns": numeric, "categorical_columns": categorical,
                     "format": "csv", "positive_rate": float(pd.Series(df[label]).astype(float).mean()) if label and pd.api.types.is_numeric_dtype(df[label]) else None})
    else:
        assert rows is not None
        require_json = bool(options.get("require_json", category == "text_to_json"))
        cleaned, rep = validators.clean_rows(category, rows, base_dir=d, require_json=require_json)
        if not rep.errors:
            strata = [schemas.stratum(category, r) for r in cleaned]
            tr, va, te = validators.split_indices(strata, valid_frac, test_frac, seed)
            for part, idx in (("train", tr), ("valid", va), ("test", te)):
                validators.write_jsonl(d / f"{part}.jsonl", [cleaned[i] for i in idx])
            validators.write_jsonl(d / "all.jsonl", cleaned)
            meta["splits"] = {"train": len(tr), "valid": len(va), "test": len(te)}
        meta.update({"rows": len(cleaned), "format": "jsonl"})
        if category == "text_classification":
            meta["labels"] = sorted({str(r["label"]) for r in cleaned})

    meta["report"] = rep.public()
    if rep.errors:
        meta["status"] = "invalid"
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    db.put("datasets", ds_id, meta)
    return meta


# --- imports -------------------------------------------------------------------------------------

def import_upload(name: str, category: str, uploads: list[tuple[str, bytes]], options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = options or {}
    ds_id = util.new_id("ds-")
    d = dataset_dir(ds_id)
    d.mkdir(parents=True, exist_ok=True)
    try:
        table_file: Path | None = None
        for fname, data in uploads:
            safe = Path(fname).name
            if safe.lower().endswith(".zip"):
                extracted = ifiles.safe_extract_zip(data, d)
                for p in extracted:
                    if p.suffix.lower() in (".jsonl", ".ndjson", ".csv", ".tsv", ".parquet", ".json") and table_file is None:
                        table_file = p
            else:
                (d / safe).write_bytes(data)
                if Path(safe).suffix.lower() in (".jsonl", ".ndjson", ".csv", ".tsv", ".parquet", ".json") and table_file is None:
                    table_file = d / safe
        if table_file is None:
            raise ValueError("no data file found; upload a CSV, TSV, Parquet, JSON or JSONL file (or a zip containing one)")
        if category in schemas.TABULAR:
            df = ifiles.read_table(table_file)
            return _finish(ds_id, name=name, category=category, source="upload", df=df, options=options)
        rows = ifiles.read_records(table_file)
        rows = _map_columns(category, rows, options)
        # speech: audio paths are relative to the table file's folder; make them relative to the dataset dir
        if category == "speech":
            rel = table_file.resolve().parent.relative_to(d.resolve())
            for r in rows:
                if r.get("audio"):
                    r["audio"] = (rel / r["audio"]).as_posix()
        return _finish(ds_id, name=name, category=category, source="upload", rows=rows, options=options)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


def _map_columns(category: str, rows: list[dict[str, Any]], options: dict[str, Any]) -> list[dict[str, Any]]:
    m = options.get("columns") or {}
    if not m:
        return rows
    out = []
    for r in rows:
        nr = dict(r)
        for target, source in m.items():
            if source in r:
                nr[target] = r[source]
        out.append(nr)
    return out


def import_hf(name: str, category: str, repo: str, *, split: str = "train", subset: str | None = None,
              max_rows: int = 50000, options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = options or {}
    df = ihf.fetch(repo, split=split, subset=subset, max_rows=max_rows)
    ds_id = util.new_id("ds-")
    if category in schemas.TABULAR:
        return _finish(ds_id, name=name, category=category, source="hf", df=df, options=options, extra={"repo": repo})
    rows = json.loads(df.to_json(orient="records", force_ascii=False))
    rows = _map_columns(category, rows, options)
    return _finish(ds_id, name=name, category=category, source="hf", rows=rows, options=options, extra={"repo": repo})


# --- synthetic -----------------------------------------------------------------------------------

def start_synthetic(name: str, category: str, spec: dict[str, Any], *, provider: str | None = None, model: str | None = None,
                    options: dict[str, Any] | None = None) -> tuple[tasks.Task, str]:
    options = options or {}
    ds_id = util.new_id("ds-")
    d = dataset_dir(ds_id)
    d.mkdir(parents=True, exist_ok=True)
    db.put("datasets", ds_id, {"id": ds_id, "name": name, "category": category, "source": "synthetic", "status": "generating",
                               "created": util.now_iso(), "rows": 0, "spec": spec, "provider": provider})

    def run(task: tasks.Task) -> dict[str, Any]:
        meta = _run(task)
        return {"dataset_id": ds_id, "status": meta.get("status"), "rows": meta.get("rows")}

    def _run(task: tasks.Task) -> dict[str, Any]:
        try:
            if category in schemas.TABULAR and not spec.get("describe"):
                task.say("simulating")
                df = synth_tabular.simulate(spec)
                return _finish(ds_id, name=name, category=category, source="synthetic", df=df,
                               options={**options, "label_column": spec.get("label_column", "label")}, extra={"spec": spec})
            prov = get_provider(provider, model)
            if category in schemas.TABULAR:
                task.say(f"asking {prov.name}/{prov.model} to design the simulator")
                tspec = synth_llm.propose_tabular_spec(spec["describe"], prov, int(spec.get("rows", 5000)))
                task.say("simulating")
                df = synth_tabular.simulate(tspec)
                return _finish(ds_id, name=name, category=category, source="synthetic", df=df,
                               options={**options, "label_column": tspec.get("label_column", "label")}, extra={"spec": tspec})
            gen = synth_llm.GENERATORS.get(category)
            if gen is None:
                raise ValueError(f"no synthetic generator for category {category}")
            task.say(f"generating with {prov.name}/{prov.model}")
            rows = gen(spec, prov, task, d)
            if task.cancelled:
                db.update("datasets", ds_id, status="cancelled", rows=len(rows))
                return {"id": ds_id, "status": "cancelled", "rows": len(rows)}
            return _finish(ds_id, name=name, category=category, source="synthetic", rows=rows, options=options,
                           extra={"spec": {k: v for k, v in spec.items() if k != "seeds"}, "provider": prov.name, "model": prov.model})
        except Exception as e:
            db.update("datasets", ds_id, status="error", error=str(e))
            raise

    task = tasks.start("dataset", f"Generate {name}", run, id=f"gen-{ds_id}")
    return task, ds_id


# --- read ----------------------------------------------------------------------------------------

def list_datasets(category: str | None = None) -> list[dict[str, Any]]:
    out = db.list_("datasets")
    return [d for d in out if not category or d["category"] == category]


def get(ds_id: str) -> dict[str, Any] | None:
    return db.get("datasets", ds_id)


def preview(ds_id: str, limit: int = 20) -> dict[str, Any]:
    meta = get(ds_id)
    if meta is None:
        raise KeyError(ds_id)
    d = dataset_dir(ds_id)
    if meta.get("format") == "csv" and (d / "train.csv").exists():
        df = pd.read_csv(d / "train.csv", nrows=limit)
        return {"columns": list(df.columns), "rows": json.loads(df.to_json(orient="values"))}
    rows = validators.read_jsonl(d / "train.jsonl")[:limit]
    return {"columns": None, "rows": rows}


def delete(ds_id: str) -> None:
    shutil.rmtree(dataset_dir(ds_id), ignore_errors=True)
    db.delete("datasets", ds_id)
