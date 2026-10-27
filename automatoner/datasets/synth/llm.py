"""LLM-backed synthetic data generators.

Shared behaviour: optional seeds of real examples set the style, generation is balanced over user-defined
cells, resumable (finished rows are appended to a raw file and reused), de-duplicated, parallel, retried.
"""
from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ... import tasks
from ..providers import Provider, ask, extract_json
from ..schemas import normalize_text
from ..validators import append_jsonl, read_jsonl


def _run_parallel(jobs: list[Any], fn, task: tasks.Task, workers: int, total: int, done_start: int = 0) -> None:
    lock = threading.Lock()
    done = {"n": done_start}
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(fn, j) for j in jobs]
    try:
        for f in as_completed(futures):
            if task.cancelled:
                break
            try:
                got = f.result()
            except Exception as e:  # noqa: BLE001
                task.say(f"batch failed: {e}")
                got = 0
            with lock:
                done["n"] += got or 0
                task.set(min(1.0, done["n"] / max(1, total)), f"{done['n']} of {total}")
    finally:
        if task.cancelled:
            for f in futures:
                f.cancel()
        pool.shutdown(wait=not task.cancelled, cancel_futures=True)


def _json_list(reply: str) -> list[Any]:
    data = extract_json(reply)
    if isinstance(data, dict):
        for k in ("examples", "items", "data", "rows"):
            if isinstance(data.get(k), list):
                return data[k]
        return [data]
    return data


# --- text classification -----------------------------------------------------------------------

def generate_text_classification(spec: dict[str, Any], provider: Provider, task: tasks.Task, workdir: Path) -> list[dict[str, Any]]:
    """spec: {labels: [{name, description}], per_label: 200, language: "English", style: "...", seeds: [{text,label}]}"""
    labels = spec["labels"]
    per_label = int(spec.get("per_label", 100))
    lang = spec.get("language", "English")
    style = spec.get("style", "short, realistic, varied wording")
    seeds = spec.get("seeds") or []
    raw = workdir / "raw.jsonl"
    rows = read_jsonl(raw)
    seen = {normalize_text(r["text"]) for r in rows}
    counts = {l["name"]: sum(1 for r in rows if r["label"] == l["name"]) for l in labels}
    batch = 20
    jobs = []
    for l in labels:
        need = per_label - counts[l["name"]]
        jobs += [l] * ((max(0, need) + batch - 1) // batch)
    lock = threading.Lock()
    system = "You write realistic training examples for a text classifier. Reply with a JSON array of strings and nothing else."

    def one(label: dict[str, Any]) -> int:
        shots = [s["text"] for s in seeds if s.get("label") == label["name"]][:5]
        user = (f"Label: {label['name']}\nMeaning: {label.get('description', label['name'])}\nLanguage: {lang}\nStyle: {style}\n"
                f"{'Real examples of this label:' + chr(10) + chr(10).join('- ' + s for s in shots) + chr(10) if shots else ''}"
                f"Write {batch} different examples of this label. Vary length, vocabulary and wording. Do not number them.")
        texts = [t for t in _json_list(ask(provider, system, user, max_tokens=3000)) if isinstance(t, str) and t.strip()]
        fresh = []
        with lock:
            for t in texts:
                k = normalize_text(t)
                if k not in seen and counts[label["name"]] + len(fresh) < per_label:
                    seen.add(k)
                    fresh.append({"text": t.strip(), "label": label["name"]})
            counts[label["name"]] += len(fresh)
            append_jsonl(raw, fresh)
            rows.extend(fresh)
        return len(fresh)

    _run_parallel(jobs, one, task, int(spec.get("workers", 4)), per_label * len(labels), done_start=len(rows))
    return rows


# --- tool calling / extraction (Needle JSONL) --------------------------------------------------

def generate_tool_calling(spec: dict[str, Any], provider: Provider, task: tasks.Task, workdir: Path) -> list[dict[str, Any]]:
    """spec: {tools: [{name, description, parameters:{...}}], total: 300, language: "English", off_topic_share: 0.125}"""
    tools = spec["tools"]
    total = int(spec.get("total", 200))
    lang = spec.get("language", "English")
    off = float(spec.get("off_topic_share", 0.125))
    raw = workdir / "raw.jsonl"
    rows = read_jsonl(raw)
    seen = {normalize_text(r["query"]) for r in rows}
    names = {t["name"] for t in tools}
    batch = 10
    n_jobs = max(0, (total - len(rows) + batch - 1) // batch)
    lock = threading.Lock()
    system = ("You write training data for a model that turns user requests into function calls. "
              "Reply with a JSON array and nothing else.")

    def one(_: int) -> int:
        n_off = max(1, round(batch * off)) if off > 0 else 0
        user = (f"Tools available:\n{json.dumps(tools, ensure_ascii=False)}\n\nLanguage of the requests: {lang}\n"
                f"Write {batch} examples. {batch - n_off} are realistic user requests that need one or more of these tools; "
                f"{n_off} are requests that no tool can do (empty answers). Each example is an object with: "
                '"query" (what the user says, natural and varied), '
                '"answers" (list of {"name": tool name, "arguments": {...}} in order; [] when no tool fits), '
                '"reasoning" (one short line linking each argument value to the words in the query). '
                "Argument values must come from the query. Use only the listed tool names and parameters.")
        got = _json_list(ask(provider, system, user, max_tokens=4000, temperature=0.9))
        fresh = []
        with lock:
            for g in got:
                if not isinstance(g, dict) or not isinstance(g.get("query"), str) or not isinstance(g.get("answers"), list):
                    continue
                if any(not isinstance(a, dict) or a.get("name") not in names for a in g["answers"]):
                    continue
                k = normalize_text(g["query"])
                if k in seen or len(rows) + len(fresh) >= total:
                    continue
                seen.add(k)
                fresh.append({"query": g["query"].strip(), "tools": tools, "answers": g["answers"],
                              "reasoning": g.get("reasoning", "")})
            append_jsonl(raw, fresh)
            rows.extend(fresh)
        return len(fresh)

    _run_parallel(list(range(n_jobs)), one, task, int(spec.get("workers", 4)), total, done_start=len(rows))
    return rows


# --- text to JSON, two stages (inputs from seeds, then targets from a rubric) --------------------

def generate_text_to_json(spec: dict[str, Any], provider: Provider, task: tasks.Task, workdir: Path) -> list[dict[str, Any]]:
    """spec: {
         total, system_prompt, rubric (markdown), seeds: [{text, lang?}],
         cells: {"lang": {"ro": {"desc": "Romanian", "share": .5}, ...}, "kind": {"software": {"desc": "...", "share": .25}, ...}},
         input_description: "a spoken note transcribed by speech recognition", output_keys: ["summary", ...] (optional)
       }"""
    total = int(spec.get("total", 200))
    system_prompt = spec["system_prompt"].strip()
    rubric = spec["rubric"]
    seeds = spec.get("seeds") or []
    cells = spec.get("cells") or {"all": {"all": {"desc": "any", "share": 1.0}}}
    input_desc = spec.get("input_description", "a short text")
    notes_file = workdir / "inputs.jsonl"
    cards_file = workdir / "raw.jsonl"

    dims = list(cells)
    combos: list[tuple[dict[str, str], float]] = [({}, 1.0)]
    for d in dims:
        combos = [({**c, d: v}, w * float(info.get("share", 1.0))) for c, w in combos for v, info in cells[d].items()]
    quota = [(c, max(1, round(total * w / sum(w2 for _, w2 in combos)))) for c, w in combos]

    # stage 1: inputs
    notes = read_jsonl(notes_file)
    seen = {normalize_text(n["text"]) for n in notes}
    lock = threading.Lock()

    def cell_key(c: dict[str, str]) -> str:
        return "|".join(f"{k}={c[k]}" for k in sorted(c))

    have: dict[str, int] = {}
    for n in notes:
        have[cell_key(n["meta"])] = have.get(cell_key(n["meta"]), 0) + 1
    batch = 8
    jobs: list[dict[str, str]] = []
    for c, q in quota:
        need = q - have.get(cell_key(c), 0)
        jobs += [c] * ((max(0, need) + batch - 1) // batch)
    sys1 = f"You write realistic examples of {input_desc}, for training data. Reply with a JSON array of strings and nothing else."

    def make_notes(c: dict[str, str]) -> int:
        desc = "; ".join(f"{d}: {cells[d][c[d]].get('desc', c[d])}" for d in dims)
        shots = random.sample(seeds, min(4, len(seeds))) if seeds else []
        user = (f"Constraints for this batch: {desc}\n"
                f"{'Real examples, copy their style and imperfections:' + chr(10) + chr(10).join('- ' + s['text'] for s in shots) + chr(10) if shots else ''}"
                f"Write {batch} new, clearly different examples. Vary length, topic and tone. Do not number them.")
        texts = [t for t in _json_list(ask(provider, sys1, user, max_tokens=3500, temperature=1.0)) if isinstance(t, str) and t.strip()]
        fresh = []
        with lock:
            for t in texts:
                k = normalize_text(t)
                if k in seen:
                    continue
                seen.add(k)
                fresh.append({"text": t.strip(), "meta": dict(c)})
            append_jsonl(notes_file, fresh)
            notes.extend(fresh)
        return len(fresh)

    task.say(f"stage 1: writing inputs ({len(jobs)} batches)")
    _run_parallel(jobs, make_notes, task, int(spec.get("workers", 4)), total * 2, done_start=len(notes))
    if task.cancelled:
        return read_jsonl(cards_file)

    # stage 2: targets
    done_inputs = {normalize_text(r["messages"][1]["content"]) for r in read_jsonl(cards_file)}
    todo = [n for n in notes if normalize_text(n["text"]) not in done_inputs][:max(0, total - len(done_inputs))]
    cards = read_jsonl(cards_file)
    sys2 = ("You produce the target output for a model that will be fine-tuned. Follow the rubric exactly. "
            "Reply with one JSON object and nothing else.\n\nRUBRIC\n" + rubric)
    keys = spec.get("output_keys")

    def make_card(n: dict[str, Any]) -> int:
        out = ask(provider, sys2, f"Input:\n{n['text']}", max_tokens=1500, temperature=0.4)
        obj = extract_json(out)
        if not isinstance(obj, dict) or (keys and list(obj) != list(keys)):
            return 0
        row = {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": n["text"]},
                            {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)}], "meta": n["meta"]}
        with lock:
            append_jsonl(cards_file, [row])
            cards.append(row)
        return 1

    task.say(f"stage 2: writing targets ({len(todo)} inputs)")
    _run_parallel(todo, make_card, task, int(spec.get("workers", 4)), total, done_start=len(cards))
    return cards


GENERATORS = {
    "text_classification": generate_text_classification,
    "tool_calling": generate_tool_calling,
    "extraction": generate_tool_calling,
    "text_to_json": generate_text_to_json,
}


# --- tabular spec proposal --------------------------------------------------------------------

def propose_tabular_spec(description: str, provider: Provider, rows: int = 5000) -> dict[str, Any]:
    from .tabular import KINDS, validate_spec
    system = "You design simulators for tabular monitoring data. Reply with one JSON object and nothing else."
    user = (f"Describe the normal behaviour and the anomalies of this system as a simulator spec.\n\nSystem: {description}\n\n"
            f"Spec format: {{\"rows\": {rows}, \"anomaly_rate\": 0.03, \"seed\": 1, \"label_column\": \"label\", \"columns\": [ ... ]}}.\n"
            f"Each column has name and kind, one of {sorted(KINDS)}. Parameters by kind: normal(mean,std,min,max), lognormal(mean_log,sigma), "
            "poisson(lam), uniform(low,high), uniform_int(low,high), seasonal(base,amplitude,period_column,period,noise), category(values,probs). "
            'Any column may have "anomaly": {"mult": x} and/or {"shift_std": k}; category columns use {"probs": [...]}. '
            "Use 4 to 8 columns with realistic magnitudes. Include a column for hour of day if traffic has a daily rhythm.")
    for _ in range(3):
        spec = extract_json(ask(provider, system, user, max_tokens=3000, temperature=0.3))
        try:
            validate_spec(spec)
            spec["rows"] = rows
            return spec
        except ValueError as e:
            user += f"\n\nYour last answer was invalid: {e}. Fix it."
    raise ValueError("model did not produce a valid simulator spec")
