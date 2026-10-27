"""Cactus Needle 3 trainer: LoRA on the frozen base through the `needle` command, then `needle build` to a .cact."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import cactus_common as cc
from ..base import TrainContext, dir_size, field_, write_manifest
from ..hf_common import read_jsonl


class NeedlePlugin:
    id = "needle"
    requires = ["needle"]
    extra = "speech"
    hardware = "CPU, CUDA (Linux) or Metal (macOS); JAX on native Windows is CPU only"
    categories = ["tool_calling", "extraction"]

    def schema(self, category: str, entry: dict[str, Any]) -> list[dict[str, Any]]:
        return [field_("epochs", "Epochs", "int", 10, min=1, max=200, help="A few hundred examples want 10 to 30 epochs. Stop when validation loss rises."),
                field_("batch_size", "Batch size", "int", 16, min=1, max=256),
                field_("lr", "Learning rate", "float", 1e-3, min=1e-6, max=1e-1, help="Needle's own default is used when this is left at 0.", advanced=True),
                field_("lora_rank", "LoRA rank", "int", 16, min=4, max=128, help="32 doubles adapter capacity for argument-heavy tasks."),
                field_("lora_alpha", "LoRA alpha", "int", 32, min=1, max=256, advanced=True),
                field_("max_len", "Max tokens per example", "int", 1024, min=128, max=4096, advanced=True),
                field_("layers", "Layers in the exported model", "int", 20, min=2, max=20, help="Local training tunes the full 20 layers, so shallower exports lose accuracy (check the test score). Cactus's hosted fine-tune trains every depth."),
                field_("seed", "Random seed", "int", 42, advanced=True)]

    def run(self, ctx: TrainContext) -> dict[str, Any]:
        cfg = ctx.config
        ckpt = cc.snapshot_file(ctx.model_path, "checkpoints/needle3.safetensors")
        rows_tr, rows_va = read_jsonl(ctx.dataset_dir / "train.jsonl"), read_jsonl(ctx.dataset_dir / "valid.jsonl")
        bad = [r for r in rows_tr[:20] if "query" not in r and "tools" not in (r.get("messages") and r) and "tools" not in r]
        if bad:
            raise ValueError("Needle needs tool-calling rows: query, tools, answers (and optionally reasoning).")
        data = ctx.job_dir / "needle_train.jsonl"
        with data.open("w", encoding="utf-8") as f:
            for r in rows_tr + rows_va:
                f.write(json.dumps({k: v for k, v in r.items() if k != "meta"}, ensure_ascii=False) + "\n")
        val_split = round(len(rows_va) / max(1, len(rows_tr) + len(rows_va)), 4)
        out = ctx.out_dir
        adapter = out / "adapter.safetensors"
        cmd = [*cc.needle_exe(), "finetune", str(data), "--checkpoint", ckpt, "--epochs", str(int(cfg.get("epochs", 10))), "--batch-size", str(int(cfg.get("batch_size", 16))),
               "--lora-rank", str(int(cfg.get("lora_rank", 16))), "--lora-alpha", str(int(cfg.get("lora_alpha", 32))), "--max-len", str(int(cfg.get("max_len", 1024))),
               "--val-split", str(val_split), "--seed", str(int(cfg.get("seed", 42))), "--out", str(adapter), "--checkpoint-dir", str(ctx.job_dir / "ckpt")]
        if float(cfg.get("lr", 0) or 0) > 0:
            cmd += ["--lr", str(cfg["lr"])]
        ctx.log(f"{len(rows_tr) + len(rows_va)} examples, {val_split:.0%} held out for validation by Needle")
        lines = cc.run_cli(ctx, cmd)
        if not adapter.exists():
            raise RuntimeError("needle finetune finished without writing the adapter")
        metrics: dict[str, Any] = {}
        for ln in lines:
            m = cc.HELD_OUT.search(ln)
            if m:
                metrics["valid"] = {"exact_match": int(m.group(1)) / max(1, int(m.group(2))), "n": float(m.group(2))}

        layers = int(cfg.get("layers", 20))
        cact = out / f"needle3-{layers}l.cact"
        cc.run_cli(ctx, [*cc.needle_exe(), "build", ckpt, "--lora", str(adapter), "--layers", str(layers), "--out", str(cact)])
        eval_info = self._evaluate(ctx, cact, layers)
        if eval_info:
            metrics["test"] = eval_info
        write_manifest(ctx, {"trainer": self.id, "algorithm": "needle3_lora", "layers": layers,
                             "artifacts": {"cact": cact.name, "adapter": adapter.name, "checkpoint": "base"},
                             "metrics": metrics, "size_bytes": dir_size(out), "cact_bytes": cact.stat().st_size,
                             "base_checkpoint": ckpt, "inference": {"mode": "tool_calling"}})
        return metrics

    def _evaluate(self, ctx: TrainContext, cact: Path, layers: int) -> dict[str, float] | None:
        """Exact-match tool-call accuracy on the test split, run through the Needle engine."""
        n = int(ctx.config.get("eval_samples", 60))
        rows = read_jsonl(ctx.dataset_dir / "test.jsonl")[:n]
        if not rows:
            return None
        try:
            import needle
            from ..evaluators import tool_calls
            exact = names = valid = 0.0
            for r in rows:
                agent = needle.Needle(tools=r["tools"], weights=str(cact))
                res = agent.run(r["query"])
                out = json.dumps(res.get("function_calls", []))
                s = tool_calls({"reference": json.dumps(r["answers"]), "output": out})
                exact += s["exact_match"]; names += s["names_match"]; valid += s["valid_json"]
            k = len(rows)
            m = {"exact_match": exact / k, "names_match": names / k, "valid_json": valid / k, "n": float(k)}
            ctx.emit(stage="evaluated", **{f"test_{a}": b for a, b in m.items()})
            ctx.log(f"test exact match {m['exact_match']:.1%} on {k} examples (names only {m['names_match']:.1%})")
            return m
        except Exception as e:  # noqa: BLE001
            ctx.log(f"could not score the test split with the Needle engine ({type(e).__name__}: {e}). The .cact file was still built.")
            return None


PLUGIN = NeedlePlugin()
