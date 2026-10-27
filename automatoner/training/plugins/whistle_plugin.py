"""Cactus Whistle trainer: add a language, your voice or your vocabulary with LoRA, then build a .cact."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import cactus_common as cc
from ..base import TrainContext, dir_size, field_, write_manifest


class WhistlePlugin:
    id = "whistle"
    requires = ["needle"]
    extra = "speech"
    hardware = "CPU, CUDA (Linux) or Metal (macOS); JAX on native Windows is CPU only"
    categories = ["speech"]

    def schema(self, category: str, entry: dict[str, Any]) -> list[dict[str, Any]]:
        return [field_("epochs", "Epochs", "int", 3, min=1, max=100),
                field_("batch_size", "Batch size", "int", 16, min=1, max=128),
                field_("lr", "Learning rate", "float", 1e-4, min=1e-6, max=1e-1, help="Cactus reached Brazilian Portuguese with a much higher rate (1e-2) on a new language."),
                field_("language", "Language code for lines without one", "text", "", help="A new code adds that language."),
                field_("tokens", "New tokens, one per line", "text", "", help="Unusual characters, names or product words that should be single tokens. Separate with | here.", advanced=True),
                field_("lora_rank", "LoRA rank", "int", 16, min=4, max=128, advanced=True),
                field_("eval_samples", "Test clips to transcribe for the score", "int", 100, min=0, max=5000, advanced=True),
                field_("layers", "Decoder layers in the exported model", "int", 8, min=2, max=8, help="2 to 8. Fewer layers: smaller and faster, less accurate."),
                field_("seed", "Random seed", "int", 42, advanced=True)]

    def run(self, ctx: TrainContext) -> dict[str, Any]:
        cfg = ctx.config
        d = ctx.dataset_dir
        for part in ("train", "valid", "test"):
            if not (d / f"{part}.jsonl").exists():
                raise ValueError(f"speech dataset is missing {part}.jsonl")
        out = ctx.out_dir
        adapter = out / "adapter.safetensors"
        cmd = [*cc.needle_exe(), "whistle", "finetune", str(d / "train.jsonl"), str(d / "valid.jsonl"), str(d / "test.jsonl"),
               "--epochs", str(int(cfg.get("epochs", 3))), "--batch-size", str(int(cfg.get("batch_size", 16))), "--lr", str(cfg.get("lr", 1e-4)),
               "--lora-rank", str(int(cfg.get("lora_rank", 16))), "--seed", str(int(cfg.get("seed", 42))), "--out", str(adapter),
               "--checkpoint-dir", str(ctx.job_dir / "ckpt")]
        if cfg.get("language"):
            cmd += ["--language", str(cfg["language"])]
        if cfg.get("tokens"):
            tokens = out / "tokens.txt"
            tokens.write_text("\n".join(t.strip() for t in str(cfg["tokens"]).replace("|", "\n").splitlines() if t.strip()) + "\n", encoding="utf-8")
            cmd += ["--tokens", str(tokens)]
        if ctx.model_path:
            ck = Path(ctx.model_path) / "checkpoints" / "whistle.safetensors"
            if ck.exists():
                cmd += ["--checkpoint", str(ck)]
        lines = cc.run_cli(ctx, cmd)
        if not adapter.exists():
            raise RuntimeError("needle whistle finetune finished without writing the adapter")
        layers = int(cfg.get("layers", 8))
        cact = out / f"whistle-{layers}l.cact"
        build = [*cc.needle_exe(), "whistle", "build", "--lora", str(adapter), "--layers", str(layers), "--out", str(cact)]
        if ctx.model_path and (Path(ctx.model_path) / "checkpoints" / "whistle.safetensors").exists():
            build.insert(3, str(Path(ctx.model_path) / "checkpoints" / "whistle.safetensors"))
        cc.run_cli(ctx, build)
        metrics: dict[str, Any] = {}
        import re
        for ln in lines:
            m = re.search(r"wer\s+([\d.]+)%\s+on\s+(\d+)\s+test", ln, re.I)
            if m:
                metrics["test_training_run"] = {"wer": float(m.group(1)) / 100, "n": float(m.group(2))}
        engine = self._evaluate(ctx, cact)
        if engine:
            metrics["test"] = engine
            ctx.emit(stage="evaluated", test_wer=engine["wer"])
        write_manifest(ctx, {"trainer": self.id, "algorithm": "whistle_lora", "layers": layers,
                             "artifacts": {"cact": cact.name, "adapter": adapter.name}, "metrics": metrics, "size_bytes": dir_size(out),
                             "cact_bytes": cact.stat().st_size, "inference": {"mode": "speech"}})
        return metrics


    def _evaluate(self, ctx: TrainContext, cact: Path) -> dict[str, float] | None:
        """Word error rate on the test split, transcribed by the built .cact through the Needle engine."""
        import json as _json
        n = int(ctx.config.get("eval_samples", 100))
        rows = [_json.loads(x) for x in (ctx.dataset_dir / "test.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()][:n]
        if not rows:
            return None
        try:
            import needle
            from whisper_normalizer.basic import BasicTextNormalizer
            try:
                from whisper_normalizer.english import EnglishTextNormalizer
                norm_en = EnglishTextNormalizer()
            except Exception:  # noqa: BLE001
                norm_en = BasicTextNormalizer()
            basic = BasicTextNormalizer()
            errors = words = 0
            for r in rows:
                lang = r.get("language") or ctx.config.get("language") or None
                out = needle.transcribe(str(ctx.dataset_dir / r["audio"]), language=lang, weights=str(cact))
                norm = norm_en if lang in (None, "en") else basic
                ref, hyp = norm(r["text"]).split(), norm(out["text"]).split()
                errors += _edit_distance(ref, hyp)
                words += len(ref)
            wer = errors / max(1, words)
            ctx.log(f"test WER {wer:.1%} on {len(rows)} clips, transcribed by the built model")
            return {"wer": wer, "n": float(len(rows))}
        except Exception as e:  # noqa: BLE001
            ctx.log(f"could not score the test clips with the engine ({type(e).__name__}: {e}). The .cact file was still built.")
            return None


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


PLUGIN = WhistlePlugin()
