"""MLX LM trainer (Apple Silicon): LoRA fine-tune, test loss, fuse to HF-format weights, rubric evaluation.

Follows the taskllm pipeline: iterations come from dataset size and epochs, the cosine schedule is tied to the real
iteration count, loss is on the assistant JSON only, and an earlier adapter can be resumed with a gentler step.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .. import evaluators
from ..base import TrainContext, dir_size, field_, write_manifest
from ..hf_common import evaluator_for, load_chat_splits

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# older mlx-lm: "Iter 20: Train loss 1.2, ..." ; current mlx-lm prints a table: "  70  1.020 ▼ 1,667  10.3k" and "  72  val 0.732  0.21s"
ITER_TRAIN = re.compile(r"(?:Iter (\d+): Train loss ([\d.]+))|(?:^\s*(\d+)\s+([\d.]+)\s*[▼▲=]?\s+[\d,]+\s)")
ITER_VAL = re.compile(r"(?:Iter (\d+): Val loss ([\d.]+))|(?:^\s*(\d+)\s+val\s+([\d.]+))")
TEST_LOSS = re.compile(r"Test loss ([\d.]+)")
NOISE = re.compile(r"it/s\]|it/sec|Calculating loss|Fetching \d+ files|^\s*$")


class MLXPlugin:
    id = "mlx_lm"
    requires = ["mlx_lm"]
    extra = "mlx"
    hardware = "Apple Silicon only"
    categories = ["text_to_json", "tool_calling"]

    def schema(self, category: str, entry: dict[str, Any]) -> list[dict[str, Any]]:
        return [field_("epochs", "Epochs", "float", 2, min=0.1, max=20, help="Validation loss usually goes flat after about 2 epochs. Add data rather than epochs."),
                field_("batch_size", "Batch size", "int", 4, min=1, max=64),
                field_("learning_rate", "Learning rate", "float", 8e-5, min=1e-6, max=1e-2),
                field_("lora_rank", "LoRA rank", "int", 16, min=2, max=128),
                field_("lora_scale", "LoRA scale", "float", 20.0, min=1, max=100, advanced=True),
                field_("lora_dropout", "LoRA dropout", "float", 0.1, min=0, max=0.5, advanced=True),
                field_("max_seq_length", "Max sequence length", "int", 1024, min=64, max=8192),
                field_("resume_from_job", "Continue from job id (optional)", "text", "", help="Keeps that job's adapter and trains on. Use a lower learning rate, for example 3e-5.", advanced=True),
                field_("eval_samples", "Test examples to generate for the report", "int", 40, min=0, max=2000),
                field_("max_new_tokens", "Max new tokens when generating", "int", 400, min=16, max=2048, advanced=True),
                field_("fuse", "Fuse adapter into standalone weights", "bool", True, help="Needed for GGUF and Core AI export."),
                field_("seed", "Random seed", "int", 42, advanced=True)]

    # ---------------------------------------------------------------------------------------------
    def _stream(self, ctx: TrainContext, cmd: list[str]) -> list[str]:
        ctx.log("$ " + " ".join(cmd))
        import os
        env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb", "PYTHONUNBUFFERED": "1"}
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert p.stdout
        lines = []
        for raw in p.stdout:
            for line in ANSI.sub("", raw).replace("\r", "\n").split("\n"):
                lines.append(line)
                m = ITER_VAL.search(line)
                if m:
                    ctx.emit(step=int(m.group(1) or m.group(3)), valid_loss=float(m.group(2) or m.group(4)))
                else:
                    m = ITER_TRAIN.search(line)
                    if m:
                        ctx.emit(step=int(m.group(1) or m.group(3)), train_loss=float(m.group(2) or m.group(4)))
                if not NOISE.search(line):
                    print(line, flush=True)
            if ctx.is_cancelled():
                p.terminate()
        code = p.wait()
        if code != 0:
            raise RuntimeError(f"command exited with code {code}")
        return lines

    def run(self, ctx: TrainContext) -> dict[str, Any]:
        import yaml
        cfg = ctx.config
        if not ctx.model_path:
            raise ValueError("this model is not downloaded")
        splits = load_chat_splits(ctx.dataset_dir)
        if not splits["train"] or not splits["valid"]:
            raise ValueError("dataset needs train and valid splits")
        data_dir = ctx.job_dir / "mlx_data"
        data_dir.mkdir(exist_ok=True)
        for part, rows in splits.items():
            with (data_dir / f"{part}.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")  # meta stays out of the training files
        n_train, bs = len(splits["train"]), int(cfg.get("batch_size", 4))
        iters = max(100, int(n_train * float(cfg.get("epochs", 2)) / bs))
        lr = float(cfg.get("learning_rate", 8e-5))
        adapters = ctx.out_dir / "adapters"
        adapters.mkdir(exist_ok=True)
        lora_cfg = {
            "model": ctx.model_path, "train": True, "data": str(data_dir), "fine_tune_type": "lora", "mask_prompt": True, "num_layers": -1,
            "lora_parameters": {"rank": int(cfg.get("lora_rank", 16)), "scale": float(cfg.get("lora_scale", 20.0)), "dropout": float(cfg.get("lora_dropout", 0.1)),
                                "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]},
            "batch_size": bs, "iters": iters, "learning_rate": lr,
            "lr_schedule": {"name": "cosine_decay", "warmup": min(100, max(10, iters // 10)), "arguments": [lr, iters, lr / 100]},
            "max_seq_length": int(cfg.get("max_seq_length", 1024)), "grad_checkpoint": True, "steps_per_report": 10,
            "steps_per_eval": max(10, iters // 8), "val_batches": 25, "save_every": max(50, iters // 4), "adapter_path": str(adapters), "seed": int(cfg.get("seed", 42)),
        }
        cfg_path = ctx.job_dir / "lora.yaml"
        cfg_path.write_text(yaml.safe_dump(lora_cfg), encoding="utf-8")
        resume: list[str] = []
        if cfg.get("resume_from_job"):
            from ... import config as appcfg
            if not re.fullmatch(r"job-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", str(cfg["resume_from_job"])):
                raise ValueError("resume_from_job must be a job id such as job-20261027-101500-ab12cd")
            prev = appcfg.jobs_dir() / str(cfg["resume_from_job"]) / "model" / "adapters" / "adapters.safetensors"
            if not prev.exists():
                raise FileNotFoundError(f"no adapter to resume at {prev}")
            resume = ["--resume-adapter-file", str(prev)]
            ctx.log(f"resuming from {prev}")
        ctx.log(f"{ctx.model['name']} · train/valid {n_train}/{len(splits['valid'])} · {iters} iterations ≈ {cfg.get('epochs', 2)} epochs at batch {bs} · lr {lr}")
        self._stream(ctx, [sys.executable, "-m", "mlx_lm", "lora", "-c", str(cfg_path), *resume])

        metrics: dict[str, Any] = {}
        ctx.log("test-set loss (answer tokens only)")
        test_cfg = {**lora_cfg, "train": False, "test": True}
        (ctx.job_dir / "lora-test.yaml").write_text(yaml.safe_dump(test_cfg), encoding="utf-8")
        lines = self._stream(ctx, [sys.executable, "-m", "mlx_lm", "lora", "-c", str(ctx.job_dir / "lora-test.yaml")])
        for line in lines:
            m = TEST_LOSS.search(line)
            if m:
                metrics["test_loss"] = float(m.group(1))

        artifacts: dict[str, str] = {"adapter": "adapters"}
        fused = ctx.out_dir / "hf"
        if cfg.get("fuse", True):
            ctx.log("fusing the adapter into standalone weights")
            self._stream(ctx, [sys.executable, "-m", "mlx_lm", "fuse", "--model", ctx.model_path, "--adapter-path", str(adapters), "--save-path", str(fused)])
            artifacts["hf_model"] = "hf"

        n_eval = min(int(cfg.get("eval_samples", 40)), len(splits["test"]))
        gen_metrics: dict[str, float] = {}
        if n_eval > 0:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
            model, tok = load(ctx.model_path, adapter_path=str(adapters))

            def gen(system: str, user: str) -> str:
                msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
                prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                return generate(model, tok, prompt=prompt, max_tokens=int(cfg.get("max_new_tokens", 400)), sampler=make_sampler(temp=0.0), verbose=False)

            gen_metrics, report = evaluators.evaluate(evaluator_for(ctx.category, cfg), splits["test"], gen, limit=n_eval,
                                                      title=f"{ctx.model['name']} on {ctx.dataset['name']}", log=ctx.log)
            (ctx.out_dir / "report.md").write_text(report, encoding="utf-8")
            artifacts["report"] = "report.md"
            ctx.emit(stage="evaluated", **{f"test_{k}": v for k, v in gen_metrics.items()})
        out_metrics = {"test": {**metrics, **gen_metrics}}
        write_manifest(ctx, {"trainer": self.id, "algorithm": "lora_sft_mlx", "iterations": iters, "artifacts": artifacts, "metrics": out_metrics,
                             "inference": {"mode": "chat", "category": ctx.category}, "size_bytes": dir_size(fused) if fused.exists() else dir_size(adapters)})
        return out_metrics


PLUGIN = MLXPlugin()
