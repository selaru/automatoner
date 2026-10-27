"""Transformers + PEFT trainer.

text_classification  -> sequence classification head (full fine-tune for small encoders, LoRA above ~200M parameters), ONNX export
text_to_json, tool_calling -> LoRA supervised fine-tuning of a small instruct model, merged HF weights out
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import evaluators
from .. import tabular_common as tc
from ..base import TrainContext, dir_size, field_, write_manifest
from ..hf_common import evaluator_for, load_chat_splits, read_jsonl
from .torch_tiny_plugin import pick_device


class HFTextPlugin:
    id = "hf_text"
    requires = ["torch", "transformers", "peft"]
    extra = "text"
    hardware = "CPU, CUDA or Apple MPS"

    def schema(self, category: str, entry: dict[str, Any]) -> list[dict[str, Any]]:
        dev = field_("device", "Device", "select", "auto", options=["auto", "cpu", "cuda", "mps"], advanced=True)
        seed = field_("seed", "Random seed", "int", 42, advanced=True)
        if category == "text_classification":
            return [field_("epochs", "Epochs", "int", 3, min=1, max=50), field_("batch_size", "Batch size", "int", 16, min=1, max=256),
                    field_("learning_rate", "Learning rate", "float", 5e-5, min=1e-6, max=1e-2),
                    field_("max_length", "Max tokens per text", "int", 128, min=16, max=2048),
                    field_("use_lora", "Use LoRA", "select", "auto", options=["auto", "yes", "no"], help="auto: LoRA for models above 200M parameters."),
                    field_("export_onnx", "Export ONNX", "bool", True), dev, seed]
        return [field_("epochs", "Epochs", "float", 2, min=0.1, max=20), field_("batch_size", "Batch size", "int", 4, min=1, max=64),
                field_("grad_accum", "Gradient accumulation", "int", 2, min=1, max=64),
                field_("learning_rate", "Learning rate", "float", 1e-4, min=1e-6, max=1e-2),
                field_("lora_rank", "LoRA rank", "int", 16, min=2, max=128), field_("lora_alpha", "LoRA alpha", "int", 32, min=1, max=256),
                field_("max_seq_length", "Max sequence length", "int", 1024, min=64, max=8192),
                field_("eval_samples", "Test examples to generate for the report", "int", 40, min=0, max=2000),
                field_("max_new_tokens", "Max new tokens when generating", "int", 400, min=16, max=2048), dev, seed]

    def run(self, ctx: TrainContext) -> dict[str, Any]:
        if ctx.category == "text_classification":
            return self._classification(ctx)
        if ctx.category in ("text_to_json", "tool_calling", "extraction"):
            return self._causal_lm(ctx)
        raise ValueError(f"hf_text cannot train {ctx.category}")

    # ---------------------------------------------------------------------------------------------
    def _classification(self, ctx: TrainContext) -> dict[str, Any]:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        cfg = ctx.config
        random.seed(int(cfg.get("seed", 42))); torch.manual_seed(int(cfg.get("seed", 42)))
        d = ctx.dataset_dir
        splits = {p: read_jsonl(d / f"{p}.jsonl") for p in ("train", "valid", "test")}
        labels = ctx.dataset["labels"]
        l2i = {l: i for i, l in enumerate(labels)}
        device = pick_device(cfg.get("device", "auto"))
        tok = AutoTokenizer.from_pretrained(ctx.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(ctx.model_path, num_labels=len(labels),
                                                                   id2label={i: l for l, i in l2i.items()}, label2id=l2i)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.pad_token_id
        n_params = sum(p.numel() for p in model.parameters())
        lora = cfg.get("use_lora", "auto")
        lora = (n_params > 200e6) if lora == "auto" else lora in (True, "yes")
        if lora:
            model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear"))
            ctx.log("LoRA on all linear layers; the classification head is trained fully")
        ctx.log(f"{n_params / 1e6:.0f}M parameters · device {device} · labels {labels}")
        model.to(device)
        max_len, bs = int(cfg.get("max_length", 128)), int(cfg.get("batch_size", 16))

        def batches(rows: list[dict[str, Any]], shuffle: bool):
            idx = list(range(len(rows)))
            if shuffle:
                random.shuffle(idx)
            for i in range(0, len(idx), bs):
                chunk = [rows[j] for j in idx[i:i + bs]]
                enc = tok([r["text"] for r in chunk], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
                yield {k: v.to(device) for k, v in enc.items()}, torch.tensor([l2i[str(r["label"])] for r in chunk], device=device)

        def evaluate(rows: list[dict[str, Any]]):
            model.eval()
            logits, ys = [], []
            with torch.no_grad():
                for enc, y in batches(rows, False):
                    logits.append(model(**enc).logits.float().cpu()); ys.append(y.cpu())
            lg, y = torch.cat(logits), torch.cat(ys)
            loss = float(torch.nn.functional.cross_entropy(lg, y))
            proba = torch.softmax(lg, -1).numpy()
            return loss, tc.classification_metrics(y.numpy(), proba.argmax(1), proba, labels)

        epochs = int(cfg.get("epochs", 3))
        steps = epochs * math.ceil(len(splits["train"]) / bs)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg.get("learning_rate", 5e-5)), weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        step, best, best_state = 0, 1e9, None
        for epoch in range(epochs):
            model.train()
            run_loss = []
            for enc, y in batches(splits["train"], True):
                if ctx.is_cancelled():
                    raise KeyboardInterrupt
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1; run_loss.append(float(loss))
                if step % 10 == 0:
                    ctx.emit(step=step, train_loss=float(np.mean(run_loss[-10:])))
            vloss, vm = evaluate(splits["valid"])
            ctx.emit(step=step, epoch=epoch + 1, valid_loss=vloss, valid_accuracy=vm["accuracy"], valid_macro_f1=vm["macro_f1"])
            ctx.log(f"epoch {epoch + 1}/{epochs}: valid loss {vloss:.4f}, accuracy {vm['accuracy']:.3f}, macro F1 {vm['macro_f1']:.3f}")
            if vloss < best:
                best = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        metrics = {"valid": evaluate(splits["valid"])[1], "test": evaluate(splits["test"])[1]}
        ctx.emit(stage="evaluated", test_accuracy=metrics["test"]["accuracy"], test_macro_f1=metrics["test"]["macro_f1"])

        out = ctx.out_dir
        if lora:
            model = model.merge_and_unload()
        model.eval().cpu()
        hf_dir = out / "hf"
        model.save_pretrained(hf_dir)
        tok.save_pretrained(hf_dir)
        artifacts: dict[str, str] = {"hf_model": "hf"}
        extra: dict[str, Any] = {}
        if (hf_dir / "tokenizer.json").exists():
            artifacts["tokenizer"] = "hf/tokenizer.json"
        if cfg.get("export_onnx", True) and n_params < 400e6:
            try:
                onnx_info = self._export_classifier_onnx(model, tok, out / "model.onnx", splits["test"][:16], max_len)
                artifacts["onnx"] = "model.onnx"
                extra["parity"] = onnx_info
                ctx.log(f"ONNX parity: max abs diff {onnx_info['max_abs_diff']:.2e}")
            except Exception as e:  # noqa: BLE001
                ctx.log(f"warning: ONNX export failed ({type(e).__name__}: {e}). The Hugging Face folder is still available.")
                extra["warnings"] = [f"ONNX export failed: {e}"]
        inference = {"mode": "text_classification", "classes": labels, "max_length": max_len, "onnx_output": "logits"}
        (out / "inference.json").write_text(json.dumps(inference, indent=2), encoding="utf-8")
        artifacts["inference"] = "inference.json"
        write_manifest(ctx, {"trainer": self.id, "algorithm": "sequence_classification", "lora": bool(lora), "device": device, "artifacts": artifacts,
                             "inference": inference, "metrics": metrics, "size_bytes": dir_size(out), **extra})
        return metrics

    @staticmethod
    def _export_classifier_onnx(model, tok, path: Path, sample_rows: list[dict[str, Any]], max_len: int) -> dict[str, Any]:
        import onnxruntime as ort
        import torch

        class Wrap(torch.nn.Module):
            def __init__(self, m) -> None:  # noqa: ANN001
                super().__init__(); self.m = m

            def forward(self, input_ids, attention_mask):  # noqa: ANN001
                return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

        w = Wrap(model).eval()
        enc = tok([r["text"] for r in sample_rows] or ["example"], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        args = (enc["input_ids"], enc["attention_mask"])
        torch.onnx.export(w, args, str(path), input_names=["input_ids", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(["logits"], {"input_ids": args[0].numpy(), "attention_mask": args[1].numpy()})[0]
        with torch.no_grad():
            want = w(*args).numpy()
        diff = float(np.max(np.abs(got - want)))
        if diff > 1e-2:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"ONNX output differs from PyTorch by {diff}")
        return {"output": "logits", "max_abs_diff": diff, "rows": int(len(want))}

    # ---------------------------------------------------------------------------------------------
    def _causal_lm(self, ctx: TrainContext) -> dict[str, Any]:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = ctx.config
        random.seed(int(cfg.get("seed", 42))); torch.manual_seed(int(cfg.get("seed", 42)))
        device = pick_device(cfg.get("device", "auto"))
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        splits = load_chat_splits(ctx.dataset_dir)
        tok = AutoTokenizer.from_pretrained(ctx.model_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(ctx.model_path, torch_dtype=dtype)
        model = get_peft_model(model, LoraConfig(r=int(cfg.get("lora_rank", 16)), lora_alpha=int(cfg.get("lora_alpha", 32)), lora_dropout=0.05,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
        model.to(device)
        max_len = int(cfg.get("max_seq_length", 1024))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ctx.log(f"LoRA rank {cfg.get('lora_rank', 16)} · {trainable / 1e6:.1f}M trainable parameters · device {device} · dtype {dtype}")

        def encode(row: dict[str, Any]):
            msgs = row["messages"]
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            full = tok.apply_chat_template(msgs, tokenize=False)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            if len(f_ids) > max_len or f_ids[:len(p_ids)] != p_ids:
                return None
            return f_ids, [-100] * len(p_ids) + f_ids[len(p_ids):]

        data = {k: [e for e in (encode(r) for r in v) if e] for k, v in splits.items()}
        ctx.log(f"examples that fit {max_len} tokens: train {len(data['train'])}/{len(splits['train'])}, valid {len(data['valid'])}")
        if not data["train"]:
            raise ValueError("no training example fits the maximum sequence length")
        bs, accum = int(cfg.get("batch_size", 4)), int(cfg.get("grad_accum", 2))

        def collate(items):  # noqa: ANN001
            n = max(len(x[0]) for x in items)
            ids = torch.full((len(items), n), tok.pad_token_id); lab = torch.full((len(items), n), -100); att = torch.zeros((len(items), n), dtype=torch.long)
            for i, (a, b) in enumerate(items):
                ids[i, :len(a)] = torch.tensor(a); lab[i, :len(b)] = torch.tensor(b); att[i, :len(a)] = 1
            return ids.to(device), att.to(device), lab.to(device)

        def valid_loss() -> float:
            model.eval(); tot, cnt = 0.0, 0
            with torch.no_grad():
                for i in range(0, min(len(data["valid"]), 64), bs):
                    ids, att, lab = collate(data["valid"][i:i + bs])
                    out = model(input_ids=ids, attention_mask=att, labels=lab)
                    n = int((lab[:, 1:] != -100).sum()); tot += float(out.loss) * n; cnt += n
            model.train()
            return tot / max(1, cnt)

        steps_per_epoch = math.ceil(len(data["train"]) / (bs * accum))
        total_steps = max(1, int(float(cfg.get("epochs", 2)) * steps_per_epoch))
        lr = float(cfg.get("learning_rate", 1e-4))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
        warm = max(1, total_steps // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))) * 0.99 + 0.01)
        model.train()
        step, run = 0, []
        order = list(range(len(data["train"])))
        pos = len(order)
        eval_every = max(1, total_steps // 8)
        t0 = time.time()
        while step < total_steps:
            for _ in range(accum):
                if pos >= len(order):
                    random.shuffle(order); pos = 0
                items = [data["train"][i] for i in order[pos:pos + bs]]; pos += bs
                ids, att, lab = collate(items)
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / accum
                loss.backward(); run.append(float(loss) * accum)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if ctx.is_cancelled():
                raise KeyboardInterrupt
            if step % 5 == 0 or step == total_steps:
                ctx.emit(step=step, train_loss=float(np.mean(run[-5 * accum:])), lr=sched.get_last_lr()[0])
                ctx.log(f"step {step}/{total_steps} loss {np.mean(run[-5 * accum:]):.4f} ({(time.time() - t0) / step:.1f}s/step)")
            if step % eval_every == 0 or step == total_steps:
                vl = valid_loss(); ctx.emit(step=step, valid_loss=vl); ctx.log(f"step {step}: valid loss {vl:.4f}")

        out = ctx.out_dir
        model.save_pretrained(out / "adapter")
        model.eval()
        metrics: dict[str, Any] = {}
        n_eval = min(int(cfg.get("eval_samples", 40)), len(splits["test"]))
        report_md = None
        if n_eval > 0:
            def gen(system: str, user: str) -> str:
                msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=int(cfg.get("max_new_tokens", 400)), do_sample=False, pad_token_id=tok.pad_token_id)
                return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

            overall, report_md = evaluators.evaluate(evaluator_for(ctx.category, cfg), splits["test"], gen, limit=n_eval,
                                                     title=f"{ctx.model['name']} on {ctx.dataset['name']}", log=ctx.log)
            metrics["test"] = overall
            (out / "report.md").write_text(report_md, encoding="utf-8")
            ctx.emit(stage="evaluated", **{f"test_{k}": v for k, v in overall.items()})
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "hf", safe_serialization=True)
        tok.save_pretrained(out / "hf")
        artifacts = {"hf_model": "hf", "adapter": "adapter"}
        if report_md:
            artifacts["report"] = "report.md"
        write_manifest(ctx, {"trainer": self.id, "algorithm": "lora_sft", "device": device, "artifacts": artifacts, "metrics": metrics,
                             "inference": {"mode": "chat", "category": ctx.category}, "size_bytes": dir_size(out / "hf")})
        return metrics


PLUGIN = HFTextPlugin()
