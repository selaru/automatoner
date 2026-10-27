"""Tiny PyTorch autoencoder for tabular anomaly detection. The reconstruction error is computed inside the
exported ONNX graph, so the Lambda handler needs nothing beyond onnxruntime."""
from __future__ import annotations

from typing import Any

import numpy as np

from .. import tabular_common as tc
from ..base import TrainContext, dir_size, field_, write_manifest


def pick_device(pref: str = "auto") -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchTinyPlugin:
    id = "torch_tiny"
    requires = ["torch", "onnx", "onnxruntime", "sklearn"]
    extra = "text"
    hardware = "CPU, CUDA or Apple MPS"
    categories = ["tabular_anomaly"]

    def schema(self, category: str, entry: dict[str, Any]) -> list[dict[str, Any]]:
        return [field_("epochs", "Epochs", "int", 40, min=1, max=500),
                field_("batch_size", "Batch size", "int", 128, min=8, max=4096),
                field_("learning_rate", "Learning rate", "float", 2e-3, min=1e-5, max=0.1),
                field_("bottleneck", "Bottleneck size", "int", 3, min=1, max=64, help="Smaller forces the model to learn only the normal pattern."),
                field_("hidden", "Hidden size", "int", 16, min=4, max=256),
                field_("contamination", "Expected anomaly share", "float", 0.02, min=0.001, max=0.4, help="Places the alert threshold when there are no labels."),
                field_("train_on_normal_only", "Train on normal rows only", "bool", True),
                field_("device", "Device", "select", "auto", options=["auto", "cpu", "cuda", "mps"], advanced=True),
                field_("seed", "Random seed", "int", 42, advanced=True)]

    def run(self, ctx: TrainContext) -> dict[str, Any]:
        import torch
        from torch import nn

        cfg = ctx.config
        seed = int(cfg.get("seed", 42))
        torch.manual_seed(seed)
        splits = tc.load_splits(ctx)
        train, valid, test = splits["train"], splits.get("valid"), splits.get("test")
        label = ctx.dataset.get("label_column")
        fit_df = train
        if label and cfg.get("train_on_normal_only", True) and train[label].nunique() > 1:
            fit_df = train[train[label] == 0]
            ctx.log(f"training on {len(fit_df)} normal rows")
        spec = tc.build_features(ctx, fit_df)
        Xtr = torch.tensor(spec.transform(fit_df))
        d = spec.dim
        hidden, bott = int(cfg.get("hidden", 16)), int(cfg.get("bottleneck", 3))
        device = pick_device(cfg.get("device", "auto"))
        ctx.log(f"autoencoder {d} -> {hidden} -> {bott} -> {hidden} -> {d} on {device}")

        class AE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, bott))
                self.dec = nn.Sequential(nn.ReLU(), nn.Linear(bott, hidden), nn.ReLU(), nn.Linear(hidden, d))

            def forward(self, x):  # noqa: ANN001
                return self.dec(self.enc(x))

        class Scorer(nn.Module):
            """Graph that returns the per-row reconstruction error, shape [N, 1]."""
            def __init__(self, ae: nn.Module) -> None:
                super().__init__()
                self.ae = ae

            def forward(self, x):  # noqa: ANN001
                return ((self.ae(x) - x) ** 2).mean(dim=1, keepdim=True)

        ae = AE().to(device)
        opt = torch.optim.Adam(ae.parameters(), lr=float(cfg.get("learning_rate", 2e-3)))
        bs = int(cfg.get("batch_size", 128))
        Xv = torch.tensor(spec.transform(valid)).to(device) if valid is not None else None
        if Xv is not None and label:
            normal = valid[label].to_numpy() == 0
            Xv_norm = Xv[torch.tensor(normal).to(device)]
        else:
            Xv_norm = Xv
        step = 0
        for epoch in range(int(cfg.get("epochs", 40))):
            if ctx.is_cancelled():
                break
            perm = torch.randperm(len(Xtr))
            total = 0.0
            ae.train()
            for i in range(0, len(Xtr), bs):
                xb = Xtr[perm[i:i + bs]].to(device)
                loss = ((ae(xb) - xb) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss) * len(xb); step += 1
            ae.eval()
            with torch.no_grad():
                vloss = float(((ae(Xv_norm) - Xv_norm) ** 2).mean()) if Xv_norm is not None and len(Xv_norm) else None
            ctx.emit(step=epoch + 1, train_loss=total / len(Xtr), **({"valid_loss": vloss} if vloss is not None else {}))
        ae.eval().cpu()
        scorer = Scorer(ae).eval()

        def raw_fn(X: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return scorer(torch.tensor(X)).numpy().reshape(-1)

        raw_tr = raw_fn(spec.transform(fit_df))
        y_valid = valid[label].to_numpy() if (valid is not None and label) else None
        thr = tc.threshold_by_f1(raw_fn(spec.transform(valid)), y_valid) if y_valid is not None else None
        meta: dict[str, Any] = {"threshold_source": "best F1 on the validation split"}
        if thr is None:
            thr = float(np.quantile(raw_tr, 1 - float(cfg.get("contamination", 0.02))))
            meta["threshold_source"] = "contamination quantile on training scores"
        lo, hi = tc.calibrate(raw_tr, thr)
        metrics = {}
        for name, df in (("valid", valid), ("test", test)):
            if df is not None:
                metrics[name] = tc.anomaly_metrics(raw_fn(spec.transform(df)), df[label].to_numpy() if label else None, thr)
        inference = {"mode": "anomaly", "onnx_output": "scores", "raw_sign": 1.0, "lo": lo, "hi": hi,
                     "threshold": 0.5, "threshold_raw": thr}

        out = ctx.out_dir
        example = torch.zeros(2, d)
        try:
            torch.onnx.export(scorer, example, str(out / "model.onnx"), input_names=["input"], output_names=["scores"],
                              dynamic_axes={"input": {0: "batch"}, "scores": {0: "batch"}}, opset_version=17, dynamo=False)
        except TypeError:  # older torch without the dynamo flag
            torch.onnx.export(scorer, example, str(out / "model.onnx"), input_names=["input"], output_names=["scores"],
                              dynamic_axes={"input": {0: "batch"}, "scores": {0: "batch"}}, opset_version=17)
        sample = spec.transform(test if test is not None else train)[:200]
        parity = tc.onnx_parity((out / "model.onnx").read_bytes(), sample, raw_fn(sample).reshape(-1, 1), "scores")
        ctx.log(f"ONNX parity: max abs diff {parity['max_abs_diff']:.2e}")
        if parity["max_abs_diff"] > 1e-3:
            raise RuntimeError(f"ONNX export disagrees with the trained model (max diff {parity['max_abs_diff']})")
        torch.save(ae.state_dict(), out / "model.pt")
        torch.jit.trace(scorer, torch.zeros(2, d)).save(str(out / "model.ts"))  # TorchScript, used by the Core ML export
        tc.save_json(out / "features.json", spec.to_dict())
        tc.save_json(out / "inference.json", inference)
        write_manifest(ctx, {"trainer": self.id, "algorithm": "autoencoder", "device": device,
                             "artifacts": {"onnx": "model.onnx", "features": "features.json", "inference": "inference.json", "torch": "model.pt", "torchscript": "model.ts"},
                             "inference": inference, "metrics": metrics, "parity": parity, "size_bytes": dir_size(out),
                             "onnx_bytes": (out / "model.onnx").stat().st_size, **meta})
        return metrics


PLUGIN = TorchTinyPlugin()
