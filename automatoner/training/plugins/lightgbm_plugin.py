"""LightGBM trainer for tabular classification, exported to ONNX."""
from __future__ import annotations

from typing import Any

import numpy as np

from .. import tabular_common as tc
from ..base import TrainContext, dir_size, field_, write_manifest


class LightGBMPlugin:
    id = "lightgbm"
    requires = ["lightgbm", "onnxmltools", "onnx", "onnxruntime", "sklearn"]
    extra = "tabular"
    hardware = "CPU"
    categories = ["tabular_classification"]

    def schema(self, category: str, entry: dict[str, Any]) -> list[dict[str, Any]]:
        return [field_("n_estimators", "Boosting rounds", "int", 200, min=10, max=3000),
                field_("learning_rate", "Learning rate", "float", 0.05, min=0.001, max=1.0),
                field_("num_leaves", "Leaves per tree", "int", 31, min=4, max=256, help="Smaller keeps the model under a few MB."),
                field_("balanced", "Balance classes", "bool", True),
                field_("early_stopping", "Early stopping rounds", "int", 30, min=0, max=500, help="Uses the validation split. 0 turns it off."),
                field_("seed", "Random seed", "int", 42, advanced=True)]

    def run(self, ctx: TrainContext) -> dict[str, Any]:
        try:
            import lightgbm as lgb
        except OSError as e:
            if "libomp" in str(e):
                raise RuntimeError("LightGBM needs the OpenMP library on macOS. Run: brew install libomp") from e
            raise
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        cfg = ctx.config
        splits = tc.load_splits(ctx)
        train, valid, test = splits["train"], splits.get("valid"), splits.get("test")
        label = ctx.dataset.get("label_column")
        if not label:
            raise ValueError("classification needs a label column")
        classes = sorted(str(v) for v in train[label].unique())
        idx = {c: i for i, c in enumerate(classes)}
        spec = tc.build_features(ctx, train)
        Xtr, ytr = spec.transform(train), train[label].astype(str).map(idx).to_numpy()
        ctx.log(f"LightGBM · train rows {len(train)} · classes {classes} · features {spec.dim}")

        model = lgb.LGBMClassifier(n_estimators=int(cfg.get("n_estimators", 200)), learning_rate=float(cfg.get("learning_rate", 0.05)),
                                   num_leaves=int(cfg.get("num_leaves", 31)), class_weight="balanced" if cfg.get("balanced", True) else None,
                                   random_state=int(cfg.get("seed", 42)), verbose=-1, n_jobs=-1)
        fit_kw: dict[str, Any] = {}
        es = int(cfg.get("early_stopping", 30))
        history: dict[str, Any] = {}
        if valid is not None and es > 0:
            yv = valid[label].astype(str).map(idx).fillna(-1).astype(int).to_numpy()
            keep = yv >= 0
            fit_kw = {"eval_set": [(spec.transform(valid)[keep], yv[keep])],
                      "callbacks": [lgb.early_stopping(es, verbose=False), lgb.record_evaluation(history)]}
        model.fit(Xtr, ytr, **fit_kw)
        for i, v in enumerate(next(iter(history.values()), {}).get("multi_logloss") or next(iter(history.values()), {}).get("binary_logloss") or []):
            ctx.emit(step=i + 1, valid_loss=float(v))

        metrics: dict[str, Any] = {}
        for name, df in (("valid", valid), ("test", test)):
            if df is not None:
                y = df[label].astype(str).map(idx).fillna(-1).astype(int).to_numpy()
                keep = y >= 0
                proba = model.predict_proba(spec.transform(df)[keep])
                metrics[name] = tc.classification_metrics(y[keep], proba.argmax(1), proba, classes)
        ctx.emit(stage="evaluated", **{f"{k}_{m}": v for k, d in metrics.items() for m, v in d.items() if isinstance(v, (int, float))})

        onx = onnxmltools.convert_lightgbm(model, initial_types=[("input", FloatTensorType([None, spec.dim]))], zipmap=False, target_opset=15)
        sample = spec.transform(test if test is not None else train)[:200]
        outputs = [o.name for o in onx.graph.output]
        prob_name = next(n for n in outputs if "prob" in n.lower())
        parity = tc.onnx_parity(onx.SerializeToString(), sample, model.predict_proba(sample), prob_name)
        ctx.log(f"ONNX parity: max abs diff {parity['max_abs_diff']:.2e}")
        if parity["max_abs_diff"] > 1e-3:
            raise RuntimeError(f"ONNX export disagrees with the trained model (max diff {parity['max_abs_diff']})")

        import joblib

        out = ctx.out_dir
        (out / "model.onnx").write_bytes(onx.SerializeToString())
        inference = {"mode": "classification", "onnx_output": prob_name, "classes": classes}
        joblib.dump({"model": model, "features": spec.to_dict(), "inference": inference}, out / "model.joblib")
        tc.save_json(out / "features.json", spec.to_dict())
        tc.save_json(out / "inference.json", inference)
        write_manifest(ctx, {"trainer": self.id, "algorithm": "lightgbm", "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
                             "artifacts": {"onnx": "model.onnx", "joblib": "model.joblib", "features": "features.json", "inference": "inference.json"},
                             "inference": inference, "metrics": metrics, "parity": parity, "size_bytes": dir_size(out),
                             "onnx_bytes": (out / "model.onnx").stat().st_size})
        return metrics


PLUGIN = LightGBMPlugin()
