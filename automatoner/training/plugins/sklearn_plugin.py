"""scikit-learn trainer: anomaly detection and tabular classification, exported to ONNX."""
from __future__ import annotations

from typing import Any

import numpy as np

from .. import tabular_common as tc
from ..base import TrainContext, dir_size, field_, write_manifest

ALGOS = {"isolation_forest", "one_class_svm", "logistic_regression", "random_forest"}


class SklearnPlugin:
    id = "sklearn"
    requires = ["sklearn", "skl2onnx", "onnx", "onnxruntime"]
    extra = "tabular"
    hardware = "CPU"
    categories = ["tabular_anomaly", "tabular_classification"]

    def schema(self, category: str, entry: dict[str, Any]) -> list[dict[str, Any]]:
        algo = (entry.get("defaults") or {}).get("algorithm")
        f = [field_("seed", "Random seed", "int", 42, advanced=True)]
        if algo == "isolation_forest":
            f = [field_("n_estimators", "Trees", "int", 100, min=10, max=1000, help="More trees give steadier scores and a larger model."),
                 field_("contamination", "Expected anomaly share", "float", 0.02, min=0.001, max=0.4,
                        help="Used to place the alert threshold when the data has no labels."),
                 field_("train_on_normal_only", "Train on normal rows only", "bool", True, help="Needs a 0/1 label column.")] + f
        elif algo == "one_class_svm":
            f = [field_("nu", "Nu (upper bound on training outliers)", "float", 0.05, min=0.001, max=0.5),
                 field_("max_train_rows", "Max training rows", "int", 10000, min=500, max=50000, help="The SVM is quadratic in rows."),
                 field_("contamination", "Expected anomaly share", "float", 0.02, min=0.001, max=0.4),
                 field_("train_on_normal_only", "Train on normal rows only", "bool", True)] + f
        elif algo == "logistic_regression":
            f = [field_("C", "Regularisation C", "float", 1.0, min=0.001, max=1000),
                 field_("max_iter", "Max iterations", "int", 500, min=50, max=5000),
                 field_("balanced", "Balance classes", "bool", True)] + f
        elif algo == "random_forest":
            f = [field_("n_estimators", "Trees", "int", 100, min=10, max=1000),
                 field_("max_depth", "Max depth", "int", 12, min=2, max=40, help="Deeper trees make the file bigger."),
                 field_("balanced", "Balance classes", "bool", True)] + f
        return f

    def run(self, ctx: TrainContext) -> dict[str, Any]:
        from skl2onnx import to_onnx
        from sklearn.ensemble import IsolationForest, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import OneClassSVM

        cfg = ctx.config
        algo = cfg.get("algorithm") or (ctx.model.get("defaults") or {}).get("algorithm")
        if algo not in ALGOS:
            raise ValueError(f"unknown algorithm {algo!r}")
        seed = int(cfg.get("seed", 42))
        splits = tc.load_splits(ctx)
        train, valid, test = splits["train"], splits.get("valid"), splits.get("test")
        label = ctx.dataset.get("label_column")
        anomaly = ctx.category == "tabular_anomaly"
        ctx.log(f"algorithm: {algo} · train rows {len(train)} · features {len(ctx.dataset['numeric_columns'])} numeric, {len(ctx.dataset['categorical_columns'])} categorical")

        fit_df = train
        if anomaly and label and cfg.get("train_on_normal_only", True) and train[label].nunique() > 1:
            fit_df = train[train[label] == 0]
            ctx.log(f"training on {len(fit_df)} normal rows (label == 0)")
        spec = tc.build_features(ctx, fit_df)
        Xtr = spec.transform(fit_df)
        ctx.emit(stage="features", dim=spec.dim)

        meta: dict[str, Any] = {}
        if anomaly:
            if algo == "isolation_forest":
                model = IsolationForest(n_estimators=int(cfg.get("n_estimators", 100)), random_state=seed, n_jobs=-1).fit(Xtr)
            elif algo == "one_class_svm":
                cap = int(cfg.get("max_train_rows", 10000))
                Xfit = Xtr if len(Xtr) <= cap else Xtr[np.random.default_rng(seed).choice(len(Xtr), cap, replace=False)]
                model = OneClassSVM(nu=float(cfg.get("nu", 0.05)), gamma="scale").fit(Xfit)
            else:
                raise ValueError(f"{algo} is not an anomaly detector")
            raw_fn = lambda X: -model.decision_function(X)  # noqa: E731 - higher means more anomalous
            raw_tr = raw_fn(Xtr)
            y_valid = valid[label].to_numpy() if (valid is not None and label) else None
            thr = tc.threshold_by_f1(raw_fn(spec.transform(valid)), y_valid) if y_valid is not None else None
            if thr is None:
                thr = float(np.quantile(raw_tr, 1 - float(cfg.get("contamination", 0.02))))
                meta["threshold_source"] = "contamination quantile on training scores"
            else:
                meta["threshold_source"] = "best F1 on the validation split"
            lo, hi = tc.calibrate(raw_tr, thr)
            metrics = {}
            for name, df in (("valid", valid), ("test", test)):
                if df is not None:
                    y = df[label].to_numpy() if label else None
                    metrics[name] = tc.anomaly_metrics(raw_fn(spec.transform(df)), y, thr)
            inference = {"mode": "anomaly", "onnx_output": "scores", "raw_sign": -1.0, "lo": lo, "hi": hi,
                         "threshold": 0.5, "threshold_raw": thr}
            onx = to_onnx(model, Xtr[:1], target_opset={"": 17, "ai.onnx.ml": 3})
            sample = spec.transform(test if test is not None else train)[:200]
            parity = tc.onnx_parity(onx.SerializeToString(), sample, model.decision_function(sample), "scores")
        else:
            if not label:
                raise ValueError("classification needs a label column")
            classes = sorted(str(v) for v in train[label].unique())
            idx = {c: i for i, c in enumerate(classes)}
            ytr = train[label].astype(str).map(idx).to_numpy()
            cw = "balanced" if cfg.get("balanced", True) else None
            if algo == "logistic_regression":
                model = LogisticRegression(C=float(cfg.get("C", 1.0)), max_iter=int(cfg.get("max_iter", 500)), class_weight=cw)
            elif algo == "random_forest":
                model = RandomForestClassifier(n_estimators=int(cfg.get("n_estimators", 100)), max_depth=int(cfg.get("max_depth", 12)),
                                               class_weight=cw, random_state=seed, n_jobs=-1)
            else:
                raise ValueError(f"{algo} is not a classifier")
            model.fit(spec.transform(train), ytr)
            metrics = {}
            for name, df in (("valid", valid), ("test", test)):
                if df is not None:
                    X = spec.transform(df); y = df[label].astype(str).map(idx).fillna(-1).astype(int).to_numpy()
                    keep = y >= 0
                    proba = model.predict_proba(X[keep])
                    metrics[name] = tc.classification_metrics(y[keep], proba.argmax(1), proba, classes)
            inference = {"mode": "classification", "onnx_output": "probabilities", "classes": classes}
            onx = to_onnx(model, spec.transform(train)[:1], target_opset={"": 17, "ai.onnx.ml": 3}, options={id(model): {"zipmap": False}})
            sample = spec.transform(test if test is not None else train)[:200]
            parity = tc.onnx_parity(onx.SerializeToString(), sample, model.predict_proba(sample), "probabilities")

        ctx.emit(stage="evaluated", **{f"{k}_{m}": v for k, d in metrics.items() for m, v in d.items() if isinstance(v, (int, float))})
        ctx.log(f"ONNX parity: max abs diff {parity['max_abs_diff']:.2e} over {parity['rows']} rows")
        if parity["max_abs_diff"] > 1e-3:
            raise RuntimeError(f"ONNX export disagrees with the trained model (max diff {parity['max_abs_diff']}); refusing to publish it")

        import joblib

        out = ctx.out_dir
        (out / "model.onnx").write_bytes(onx.SerializeToString())
        joblib.dump({"model": model, "features": spec.to_dict(), "inference": inference}, out / "model.joblib")
        tc.save_json(out / "features.json", spec.to_dict())
        tc.save_json(out / "inference.json", inference)
        write_manifest(ctx, {
            "trainer": self.id, "algorithm": algo, "artifacts": {"onnx": "model.onnx", "joblib": "model.joblib", "features": "features.json", "inference": "inference.json"},
            "inference": inference, "metrics": metrics, "parity": parity, "size_bytes": dir_size(out), "onnx_bytes": (out / "model.onnx").stat().st_size, **meta,
        })
        return metrics


PLUGIN = SklearnPlugin()
