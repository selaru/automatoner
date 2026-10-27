"""Shared pieces for the scikit-learn, LightGBM and tiny PyTorch trainers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import TrainContext
from .features import FeatureSpec


def load_splits(ctx: TrainContext) -> dict[str, pd.DataFrame]:
    d = ctx.dataset_dir
    return {p: pd.read_csv(d / f"{p}.csv") for p in ("train", "valid", "test") if (d / f"{p}.csv").exists()}


def build_features(ctx: TrainContext, train: pd.DataFrame) -> FeatureSpec:
    ds = ctx.dataset
    return FeatureSpec.fit(train, ds["numeric_columns"], ds["categorical_columns"])


def threshold_by_f1(raw_valid: np.ndarray, y_valid: np.ndarray) -> float | None:
    if y_valid is None or len(set(y_valid.tolist())) < 2:
        return None
    best, best_f1 = None, -1.0
    for t in np.unique(np.quantile(raw_valid, np.linspace(0.5, 0.999, 200))):
        pred = raw_valid >= t
        tp = float((pred & (y_valid == 1)).sum()); fp = float((pred & (y_valid == 0)).sum()); fn = float(((~pred) & (y_valid == 1)).sum())
        f1 = 2 * tp / max(1.0, 2 * tp + fp + fn)
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return best


def anomaly_metrics(raw: np.ndarray, y: np.ndarray | None, threshold: float | None) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    out: dict[str, Any] = {"n": int(len(raw))}
    if y is None or len(set(y.tolist())) < 2:
        out["note"] = "no labels in the test split; only score distribution is available"
        out["score_p50"] = float(np.median(raw)); out["score_p99"] = float(np.quantile(raw, 0.99))
        return out
    out["auroc"] = float(roc_auc_score(y, raw))
    out["average_precision"] = float(average_precision_score(y, raw))
    k = int(y.sum())
    top = np.argsort(-raw)[:k]
    out["precision_at_k"] = float(y[top].mean())
    out["k"] = k
    if threshold is not None:
        pred = raw >= threshold
        tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum()); fn = float(((~pred) & (y == 1)).sum())
        out.update({"threshold_precision": tp / max(1.0, tp + fp), "threshold_recall": tp / max(1.0, tp + fn),
                    "threshold_f1": 2 * tp / max(1.0, 2 * tp + fp + fn)})
    return out


def calibrate(raw_train: np.ndarray, threshold_raw: float) -> tuple[float, float]:
    """Map raw anomaly scores to 0..1 so the alert threshold is exactly 0.5.

    0 is the typical normal score (the median on training data); the scale saturates at 1 for scores twice as far
    above normal as the threshold is. Returns (lo, hi).
    """
    lo = float(np.median(raw_train))
    gap = max(float(threshold_raw) - lo, 1e-9)
    return lo, lo + 2.0 * gap


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None, classes: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
    out: dict[str, Any] = {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
                           "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(classes))), zero_division=0)
    out["per_class"] = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(classes)}
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()
    if proba is not None and len(classes) == 2 and len(set(y_true.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y_true, proba[:, 1]))
    return out


def onnx_parity(onnx_bytes: bytes, X: np.ndarray, expected: np.ndarray, output_name: str) -> dict[str, Any]:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    got = sess.run([output_name], {sess.get_inputs()[0].name: X.astype(np.float32)})[0]
    got = np.asarray(got, dtype=float).reshape(expected.shape)
    return {"output": output_name, "outputs": names, "max_abs_diff": float(np.max(np.abs(got - expected))), "rows": int(len(X))}


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
