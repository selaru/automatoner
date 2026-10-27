"""Core ML (.mlpackage) for iOS and macOS apps. Built from the tiny PyTorch autoencoder.

Not offered, and why:
- scikit-learn and LightGBM: coremltools' scikit-learn converter lags behind current scikit-learn. Use ONNX Runtime instead.
- Hugging Face text models: coremltools 9 has only been tested up to torch 2.7 and fails to trace BERT-style models on newer torch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..base import ExportContext, Exporter, ExportError, copy, describe_files, write, zip_dir

SWIFT_TABULAR = '''# {title} for Core ML

`model.mlpackage` takes a Float32 array `input` of shape [N, {dim}] and returns `{out}`.
Build the input from `features.json`: for each numeric column (value - mean) / std (median when missing),
then a one-hot block for each categorical column in the order listed.

```swift
import CoreML
let model = try MLModel(contentsOf: try MLModel.compileModel(at: modelURL))
let x = try MLMultiArray(shape: [1, {dim}], dataType: .float32)
// fill x from the feature vector
let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input": x]))
let rawScore = out.featureValue(for: "{out}")!.multiArrayValue![0].doubleValue
// score = clamp((sign * rawScore - lo) / (hi - lo), 0, 1); alert when score >= threshold (inference.json)
```
'''


class CoreML(Exporter):
    id = "coreml"
    name = "Core ML (iOS and macOS)"
    blurb = "An .mlpackage for apps. Available for the tiny autoencoder."
    host = ("darwin",)
    requires = ("coremltools",)

    def applicable(self, manifest: dict[str, Any], category: str) -> tuple[bool, str]:
        arts = manifest.get("artifacts", {})
        if "torchscript" in arts:
            return True, ""
        if category == "text_classification":
            return False, "coremltools 9 cannot yet convert Transformers models on current PyTorch. Use the ONNX export."
        if manifest.get("trainer") in ("sklearn", "lightgbm"):
            return False, "coremltools cannot convert current scikit-learn and LightGBM models. Use the ONNX or Lambda exports."
        return False, "no PyTorch artifact to convert"

    def export(self, ctx: ExportContext) -> dict[str, Any]:
        import coremltools as ct
        import numpy as np
        import torch

        pkg = ctx.out_dir / "package"
        pkg.mkdir(parents=True)
        ts = torch.jit.load(str(ctx.artifact_path("torchscript")))
        feats = json.loads(ctx.artifact_path("features").read_text(encoding="utf-8"))
        dim = len(feats["names"])
        ml = ct.convert(ts, inputs=[ct.TensorType(name="input", shape=(ct.RangeDim(1, 4096), dim), dtype=np.float32)],
                        outputs=[ct.TensorType(name="scores")], convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17)
        ml.save(str(pkg / "model.mlpackage"))
        copy(ctx.artifact_path("features"), pkg / "features.json")
        copy(ctx.artifact_path("inference"), pkg / "inference.json")
        write(pkg / "README.md", SWIFT_TABULAR.format(title=ctx.job["model_name"], dim=dim, out="scores"))
        sanity = ml.predict({"input": np.zeros((2, dim), dtype=np.float32)})
        ctx.log(f"prediction check ok: {list(sanity)}")
        zip_dir(pkg, ctx.out_dir.parent / f"{ctx.out_dir.name}.zip")
        return {"kind": "folder+zip", "root": "package", "download": f"{ctx.out_dir.name}.zip", "location": str(pkg), "files": describe_files(pkg)[:40]}


EXPORTERS = [CoreML()]
