"""Exporters for ONNX tabular models: Lambda zip, Lambda container, Docker server, browser, joblib, plain ONNX."""
from __future__ import annotations

import json
import re
from typing import Any

from ...util import slug
from .. import templates as T
from ..base import ExportContext, Exporter, ExportError, copy, describe_files, sha256, write, zip_dir

TABULAR_CATEGORIES = ("tabular_anomaly", "tabular_classification")


def _sample_event(ctx: ExportContext) -> dict[str, Any]:
    f = json.loads(ctx.artifact_path("features").read_text(encoding="utf-8"))
    rec: dict[str, Any] = {c: round(f["mean"][c], 4) for c in f["numeric"]}
    for c, vals in f["categorical"].items():
        rec[c] = vals[0]
    return {"record": rec}


def _metrics_block(ctx: ExportContext) -> str:
    m = (ctx.manifest.get("metrics") or {}).get("test") or {}
    keep = {k: v for k, v in m.items() if isinstance(v, (int, float)) and k in ("auroc", "precision_at_k", "threshold_f1", "accuracy", "macro_f1")}
    if not keep:
        return ""
    return "Test metrics: " + ", ".join(f"{k} {v:.3f}" for k, v in keep.items()) + "."


def _fn_name(ctx: ExportContext) -> str:
    return slug(ctx.options.get("function_name") or f"{ctx.job['model_id']}-{ctx.job['dataset_name']}")[:60]


def _copy_runtime_files(ctx: ExportContext, dest) -> None:
    copy(ctx.artifact_path("onnx"), dest / "model.onnx")
    copy(ctx.artifact_path("features"), dest / "features.json")
    copy(ctx.artifact_path("inference"), dest / "inference.json")
    write(dest / "handler.py", T.HANDLER)
    write(dest / "requirements.txt", T.LAMBDA_REQUIREMENTS)
    write(dest / "sample_event.json", json.dumps(_sample_event(ctx), indent=2))
    write(dest / "test_local.py", T.TEST_LOCAL)


class LambdaZip(Exporter):
    id = "lambda_zip"
    name = "AWS Lambda (zip)"
    blurb = "handler, ONNX model, SAM and Terraform snippets, arm64. Add onnxruntime as a layer."
    needs_artifacts = ("onnx", "features", "inference")
    categories = TABULAR_CATEGORIES
    options = (
        {"key": "function_name", "label": "Function name", "type": "text", "default": ""},
        {"key": "memory", "label": "Memory (MB)", "type": "int", "default": 256},
    )

    def export(self, ctx: ExportContext) -> dict[str, Any]:
        fn = _fn_name(ctx)
        memory = int(ctx.options.get("memory") or 256)
        pkg = ctx.out_dir / "package"
        _copy_runtime_files(ctx, pkg)
        zip_name = f"{fn}.zip"
        code = ctx.out_dir / "code"
        for name in ("handler.py", "model.onnx", "features.json", "inference.json"):
            copy(pkg / name, code / name)
        zip_dir(code, ctx.out_dir / zip_name)
        write(pkg / "template.yaml", T.SAM_TEMPLATE.format(description=f"{ctx.job['model_name']} scorer", logical=re.sub(r"[^A-Za-z0-9]", "", fn.title()) or "Scorer",
                                                          function_name=fn, memory=memory))
        write(pkg / "main.tf", T.TERRAFORM.format(tf_name=fn.replace("-", "_"), function_name=fn, zip_name=zip_name, memory=memory))
        features = json.loads((pkg / "features.json").read_text(encoding="utf-8"))
        write(pkg / "README.md", T.LAMBDA_README.format(title=f"{fn} for AWS Lambda", job_id=ctx.job["id"], model_name=ctx.job["model_name"],
                                                        dataset_name=ctx.job["dataset_name"], metrics_block=_metrics_block(ctx), zip_name=zip_name,
                                                        columns=", ".join(features["numeric"] + list(features["categorical"]))))
        copy(ctx.out_dir / zip_name, pkg / zip_name)
        (ctx.out_dir / zip_name).unlink()
        shutil_rmtree(code)
        zip_dir(pkg, ctx.out_dir.parent / f"{ctx.out_dir.name}.zip")
        return {"kind": "folder+zip", "root": "package", "download": f"{ctx.out_dir.name}.zip", "files": describe_files(pkg),
                "smoke_test": "cd package && pip install -r requirements.txt && python test_local.py", "sha256": sha256(pkg / zip_name)}


class LambdaContainer(Exporter):
    id = "lambda_container"
    name = "AWS Lambda (container image)"
    blurb = "Dockerfile on the Lambda arm64 base image with the model baked in. No layers needed."
    needs_artifacts = ("onnx", "features", "inference")
    categories = TABULAR_CATEGORIES
    tools = ("docker",)

    def export(self, ctx: ExportContext) -> dict[str, Any]:
        pkg = ctx.out_dir / "package"
        _copy_runtime_files(ctx, pkg)
        write(pkg / "Dockerfile", T.LAMBDA_DOCKERFILE)
        write(pkg / "README.md", f"# Lambda container\n\n```\ndocker buildx build --platform linux/arm64 -t {_fn_name(ctx)} .\n```\n\nPush to ECR and create a function from the image (architecture arm64).\n\n{_metrics_block(ctx)}\n")
        zip_dir(pkg, ctx.out_dir.parent / f"{ctx.out_dir.name}.zip")
        return {"kind": "folder+zip", "root": "package", "download": f"{ctx.out_dir.name}.zip", "files": describe_files(pkg),
                "smoke_test": f"cd package && docker buildx build --platform linux/arm64 -t {_fn_name(ctx)} ."}


class DockerServer(Exporter):
    id = "docker"
    name = "Docker (HTTP server)"
    blurb = "FastAPI server with POST /predict, for ECS, Kubernetes or any machine with Docker."
    needs_artifacts = ("onnx", "features", "inference")
    categories = TABULAR_CATEGORIES
    tools = ("docker",)

    def export(self, ctx: ExportContext) -> dict[str, Any]:
        pkg = ctx.out_dir / "package"
        _copy_runtime_files(ctx, pkg)
        write(pkg / "server.py", T.SERVER_PY.format(title=_fn_name(ctx)))
        write(pkg / "Dockerfile", T.SERVER_DOCKERFILE)
        write(pkg / "README.md", f"# Docker server\n\n```\ndocker build -t {_fn_name(ctx)} .\ndocker run -p 8080:8080 {_fn_name(ctx)}\ncurl -s localhost:8080/predict -H 'content-type: application/json' -d @sample_event.json\n```\n")
        zip_dir(pkg, ctx.out_dir.parent / f"{ctx.out_dir.name}.zip")
        return {"kind": "folder+zip", "root": "package", "download": f"{ctx.out_dir.name}.zip", "files": describe_files(pkg),
                "smoke_test": f"cd package && docker build -t {_fn_name(ctx)} ."}


class OnnxWeb(Exporter):
    id = "onnx_web"
    name = "ONNX for browsers"
    blurb = "The model plus an index.html that scores a record with onnxruntime-web."
    needs_artifacts = ("onnx", "features", "inference")
    categories = TABULAR_CATEGORIES

    def export(self, ctx: ExportContext) -> dict[str, Any]:
        pkg = ctx.out_dir / "package"
        copy(ctx.artifact_path("onnx"), pkg / "model.onnx")
        copy(ctx.artifact_path("features"), pkg / "features.json")
        copy(ctx.artifact_path("inference"), pkg / "inference.json")
        sample = json.dumps(_sample_event(ctx)["record"], indent=2)
        write(pkg / "index.html", T.BROWSER_HTML.format(title=_fn_name(ctx), sample=sample))
        zip_dir(pkg, ctx.out_dir.parent / f"{ctx.out_dir.name}.zip")
        return {"kind": "folder+zip", "root": "package", "download": f"{ctx.out_dir.name}.zip", "files": describe_files(pkg),
                "smoke_test": "cd package && python -m http.server 8000  # open http://localhost:8000"}


class Joblib(Exporter):
    id = "joblib"
    name = "Python bundle (joblib)"
    blurb = "model.joblib plus predict.py. For any Python host that has scikit-learn."
    needs_artifacts = ("joblib",)
    categories = TABULAR_CATEGORIES

    def export(self, ctx: ExportContext) -> dict[str, Any]:
        pkg = ctx.out_dir / "package"
        copy(ctx.artifact_path("joblib"), pkg / "model.joblib")
        extra = "lightgbm" if ctx.manifest.get("algorithm") == "lightgbm" else ""
        write(pkg / "predict.py", T.JOBLIB_PREDICT.replace("{extra_pkgs}", extra))
        write(pkg / "README.md", "# Python bundle\n\n```\necho '{\"records\": [{...}]}' | python predict.py\n```\n")
        zip_dir(pkg, ctx.out_dir.parent / f"{ctx.out_dir.name}.zip")
        return {"kind": "folder+zip", "root": "package", "download": f"{ctx.out_dir.name}.zip", "files": describe_files(pkg),
                "smoke_test": "cd package && echo '{\"records\": []}' | python predict.py"}


class PlainOnnx(Exporter):
    id = "onnx"
    name = "ONNX file"
    blurb = "Just model.onnx with the feature and inference descriptions."
    needs_artifacts = ("onnx",)

    def export(self, ctx: ExportContext) -> dict[str, Any]:
        pkg = ctx.out_dir / "package"
        for k in ("onnx", "features", "inference"):
            if k in ctx.artifacts:
                copy(ctx.artifact_path(k), pkg / ctx.artifacts[k])
        zip_dir(pkg, ctx.out_dir.parent / f"{ctx.out_dir.name}.zip")
        return {"kind": "folder+zip", "root": "package", "download": f"{ctx.out_dir.name}.zip", "files": describe_files(pkg)}


def shutil_rmtree(p) -> None:
    import shutil
    shutil.rmtree(p, ignore_errors=True)


EXPORTERS = [LambdaZip(), LambdaContainer(), DockerServer(), OnnxWeb(), Joblib(), PlainOnnx()]
