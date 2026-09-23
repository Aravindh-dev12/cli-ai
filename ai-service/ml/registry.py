from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.getenv("MODEL_REGISTRY_ROOT", "/cache/clinevo-owned/registry"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def register(checkpoint: str, metrics: dict, dataset_version: str, model_version: str) -> Path:
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_version": model_version,
        "artifact": path.name,
        "sha256": sha256_file(path),
        "dataset_version": dataset_version,
        "metrics": metrics,
        "git_commit": os.getenv("GIT_COMMIT"),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    target = ROOT / f"{model_version}.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def promote(
    model_version: str,
    min_accuracy: float,
    max_p95_ms: float,
    min_macro_f1: float = 0.0,
    max_ece: float = 1.0,
    baseline_max: float | None = None,
) -> Path:
    path = ROOT / f"{model_version}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    metrics = manifest["metrics"]

    accuracy = float(metrics.get("route_accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    ece = float(metrics.get("ece", 1.0))
    p95 = float(metrics.get("p95_latency_ms", float("inf")))

    if accuracy < min_accuracy:
        raise RuntimeError(f"promotion gate failed: route accuracy {accuracy:.4f} < {min_accuracy:.4f}")
    if macro_f1 < min_macro_f1:
        raise RuntimeError(f"promotion gate failed: macro F1 {macro_f1:.4f} < {min_macro_f1:.4f}")
    if ece > max_ece:
        raise RuntimeError(f"promotion gate failed: ECE {ece:.4f} > {max_ece:.4f}")
    if p95 > max_p95_ms:
        raise RuntimeError(f"promotion gate failed: p95 latency {p95:.1f} ms > {max_p95_ms:.1f} ms")
    if baseline_max is not None and accuracy <= baseline_max:
        raise RuntimeError("promotion gate failed: route accuracy does not exceed declared baseline")

    latest = ROOT / "latest.json"
    latest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return latest
