import json

from ml import registry


def test_register_normalizes_nested_metrics_and_promotes(tmp_path, monkeypatch):
    root = tmp_path / "registry"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    monkeypatch.setattr(registry, "ROOT", root)

    path = registry.register(
        str(checkpoint),
        {"model_version": "wrapped", "metrics": {
            "route_accuracy": 0.91,
            "macro_f1": 0.90,
            "ece": 0.08,
            "p95_latency_ms": 120.0,
        }},
        "synthetic-v1-test",
        "test-model",
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["metrics"]["route_accuracy"] == 0.91
    assert "metrics" not in manifest["metrics"]

    latest = registry.promote(
        "test-model",
        min_accuracy=0.80,
        max_p95_ms=1500.0,
        min_macro_f1=0.80,
        max_ece=0.15,
    )
    assert latest == root / "latest.json"
