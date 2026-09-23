from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from app.decision_engine import INBOX_DECISION_QUESTIONS, LayaProvider, JevProvider, _deterministic_answers
from app.owned_model import ClinevoOne, _question_text, text_to_ids
from ml.synthetic_dataset import build_dataset


def ece_binary(probs, labels, bins=10):
    if not probs:
        return 0.0
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def main():
    parser = argparse.ArgumentParser(description="Live CLI/ClinevoOne vs Laya vs Jev comparison")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--out", default="/tmp/live-compare.json")
    args = parser.parse_args()

    data = build_dataset(args.size, seed=20260923)
    question = INBOX_DECISION_QUESTIONS["route"]
    labels = list(question["criteria"])
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ClinevoOne().eval()
    model.load_state_dict(payload["model"], strict=True)

    providers = {"cli": None, "laya": LayaProvider()}
    if os.getenv("JEV_API_KEY", "").strip():
        providers["jev"] = JevProvider()

    results = {}
    for name, provider in providers.items():
        preds, confs, truth, latencies = [], [], [], []
        errors = 0
        for item in data:
            started = time.perf_counter()
            try:
                if name == "cli":
                    with torch.inference_mode():
                        out = model(
                            text_to_ids(item.text),
                            text_to_ids(_question_text("route", question)),
                            "choice",
                            choice_count=len(labels),
                        )
                    probabilities = torch.softmax(out["choice_logits"], dim=-1)
                    idx = int(probabilities.argmax())
                    pred, conf = labels[idx], float(probabilities[idx])
                elif name == "laya":
                    _, answers, _ = provider.predict(item.text, {"route": question})
                    answer = answers["route"]
                    pred, conf = answer.get("choice"), float(answer.get("confidence", 0.0))
                else:
                    _, answers, _ = provider.predict(item.text, {"route": question})
                    answer = answers["route"]
                    pred, conf = answer.get("choice"), float(answer.get("confidence", 0.0))
                preds.append(pred)
                confs.append(conf)
                truth.append(int(pred == item.labels["route"]))
            except Exception as exc:
                errors += 1
                print(f"{name} error: {type(exc).__name__}: {exc}")
            latencies.append((time.perf_counter() - started) * 1000)

        accuracy = float(np.mean(truth)) if truth else 0.0
        results[name] = {
            "accuracy": accuracy,
            "error_rate": errors / max(len(data), 1),
            "ece_binary_on_correctness": ece_binary(confs, truth),
            "mean_confidence": float(np.mean(confs)) if confs else 0.0,
            "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else None,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "sample_size": len(data),
        }

    output = {
        "protocol": "cli-vs-laya-vs-jev-live-v1",
        "dataset": {"version": "synthetic-v1", "seed": 20260923, "size": len(data)},
        "models": results,
        "jev_live": bool(os.getenv("JEV_API_KEY", "").strip()),
        "note": "Synthetic engineering test only. Jev is comparison-only and its outputs are not used for training/distillation.",
    }
    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
