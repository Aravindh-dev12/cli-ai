from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from app.decision_engine import INBOX_DECISION_QUESTIONS
from app.owned_model import ClinevoOne, _question_text, text_to_ids
from ml.synthetic_dataset import build_dataset, split_dataset


def _ece(confidence, correct, bins=10):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        if mask.any():
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ClinevoOne on a frozen synthetic held-out suite")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--out", default="/cache/clinevo-owned/metrics.json")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ClinevoOne()
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    data = build_dataset(args.size, seed=991)
    question = INBOX_DECISION_QUESTIONS["route"]
    choices = list(question["criteria"])
    correct = 0
    latencies = []
    confidences = []
    correctness = []
    confusion = {choice: {other: 0 for other in choices} for choice in choices}

    for item in data:
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(
                text_to_ids(item.text),
                text_to_ids(_question_text("route", question)),
                "choice",
                choice_count=len(choices),
                audio_waveform=torch.tensor(item.audio),
                accel=torch.tensor(item.accel),
            )
        probs = torch.softmax(output["choice_logits"], dim=-1)
        idx = int(probs.argmax())
        pred = choices[idx]
        target = item.labels["route"]
        is_correct = int(pred == target)
        correct += is_correct
        confidences.append(float(probs[idx]))
        correctness.append(is_correct)
        confusion[target][pred] += 1
        latencies.append((time.perf_counter() - started) * 1000)

    per_class_f1 = {}
    for choice in choices:
        tp = confusion[choice][choice]
        fp = sum(confusion[other][choice] for other in choices if other != choice)
        fn = sum(confusion[choice][other] for other in choices if other != choice)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        per_class_f1[choice] = 2 * precision * recall / max(precision + recall, 1e-9)

    route_accuracy = correct / max(len(data), 1)
    metrics = {
        "route_accuracy": route_accuracy,
        "macro_f1": float(np.mean(list(per_class_f1.values()))),
        "per_class_f1": per_class_f1,
        "ece": _ece(confidences, correctness),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "mean_latency_ms": float(np.mean(latencies)),
        "parameter_count": float(sum(p.numel() for p in model.parameters())),
        "dataset_version": payload.get("dataset_version"),
        "split": "subject-held-out",
        "test_subject_count": len({item.subject_id for item in data}),
    }
    result = {"model_version": payload.get("model_version"), "metrics": metrics}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
