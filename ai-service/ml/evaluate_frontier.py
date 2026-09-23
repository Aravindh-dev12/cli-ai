from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from app.decision_engine import INBOX_DECISION_QUESTIONS
from app.frontier_model import ClinevoOneFrontier
from ml.synthetic_dataset import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ClinevoOne-Frontier on a frozen route suite")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", default="answerdotai/ModernBERT-large")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ClinevoOneFrontier(args.model_id)
    model.load_state_dict(payload.get("model", payload.get("adapter")), strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    question = INBOX_DECISION_QUESTIONS["route"]
    options = list(question["criteria"])
    data = build_dataset(args.size, seed=991)
    correct = 0
    latencies = []

    for item in data:
        state = tokenizer([item.text], return_tensors="pt", padding=True, truncation=True, max_length=args.max_tokens)
        qtext = f"route. {question['instructions']}. " + " ".join(f"{k}: {v}" for k, v in question["criteria"].items())
        q = tokenizer([qtext], return_tensors="pt", padding=True, truncation=True, max_length=args.max_tokens)
        opts = tokenizer(options, return_tensors="pt", padding=True, truncation=True, max_length=args.max_tokens)
        started = time.perf_counter()
        with torch.inference_mode():
            st = model.encode_batch(state["input_ids"], state["attention_mask"], torch.tensor(item.audio), torch.tensor(item.accel))
            qe = model.text_backbone(q["input_ids"], q["attention_mask"])
            oe = model.text_backbone(opts["input_ids"], opts["attention_mask"]).unsqueeze(0)
            out = model.decide_from_state(st, qe, "choice", oe)
        probabilities = torch.softmax(out["choice_logits"], dim=-1)[0]
        pred = options[int(probabilities.argmax())]
        correct += int(pred == item.labels["route"])
        latencies.append((time.perf_counter() - started) * 1000)

    result = {
        "model_version": payload.get("model_version"),
        "route_accuracy": correct / max(len(data), 1),
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "base_model": args.model_id,
        "dataset": "synthetic-route-v1",
    }
    out = Path("/cache/clinevo-owned/frontier-metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
