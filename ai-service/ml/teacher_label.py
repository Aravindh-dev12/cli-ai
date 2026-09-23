from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.decision_engine import INBOX_DECISION_QUESTIONS, JevProvider, LayaProvider
from ml.synthetic_dataset import build_dataset


def _provider(name: str):
    if name == "laya":
        return LayaProvider()
    if name == "jev":
        if not os.getenv("JEV_API_KEY", "").strip():
            raise RuntimeError("JEV_API_KEY is required for Jev teacher labeling")
        return JevProvider()
    raise ValueError(f"unsupported teacher: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher decision records for ClinevoOne distillation")
    parser.add_argument("--teachers", default="laya")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--out", default="/cache/clinevo-owned/teacher-decisions.jsonl")
    args = parser.parse_args()

    question = {"route": INBOX_DECISION_QUESTIONS["route"]}
    providers = [(name.strip(), _provider(name.strip())) for name in args.teachers.split(",") if name.strip()]
    data = build_dataset(args.size, seed=args.seed)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in data:
            merged: dict[str, float] = {}
            sources = {}
            for name, provider in providers:
                _, answers, routing = provider.predict(item.text, question)
                answer = answers["route"]
                probabilities = answer.get("probabilities") or {}
                total = sum(float(value) for value in probabilities.values())
                if total <= 0:
                    continue
                normalized = {str(key): float(value) / total for key, value in probabilities.items()}
                sources[name] = {"routing": routing, "confidence": answer.get("confidence")}
                for key, value in normalized.items():
                    merged[key] = merged.get(key, 0.0) + value
            if not merged:
                continue
            divisor = max(len(sources), 1)
            merged = {key: value / divisor for key, value in merged.items()}
            handle.write(json.dumps({
                "text": item.text,
                "label": item.labels["route"],
                "route_probabilities": merged,
                "teachers": sources,
                "dataset_version": f"synthetic-v1-{args.size}-{args.seed}",
            }, ensure_ascii=False) + "\n")

    print(json.dumps({"output": str(output), "rows": len(data), "teachers": [name for name, _ in providers]}, indent=2))


if __name__ == "__main__":
    main()
