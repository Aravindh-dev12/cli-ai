#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.registry import promote, register


def main() -> None:
    parser = argparse.ArgumentParser(description="ClinevoOne model registry operations")
    sub = parser.add_subparsers(dest="cmd", required=True)

    register_parser = sub.add_parser("register")
    register_parser.add_argument("--checkpoint", required=True)
    register_parser.add_argument("--metrics", required=True)
    register_parser.add_argument("--dataset-version", required=True)
    register_parser.add_argument("--model-version", required=True)

    promote_parser = sub.add_parser("promote")
    promote_parser.add_argument("--model-version", required=True)
    promote_parser.add_argument("--min-accuracy", type=float, default=0.80)
    promote_parser.add_argument("--min-macro-f1", type=float, default=0.80)
    promote_parser.add_argument("--max-ece", type=float, default=0.15)
    promote_parser.add_argument("--max-p95-ms", type=float, default=1500.0)
    promote_parser.add_argument("--baseline-max", type=float, default=None)

    args = parser.parse_args()
    if args.cmd == "register":
        metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
        print(register(args.checkpoint, metrics, args.dataset_version, args.model_version))
        return

    print(
        promote(
            model_version=args.model_version,
            min_accuracy=args.min_accuracy,
            max_p95_ms=args.max_p95_ms,
            min_macro_f1=args.min_macro_f1,
            max_ece=args.max_ece,
            baseline_max=args.baseline_max,
        )
    )


if __name__ == "__main__":
    main()
