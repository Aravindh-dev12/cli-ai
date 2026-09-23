#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ml.registry import register,promote

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    r=sub.add_parser('register'); r.add_argument('--checkpoint',required=True); r.add_argument('--metrics',required=True); r.add_argument('--dataset-version',required=True); r.add_argument('--model-version',required=True)
    q=sub.add_parser('promote'); q.add_argument('--model-version',required=True); q.add_argument('--min-accuracy',type=float,default=0.80); q.add_argument('--max-p95-ms',type=float,default=1500.0); q.add_argument('--baseline-max',type=float,default=None)
    a=p.parse_args()
    if a.cmd=='register':
        metrics=json.loads(Path(a.metrics).read_text(encoding='utf-8')); print(register(a.checkpoint,metrics,a.dataset_version,a.model_version))
    else: print(promote(a.model_version,a.max_p95_ms,a.min_accuracy,a.baseline_max))

if __name__=='__main__': main()