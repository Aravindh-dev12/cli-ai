from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(os.getenv('MODEL_REGISTRY_ROOT','/cache/clinevo-owned/registry'))

def sha256_file(path: Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def register(checkpoint:str, metrics:dict, dataset_version:str, model_version:str)->Path:
    path=Path(checkpoint); ROOT.mkdir(parents=True,exist_ok=True)
    manifest={'model_version':model_version,'artifact':path.name,'sha256':sha256_file(path),'dataset_version':dataset_version,'metrics':metrics,'git_commit':os.getenv('GIT_COMMIT'),'registered_at_utc':datetime.now(timezone.utc).isoformat()}
    target=ROOT/f'{model_version}.json'; target.write_text(json.dumps(manifest,indent=2),encoding='utf-8'); return target

def promote(model_version:str,min_accuracy:float,max_p95_ms:float,baseline_max:float|None=None)->Path:
    path=ROOT/f'{model_version}.json'; manifest=json.loads(path.read_text(encoding='utf-8')); metrics=manifest['metrics']
    if float(metrics.get('route_accuracy',0)) < min_accuracy: raise RuntimeError('promotion gate failed: route accuracy')
    if float(metrics.get('p95_latency_ms',float('inf'))) > max_p95_ms: raise RuntimeError('promotion gate failed: p95 latency')
    if baseline_max is not None and float(metrics.get('route_accuracy',0)) <= baseline_max: raise RuntimeError('promotion gate failed: owned model does not beat benchmark baseline')
    latest=ROOT/'latest.json'; latest.write_text(json.dumps(manifest,indent=2),encoding='utf-8'); return latest