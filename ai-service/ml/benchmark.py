from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from app.decision_engine import INBOX_DECISION_QUESTIONS, LayaProvider, JevProvider, _deterministic_answers
from app.owned_model import ClinevoOne, _question_text, text_to_ids
from ml.synthetic_dataset import build_dataset
import torch

def _owned_predict(model, text, question):
    import torch
    with torch.inference_mode():
        out=model(text_to_ids(text), text_to_ids(_question_text('route',question)), 'choice', choice_count=len(question['criteria']))
    probs=torch.softmax(out['choice_logits'],dim=-1)
    idx=int(probs.argmax())
    return list(question['criteria'])[idx], float(probs[idx])

def _provider_predict(provider, state, questions):
    _, answers, _ = provider.predict(state, questions)
    answer=answers['route']
    return answer.get('choice'), float(answer.get('confidence',0.0))

def main():
    p=argparse.ArgumentParser(description='Benchmark ClinevoOne against deterministic/Laya/Jev on one frozen text-only suite')
    p.add_argument('--checkpoint',required=True); p.add_argument('--size',type=int,default=256); p.add_argument('--out',default='/cache/clinevo-owned/benchmark.json')
    args=p.parse_args()
    data=build_dataset(args.size,seed=991); question=INBOX_DECISION_QUESTIONS['route']
    results={}
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False); model=ClinevoOne(); model.load_state_dict(payload['model'],strict=True); model.eval()
    providers={'owned':None,'deterministic':None}
    for name in ('laya','jev'):
        if name=='laya': providers[name]=LayaProvider()
        elif os.getenv('JEV_API_KEY'): providers[name]=JevProvider()
    for name, provider in providers.items():
        correct=0; confidences=[]; lat=[]; errors=0
        for item in data:
            start=time.perf_counter()
            try:
                if name=='owned': pred,conf=_owned_predict(model,item.text,question)
                elif name=='deterministic':
                    _,answers,_=_deterministic_answers(item.text); a=answers['route']; pred=a['choice']; conf=float(a['confidence'])
                else: pred,conf=_provider_predict(provider, item.text, {'route':question})
                correct += int(pred==item.labels['route']); confidences.append(conf)
            except Exception:
                errors += 1
            lat.append((time.perf_counter()-start)*1000)
        results[name]={'accuracy':correct/len(data),'mean_confidence':float(np.mean(confidences)) if confidences else 0.0,'p50_latency_ms':float(np.percentile(lat,50)),'p95_latency_ms':float(np.percentile(lat,95)),'error_rate':errors/len(data)}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'dataset':'synthetic-route-v1-heldout','size':len(data),'models':results},indent=2),encoding='utf-8'); print(json.dumps({'dataset':'synthetic-route-v1-heldout','models':results},indent=2))

if __name__=='__main__': main()