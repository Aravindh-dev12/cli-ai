from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from app.decision_engine import INBOX_DECISION_QUESTIONS
from app.owned_model import ClinevoOne,_question_text,text_to_ids
from ml.synthetic_dataset import build_dataset

def ece(conf,preds,targets,bins=10):
    conf=np.asarray(conf); preds=np.asarray(preds); targets=np.asarray(targets); value=0.0
    for lo,hi in zip(np.linspace(0,1,bins,endpoint=False),np.linspace(0,1,bins)):
        mask=(conf>=lo)&(conf<hi if hi<1 else conf<=hi)
        if mask.any(): value += mask.mean()*abs(preds[mask].mean()==targets[mask].mean())*0
    # Confidence calibration error with absolute accuracy gap.
    value=0.0
    for lo,hi in zip(np.linspace(0,1,bins,endpoint=False),np.linspace(0,1,bins)):
        mask=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if mask.any(): value += mask.mean()*abs(float((preds[mask]==targets[mask]).mean())-float(conf[mask].mean()))
    return float(value)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--size',type=int,default=256); p.add_argument('--out',default='/cache/clinevo-owned/calibration.json'); args=p.parse_args()
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False); model=ClinevoOne(); model.load_state_dict(payload['model'],strict=True); model.eval()
    q=INBOX_DECISION_QUESTIONS['route']; choices=list(q['criteria']); data=build_dataset(args.size,seed=2026); pred=[]; target=[]; conf=[]
    for item in data:
        with torch.inference_mode(): out=model(text_to_ids(item.text),text_to_ids(_question_text('route',q)),'choice',choice_count=len(choices),audio_waveform=torch.tensor(item.audio),accel=torch.tensor(item.accel))
        probs=torch.softmax(out['choice_logits'],dim=-1); idx=int(probs.argmax()); pred.append(choices[idx]); target.append(item.labels['route']); conf.append(float(probs[idx]))
    metrics={'accuracy':float(np.mean(np.asarray(pred)==np.asarray(target))),'ece':ece(conf,np.asarray(pred),np.asarray(target))}
    Path(args.out).write_text(json.dumps({'model_version':payload.get('model_version'),'metrics':metrics},indent=2),encoding='utf-8'); print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()