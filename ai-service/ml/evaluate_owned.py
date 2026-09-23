from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from app.decision_engine import INBOX_DECISION_QUESTIONS
from app.owned_model import ClinevoOne, _question_text, text_to_ids
from ml.synthetic_dataset import build_dataset

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--size',type=int,default=256)
    args=parser.parse_args()
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=ClinevoOne(); model.load_state_dict(payload['model'],strict=True); model.eval()
    data=build_dataset(args.size,seed=991); question=INBOX_DECISION_QUESTIONS['route']; choices=list(question['criteria'])
    correct=0; lat=[]
    for item in data:
        t=time.perf_counter()
        with torch.inference_mode():
            out=model(text_to_ids(item.text),text_to_ids(_question_text('route',question)),'choice',choice_count=len(choices),audio_waveform=torch.tensor(item.audio),accel=torch.tensor(item.accel))
        pred=choices[int(torch.softmax(out['choice_logits'],dim=-1).argmax())]; correct += int(pred==item.labels['route']); lat.append((time.perf_counter()-t)*1000)
    metrics={'route_accuracy':correct/len(data),'p95_latency_ms':float(np.percentile(lat,95)),'mean_latency_ms':float(np.mean(lat)),'parameter_count':float(sum(p.numel() for p in model.parameters()))}
    print(json.dumps({'model_version':payload.get('model_version'),'metrics':metrics},indent=2))

if __name__=='__main__': main()