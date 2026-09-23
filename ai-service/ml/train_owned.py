from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.decision_engine import INBOX_DECISION_QUESTIONS
from app.owned_model import ClinevoOne, LoRALinear, _question_text, text_to_ids
from ml.synthetic_dataset import Example, build_dataset

def _loss(model, example: Example, name: str, question: dict) -> torch.Tensor:
    qtype = question['type']
    output = model(text_to_ids(example.text), text_to_ids(_question_text(name, question)), qtype,
                   choice_count=len(question.get('criteria', {})) if isinstance(question.get('criteria'), dict) else 0,
                   audio_waveform=torch.tensor(example.audio), accel=torch.tensor(example.accel))
    target = example.labels[name]
    if qtype == 'noul':
        return nn.functional.binary_cross_entropy_with_logits(output['noul_logit'], torch.tensor(float(target)))
    if qtype == 'score':
        return nn.functional.cross_entropy(output['score_logits'].unsqueeze(0), torch.tensor([int(target)]))
    choices = list(question.get('criteria', {}).keys())
    return nn.functional.cross_entropy(output['choice_logits'].unsqueeze(0), torch.tensor([choices.index(target)]))

def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dataset = build_dataset(args.size, args.seed)
    model = ClinevoOne()
    if args.init:
        payload = torch.load(args.init, map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model'], strict=True)
    if args.lora:
        for module in model.modules():
            if isinstance(module, LoRALinear): module.freeze_base()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    questions = INBOX_DECISION_QUESTIONS
    for epoch in range(args.epochs):
        random.shuffle(dataset); losses=[]
        for example in dataset:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack([_loss(model, example, name, q) for name, q in questions.items()]).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
            losses.append(float(loss.detach()))
        print(f'epoch={epoch+1} loss={sum(losses)/len(losses):.6f}')
    version = f'clinevoone-{hashlib.sha256(f"{args.seed}-{args.size}".encode()).hexdigest()[:10]}'
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_version':version,'architecture':'ClinevoOne-v1','dataset_version':f'synthetic-v1-{args.size}-{args.seed}','model':model.state_dict(),'training':vars(args)}, output)
    print(json.dumps({'checkpoint':str(output),'model_version':version,'parameters':sum(p.numel() for p in model.parameters())}, indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Train the in-house ClinevoOne multimodal decision model')
    parser.add_argument('--size',type=int,default=128)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--output',default='/cache/clinevo-owned/latest.pt')
    parser.add_argument('--init',default='')
    parser.add_argument('--lora',action='store_true')
    train(parser.parse_args())