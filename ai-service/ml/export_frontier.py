from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

def main() -> None:
    parser = argparse.ArgumentParser(description='Export ClinevoOne-Frontier adapter and Hub metadata')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='/cache/clinevo-owned/hub')
    parser.add_argument('--model-id', default='answerdotai/ModernBERT-large')
    parser.add_argument('--repo-id', default='Aravindhan11/clinevo-one-frontier')
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    weights = payload.get('model', payload.get('adapter'))
    if weights is None:
        raise ValueError('checkpoint does not contain model/adapter weights')

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in weights.items() if torch.is_tensor(v)}, str(output / 'clinevo_frontier_adapter.safetensors'))
    config = {
        'model_type': 'clinevo-one-frontier',
        'base_model': args.model_id,
        'architecture': 'ModernBERT-large + audio-STFT-CNN + accelerometer-CNN-BiGRU + typed decision heads',
        'typed_decisions': ['noul', 'score', 'choice', 'action'],
        'action_space': ['enrich', 'review', 'escalate', 'hold'],
        'checkpoint': payload.get('model_version'),
        'adapter_only': bool(payload.get('adapter_only', True)),
        'training_method': payload.get('training_method'),
    }
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    card = '''---
library_name: transformers
license: apache-2.0
tags:
- system-one
- calibrated-decisions
- multimodal
- audio-classification
- accelerometer
- reinforcement-learning
- rlcd
- lora
pipeline_tag: text-classification
base_model:
- answerdotai/ModernBERT-large
---

# ClinevoOne-Frontier

ClinevoOne-Frontier is an in-house multimodal typed-decision model inspired by the software-facing decision interface described by TypeSafe System One and the open Clinevo/Laya architecture.

It combines a ModernBERT-large text encoder with audio STFT/CNN and accelerometer CNN/BiGRU encoders, then emits typed probabilistic decisions instead of free-form prose.

This repository contains an adapter checkpoint. The upstream ModernBERT model remains a dependency and keeps its own license/model card.

## Decision types

- noul: calibrated boolean probability
- score: expected ordinal level plus distribution
- choice: request-time option scoring
- action: bounded agent-action recommendation

Allowed agent actions: enrich, review, escalate, hold. The model must not be used to make an autonomous terminal pharmacovigilance decision.

## Training

The training stack supports supervised fine-tuning and an RLCD-style stage using proper-scoring rewards. The RL stage is designed to improve probability quality, not only argmax accuracy.

## Data status

Synthetic fixtures are engineering-only regression data. They are not evidence of veterinary efficacy, pharmacovigilance validity, or superiority over an external model.

## Deployment

Pair the adapter with the Clinevo runtime and human-review controls. Production promotion requires representative subject-held-out validation, calibration, latency, and safety gates.
'''
    (output / 'README.md').write_text(card, encoding='utf-8')
    shutil.copy2(Path(__file__).resolve().parents[1] / 'app' / 'frontier_model.py', output / 'frontier_model.py')
    print(json.dumps({'output': str(output), 'repo_id': args.repo_id, 'adapter': 'clinevo_frontier_adapter.safetensors'}, indent=2))

if __name__ == '__main__':
    main()
