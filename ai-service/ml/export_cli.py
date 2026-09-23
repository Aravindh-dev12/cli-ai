from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

def main() -> None:
    parser = argparse.ArgumentParser(description='Export CLI ClinevoOne-v1 model for Hugging Face')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='/tmp/hf-cli')
    parser.add_argument('--repo-id', default='Aravindhan11/cli')
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    weights = payload.get('model')
    if weights is None:
        raise ValueError('checkpoint has no model weights')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'model_version': payload.get('model_version'), 'architecture': 'ClinevoOne-v1', 'model': weights}, output / 'clinevo_one.pt')
    config = {
        'model_type': 'clinevo-one-v1',
        'architecture': 'text-multiscale-cnn-bigru + audio-stft-cnn + accelerometer-cnn-bigru + gated-fusion',
        'typed_decisions': ['noul', 'score', 'choice'],
        'training': payload.get('training', {}),
        'dataset_version': payload.get('dataset_version'),
        'model_version': payload.get('model_version'),
        'repository_id': args.repo_id,
    }
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    card = '''---
license: apache-2.0
tags:
- multimodal
- audio-classification
- accelerometer
- lora
- typed-decisions
- system-one
pipeline_tag: text-classification
---

# CLI

CLI is an in-house multimodal typed-decision research model. This initial Hub artifact is ClinevoOne-v1, using a compact custom neural architecture rather than a pretrained language backbone.

## Modalities

- text: multi-scale convolution + bidirectional GRU
- audio: STFT + convolutional encoder
- accelerometer: 3-axis + magnitude, convolution + bidirectional GRU
- fusion: gated multimodal state

## Outputs

- NOUL: probability for boolean decisions
- score: four-level ordinal distribution
- choice: request-time option classification

## Safety and evaluation

This artifact is an engineering/research release. Synthetic fixtures are not evidence of veterinary efficacy or pharmacovigilance validity. The model must not make an autonomous terminal pharmacovigilance disposition.
Production use requires representative subject-held-out evaluation, calibration, latency, security, and human-review validation.
'''
    (output / 'README.md').write_text(card, encoding='utf-8')
    print(json.dumps({'output': str(output), 'repo_id': args.repo_id, 'file': 'clinevo_one.pt'}, indent=2))

if __name__ == '__main__':
    main()
