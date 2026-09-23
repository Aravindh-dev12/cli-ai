from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

TEXT_BUCKETS = 16384
TEXT_EMBED_DIM = 192
HIDDEN = 256
AUDIO_RATE = 16000

def _hash_token(value: str) -> int:
    digest = hashlib.blake2b(value.encode('utf-8', 'ignore'), digest_size=8).digest()
    return int.from_bytes(digest, 'little') % TEXT_BUCKETS

def text_to_ids(text: str, max_bytes: int = 16384) -> Tensor:
    raw = text.encode('utf-8', 'ignore')[:max_bytes]
    tokens, current = [], []
    for byte in raw:
        if byte in b' \t\r\n.,;:!?/\\()[]{}<>|':
            if current:
                tokens.append(_hash_token(bytes(current).decode('utf-8', 'ignore')))
                current.clear()
        else:
            current.append(byte)
    if current:
        tokens.append(_hash_token(bytes(current).decode('utf-8', 'ignore')))
    if not tokens:
        tokens = [_hash_token('<empty>')]
    return torch.tensor(tokens, dtype=torch.long)

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        update = self.dropout(x) @ self.lora_a.t() @ self.lora_b.t() * self.scaling
        return self.base(x) + update

    def freeze_base(self) -> None:
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TEXT_BUCKETS, TEXT_EMBED_DIM)
        self.convs = nn.ModuleList([nn.Conv1d(TEXT_EMBED_DIM, 96, kernel_size=k, padding=k // 2) for k in (3, 5, 7)])
        self.norm = nn.LayerNorm(288)
        self.gru = nn.GRU(288, HIDDEN // 2, batch_first=True, bidirectional=True)
        self.proj = LoRALinear(HIDDEN, HIDDEN, rank=16, alpha=32)

    def forward(self, ids: Tensor) -> Tensor:
        x = self.embedding(ids.unsqueeze(0) if ids.dim() == 1 else ids).transpose(1, 2)
        x = torch.cat([torch.nn.functional.gelu(conv(x)) for conv in self.convs], dim=1).transpose(1, 2)
        x = self.norm(x)
        x, _ = self.gru(x)
        return torch.tanh(self.proj(x.mean(dim=1)))

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.GELU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.proj = LoRALinear(128 * 4 * 4, HIDDEN, rank=16, alpha=32)

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.numel() == 0:
            return torch.zeros(HIDDEN, device=self.proj.base.weight.device)
        waveform = waveform.flatten().float()[: AUDIO_RATE * 12]
        n_fft, hop = 512, 160
        window = torch.hann_window(n_fft, device=waveform.device)
        spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        x = self.net(torch.log1p(spec.abs()).unsqueeze(0).unsqueeze(0))
        return torch.tanh(self.proj(x.flatten()))

class AccelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3), nn.GELU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 5, padding=2), nn.GELU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 192, 3, padding=1), nn.GELU())
        self.gru = nn.GRU(192, HIDDEN // 2, batch_first=True, bidirectional=True)
        self.proj = LoRALinear(HIDDEN, HIDDEN, rank=16, alpha=32)

    def forward(self, accel: Tensor) -> Tensor:
        if accel.numel() == 0:
            return torch.zeros(HIDDEN, device=self.proj.base.weight.device)
        x = accel.float()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x[:, :, :3]
        magnitude = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        x = self.conv(torch.cat([x, magnitude], dim=-1).transpose(1, 2)).transpose(1, 2)
        x, _ = self.gru(x)
        return torch.tanh(self.proj(x.mean(dim=1).squeeze(0)))

class ClinevoOne(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = TextEncoder()
        self.audio = AudioEncoder()
        self.accel = AccelEncoder()
        self.fusion_gate = LoRALinear(HIDDEN * 3 + 3, HIDDEN * 2, rank=16, alpha=32)
        self.state_proj = LoRALinear(HIDDEN * 2, HIDDEN, rank=16, alpha=32)
        self.question = TextEncoder()
        self.noul_head = LoRALinear(HIDDEN * 2, 1, rank=8, alpha=16)
        self.score_head = LoRALinear(HIDDEN * 2, 4, rank=8, alpha=16)
        self.choice_head = LoRALinear(HIDDEN * 2, 32, rank=8, alpha=16)

    def encode_state(self, text_ids: Tensor, audio_waveform: Tensor | None = None, accel: Tensor | None = None) -> Tensor:
        text_vec = self.text(text_ids)
        audio_vec = self.audio(audio_waveform) if audio_waveform is not None else torch.zeros_like(text_vec)
        accel_vec = self.accel(accel) if accel is not None else torch.zeros_like(text_vec)
        mask = torch.tensor([1.0, float(audio_waveform is not None and audio_waveform.numel() > 0), float(accel is not None and accel.numel() > 0)], device=text_vec.device)
        fused = torch.cat([text_vec, audio_vec, accel_vec, mask])
        projected = self.fusion_gate(fused)
        return torch.tanh(self.state_proj(torch.sigmoid(projected) * projected))

    def forward(self, text_ids: Tensor, question_ids: Tensor, question_type: str, choice_count: int = 0, audio_waveform: Tensor | None = None, accel: Tensor | None = None):
        state = self.encode_state(text_ids, audio_waveform, accel)
        question = self.question(question_ids)
        pair = torch.cat([state, question], dim=-1)
        if question_type == 'noul':
            return {'noul_logit': self.noul_head(pair).squeeze(-1)}
        if question_type == 'score':
            return {'score_logits': self.score_head(pair)}
        if question_type == 'choice':
            logits = self.choice_head(pair)
            return {'choice_logits': logits[:choice_count] if choice_count else logits}
        raise ValueError(f'Unsupported decision type: {question_type}')

def _state_text(state: Any) -> str:
    if isinstance(state, str): return state
    if isinstance(state, dict):
        return '\n'.join(f'{k}: {v}' for k, v in state.items() if k not in {'audio', 'audio_base64', 'accelerometer'} and v not in (None, ''))
    return json.dumps(state, ensure_ascii=False)

def _question_text(name: str, question: dict[str, Any]) -> str:
    criteria = question.get('criteria')
    if isinstance(criteria, dict):
        criteria = ' '.join(f'{k}: {v}' for k, v in criteria.items())
    elif isinstance(criteria, list):
        criteria = ' '.join(map(str, criteria))
    else:
        criteria = ''
    return f'{name} {question.get("type", "noul")} {question.get("instructions", "")} {criteria}'

def _audio_tensor(state: Any) -> Tensor | None:
    if not isinstance(state, dict): return None
    if state.get('audio') is not None:
        from ml.signal_schema import validate_audio
        return torch.tensor(validate_audio(state['audio']), dtype=torch.float32)
    encoded = state.get('audio_base64')
    if encoded:
        return torch.from_numpy(np.frombuffer(base64.b64decode(encoded), dtype=np.float32).copy())
    return None

def _accel_tensor(state: Any) -> Tensor | None:
    if not isinstance(state, dict) or state.get('accelerometer') is None: return None
    from ml.signal_schema import validate_accelerometer
    return torch.tensor(validate_accelerometer(state['accelerometer']), dtype=torch.float32)

class OwnedModelProvider:
    name = 'owned'
    def __init__(self): self._model, self._version = None, 'unloaded'
    def _checkpoint(self) -> Path: return Path(os.getenv('OWNED_MODEL_CHECKPOINT', '/cache/clinevo-owned/latest.pt'))
    def _load(self) -> ClinevoOne:
        if self._model is not None: return self._model
        path = self._checkpoint()
        if not path.exists(): raise RuntimeError(f'owned model checkpoint not found: {path}; run make owned-train')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        model = ClinevoOne(); model.load_state_dict(payload['model'], strict=True); model.eval()
        self._version = str(payload.get('model_version', path.stem)); self._model = model
        return model
    def predict(self, state: Any, questions: dict[str, Any]):
        model = self._load(); text_ids = text_to_ids(_state_text(state)); audio = _audio_tensor(state); accel = _accel_tensor(state); answers = {}
        with torch.inference_mode():
            for name, question in questions.items():
                qtype = str(question.get('type', 'noul')); qids = text_to_ids(_question_text(name, question))
                output = model(text_ids, qids, qtype, choice_count=len(question.get('criteria', {})) if isinstance(question.get('criteria'), dict) else 0, audio_waveform=audio, accel=accel)
                if qtype == 'noul':
                    p = float(torch.sigmoid(output['noul_logit']).item()); answers[name] = {'type':'noul','noul':p,'confidence':abs(p-0.5)*2}
                elif qtype == 'score':
                    probs = torch.softmax(output['score_logits'], dim=-1); score = float((probs * torch.arange(4)).sum().item())
                    answers[name] = {'type':'score','score':score,'confidence':float(probs.max().item()),'probabilities':{str(i):float(v) for i,v in enumerate(probs.tolist())}}
                else:
                    choices = list(question.get('criteria', {}).keys()) or [f'choice_{i}' for i in range(output['choice_logits'].numel())]
                    probs = torch.softmax(output['choice_logits'], dim=-1); idx = min(int(probs.argmax().item()), len(choices)-1)
                    answers[name] = {'type':'choice','choice':choices[idx],'confidence':float(probs[idx].item()),'probabilities':{choices[i]:float(probs[i].item()) for i in range(min(len(choices), probs.numel()))}}
        return self._version, answers, {'architecture':'ClinevoOne-v1','multimodal':{'text':True,'audio':audio is not None,'accelerometer':accel is not None},'parameters':sum(p.numel() for p in model.parameters()),'low_rank_adaptation':True}

def owned_health() -> dict[str, Any]:
    checkpoint = Path(os.getenv('OWNED_MODEL_CHECKPOINT', '/cache/clinevo-owned/latest.pt'))
    return {'backend':'owned','checkpoint':str(checkpoint),'available':checkpoint.exists(),'architecture':'ClinevoOne-v1','modalities':['text','audio','accelerometer'],'typed_decisions':['noul','score','choice']}

__all__ = ['ClinevoOne','LoRALinear','OwnedModelProvider','owned_health','text_to_ids','_question_text']