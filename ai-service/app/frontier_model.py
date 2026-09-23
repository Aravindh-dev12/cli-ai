from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .owned_model import AccelEncoder, AudioEncoder, LoRALinear, _accel_tensor, _audio_tensor, _state_text

FRONTIER_DEFAULT_MODEL = "answerdotai/ModernBERT-large"
FRONTIER_HIDDEN = 1024
FUSION_HIDDEN = 1024
ACTION_CHOICES = ("enrich", "review", "escalate", "hold")


class FrontierTextBackbone(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("transformers is required for the frontier model") from exc
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.encoder = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        self.hidden_size = int(getattr(self.encoder.config, "hidden_size", FRONTIER_HIDDEN))

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled.float()


class ClinevoOneFrontier(nn.Module):
    """Heavy System-One-style decision model with multimodal state fusion."""

    def __init__(self, model_id: str | None = None):
        super().__init__()
        self.model_id = model_id or os.getenv("FRONTIER_MODEL_ID", FRONTIER_DEFAULT_MODEL)
        self.text_backbone = FrontierTextBackbone(self.model_id)
        hidden = self.text_backbone.hidden_size
        self.audio = AudioEncoder()
        self.accel = AccelEncoder()
        self.audio_proj = LoRALinear(256, hidden, rank=16, alpha=32)
        self.accel_proj = LoRALinear(256, hidden, rank=16, alpha=32)
        self.state_fusion = LoRALinear(hidden * 3 + 3, hidden, rank=32, alpha=64)
        self.question_proj = LoRALinear(hidden, hidden, rank=16, alpha=32)
        self.pair_proj = LoRALinear(hidden * 2, hidden, rank=16, alpha=32)
        self.choice_scorer = LoRALinear(hidden, hidden, rank=16, alpha=32)
        self.choice_value = LoRALinear(hidden, 1, rank=8, alpha=16)
        self.noul_head = LoRALinear(hidden, 1, rank=8, alpha=16)
        self.score_head = LoRALinear(hidden, 4, rank=8, alpha=16)
        self.action_head = LoRALinear(hidden, len(ACTION_CHOICES), rank=8, alpha=16)

    def encode_batch(self, input_ids: Tensor, attention_mask: Tensor, audio_waveform: Tensor | None, accel: Tensor | None) -> Tensor:
        text_vec = self.text_backbone(input_ids, attention_mask)
        device = text_vec.device
        audio_vec = self.audio(audio_waveform).to(device) if audio_waveform is not None else torch.zeros(256, device=device)
        accel_vec = self.accel(accel).to(device) if accel is not None else torch.zeros(256, device=device)
        audio_vec = self.audio_proj(audio_vec.float())
        accel_vec = self.accel_proj(accel_vec.float())
        mask = torch.tensor(
            [
                1.0,
                float(audio_waveform is not None and audio_waveform.numel() > 0),
                float(accel is not None and accel.numel() > 0),
            ],
            device=device,
            dtype=text_vec.dtype,
        )
        fused = torch.cat([text_vec.float(), audio_vec.float(), accel_vec.float(), mask.expand(text_vec.shape[0], -1)], dim=-1)
        return torch.tanh(self.state_fusion(fused))

    def _pair(self, state: Tensor, question: Tensor) -> Tensor:
        return torch.tanh(self.pair_proj(torch.cat([state, question], dim=-1)))

    def decide_from_state(
        self,
        state: Tensor,
        question_embeddings: Tensor,
        question_type: str,
        option_embeddings: Tensor | None = None,
    ) -> dict[str, Tensor]:
        question = torch.tanh(self.question_proj(question_embeddings.float()))
        pair = self._pair(state, question)
        if question_type == "noul":
            return {"noul_logit": self.noul_head(pair).squeeze(-1)}
        if question_type == "score":
            return {"score_logits": self.score_head(pair)}
        if question_type == "action":
            return {"action_logits": self.action_head(pair)}
        if question_type == "choice":
            if option_embeddings is None:
                raise ValueError("choice questions require option embeddings")
            option = torch.tanh(self.choice_scorer(option_embeddings.float()))
            pair_expanded = pair.unsqueeze(1).expand(-1, option.shape[1], -1)
            option_state = torch.tanh(pair_expanded + option)
            return {"choice_logits": self.choice_value(option_state).squeeze(-1)}
        raise ValueError(f"unsupported frontier decision type: {question_type}")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        question_embeddings: Tensor,
        question_type: str,
        option_embeddings: Tensor | None = None,
        audio_waveform: Tensor | None = None,
        accel: Tensor | None = None,
    ) -> dict[str, Tensor]:
        state = self.encode_batch(input_ids, attention_mask, audio_waveform, accel)
        question = torch.tanh(self.question_proj(question_embeddings.float()))
        pair = self._pair(state, question)
        if question_type == "noul":
            return {"noul_logit": self.noul_head(pair).squeeze(-1), "state": state}
        if question_type == "score":
            return {"score_logits": self.score_head(pair), "state": state}
        if question_type == "choice":
            if option_embeddings is None:
                raise ValueError("choice questions require option embeddings")
            option = torch.tanh(self.choice_scorer(option_embeddings.float()))
            pair_expanded = pair.unsqueeze(1).expand(-1, option.shape[1], -1)
            option_state = torch.tanh(pair_expanded + option)
            logits = self.choice_value(option_state).squeeze(-1)
            return {"choice_logits": logits, "state": state}
        if question_type == "action":
            return {"action_logits": self.action_head(pair), "state": state}
        raise ValueError(f"unsupported frontier decision type: {question_type}")


def _tokenizer(model_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


class FrontierModelProvider:
    name = "frontier"

    def __init__(self):
        self._model: ClinevoOneFrontier | None = None
        self._tokenizer: Any | None = None
        self._version = "unloaded"

    def _checkpoint(self) -> Path:
        return Path(os.getenv("FRONTIER_MODEL_CHECKPOINT", "/cache/clinevo-owned/frontier.pt"))

    def _load(self) -> ClinevoOneFrontier:
        if self._model is not None:
            return self._model
        model_id = os.getenv("FRONTIER_MODEL_ID", FRONTIER_DEFAULT_MODEL)
        model = ClinevoOneFrontier(model_id)
        checkpoint = self._checkpoint()
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
            self._version = str(payload.get("model_version", checkpoint.stem))
        else:
            self._version = f"{model_id}:base"
        model.eval()
        self._tokenizer = _tokenizer(model_id)
        self._model = model
        return model

    def _encode_texts(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        tokenizer = self._tokenizer or _tokenizer(os.getenv("FRONTIER_MODEL_ID", FRONTIER_DEFAULT_MODEL))
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=int(os.getenv("FRONTIER_MAX_TOKENS", "1024")))
        return batch["input_ids"], batch["attention_mask"]

    def predict(self, state: Any, questions: dict[str, Any]):
        model = self._load()
        state_text = _state_text(state)
        input_ids, attention_mask = self._encode_texts([state_text])
        audio = _audio_tensor(state)
        accel = _accel_tensor(state)

        question_names = list(questions)
        question_texts = []
        option_groups: list[list[str]] = []
        for name in question_names:
            question = questions[name]
            criteria = question.get("criteria")
            options = list(criteria.keys()) if isinstance(criteria, dict) else []
            if not options and question.get("type") == "action":
                options = list(ACTION_CHOICES)
            option_groups.append(options)
            rendered = f"{name}. {question.get('instructions', '')}. " + " ".join(f"{key}: {value}" for key, value in (criteria.items() if isinstance(criteria, dict) else enumerate(criteria or [])))
            question_texts.append(rendered)

        q_ids, q_mask = self._encode_texts(question_texts)
        with torch.inference_mode():
            q_emb = model.text_backbone(q_ids, q_mask)
            state = model.encode_batch(input_ids, attention_mask, audio, accel)
            answers: dict[str, Any] = {}
            for index, name in enumerate(question_names):
                question = questions[name]
                qtype = str(question.get("type", "noul"))
                options = option_groups[index]
                option_embeddings = None
                if qtype == "choice":
                    option_ids, option_mask = self._encode_texts(options)
                    option_embeddings = model.text_backbone(option_ids, option_mask).unsqueeze(0)
                output = model.decide_from_state(
                    state,
                    q_emb[index:index + 1],
                    qtype,
                    option_embeddings=option_embeddings,
                )
                temperature = 1.0
                if qtype == "noul":
                    p = float(torch.sigmoid(output["noul_logit"] / temperature).item())
                    answers[name] = {"type": "noul", "noul": p, "confidence": abs(p - 0.5) * 2}
                elif qtype == "score":
                    probs = torch.softmax(output["score_logits"] / temperature, dim=-1)[0]
                    score = float((probs * torch.arange(4)).sum().item())
                    answers[name] = {"type": "score", "score": score, "confidence": float(probs.max().item()), "probabilities": {str(i): float(v) for i, v in enumerate(probs.tolist())}}
                else:
                    probs = torch.softmax(output["action_logits" if qtype == "action" else "choice_logits"] / temperature, dim=-1)[0]
                    idx = int(probs.argmax().item())
                    answers[name] = {"type": "action" if qtype == "action" else "choice", "choice": options[idx], "confidence": float(probs[idx].item()), "probabilities": {options[i]: float(probs[i].item()) for i in range(len(options))}}
                    if qtype == "action":
                        answers[name]["action_allowed"] = options[idx] in {"enrich", "review", "escalate", "hold"}
        return self._version, answers, {"architecture": "ClinevoOne-Frontier", "backbone": self._load().model_id, "multimodal": {"text": True, "audio": audio is not None, "accelerometer": accel is not None}, "typed_decisions": True, "action_head": True}


def frontier_health() -> dict[str, Any]:
    checkpoint = Path(os.getenv("FRONTIER_MODEL_CHECKPOINT", "/cache/clinevo-owned/frontier.pt"))
    return {
        "backend": "frontier",
        "available": checkpoint.exists(),
        "checkpoint": str(checkpoint),
        "model_id": os.getenv("FRONTIER_MODEL_ID", FRONTIER_DEFAULT_MODEL),
        "architecture": "ClinevoOne-Frontier",
        "action_space": list(ACTION_CHOICES),
        "base_backbone_parameters": 395_000_000,
    }


__all__ = ["ClinevoOneFrontier", "FrontierModelProvider", "frontier_health", "ACTION_CHOICES"]
