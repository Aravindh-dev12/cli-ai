from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from app.decision_engine import INBOX_DECISION_QUESTIONS
from app.frontier_model import ClinevoOneFrontier
from ml.train_frontier import _set_backbone_trainable
from ml.synthetic_dataset import build_dataset


def proper_reward(logits: torch.Tensor, target: torch.Tensor, ordinal: bool = False, eps: float = 1e-7) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    log_score = torch.log((probabilities * target).sum().clamp_min(eps))
    spherical = (probabilities * target).sum() / probabilities.square().sum().sqrt().clamp_min(eps)
    if not ordinal:
        return log_score + spherical
    cdf_p = probabilities.cumsum(dim=-1)
    cdf_t = target.cumsum(dim=-1)
    ranked = 1.0 - (cdf_p[:-1] - cdf_t[:-1]).square().mean()
    return log_score + spherical + ranked


def adapter_state(model: ClinevoOneFrontier):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("text_backbone.encoder.")}


def main() -> None:
    parser = argparse.ArgumentParser(description="RLCD-style reinforcement learning for calibrated decisions")
    parser.add_argument("--model-id", default="answerdotai/ModernBERT-large")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--init", default="")
    parser.add_argument("--output", default="/cache/clinevo-owned/frontier-rlcd.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data = build_dataset(args.size, args.seed)
    model = ClinevoOneFrontier(args.model_id)

    if args.init:
        payload = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model", payload.get("adapter")), strict=False)

    _set_backbone_trainable(model, args.unfreeze_last_n)
    for parameter in model.audio.parameters():
        parameter.requires_grad = False
    for parameter in model.accel.parameters():
        parameter.requires_grad = False

    model.to(device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    question = INBOX_DECISION_QUESTIONS["route"]
    options = list(question["criteria"])
    option_tokens = tokenizer(options, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    option_ids = option_tokens["input_ids"].to(device)
    option_mask = option_tokens["attention_mask"].to(device)
    question_text = f"route. {question['instructions']}. " + " ".join(f"{k}: {v}" for k, v in question["criteria"].items())
    question_tokens = tokenizer([question_text], return_tensors="pt", padding=True, truncation=True, max_length=1024)
    question_ids = question_tokens["input_ids"].to(device)
    question_mask = question_tokens["attention_mask"].to(device)

    baseline = 0.0
    for epoch in range(args.epochs):
        random.shuffle(data)
        rewards = []
        for item in data:
            state_tokens = tokenizer([item.text], return_tensors="pt", padding=True, truncation=True, max_length=1024)
            state_ids = state_tokens["input_ids"].to(device)
            state_mask = state_tokens["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            state = model.encode_batch(
                state_ids,
                state_mask,
                torch.tensor(item.audio, device=device),
                torch.tensor(item.accel, device=device),
            )
            question_embedding = model.text_backbone(question_ids, question_mask)
            with torch.no_grad():
                option_embedding = model.text_backbone(option_ids, option_mask).unsqueeze(0)
            output = model.decide_from_state(state, question_embedding, "choice", option_embedding)
            logits = output["choice_logits"][0]
            target = torch.zeros_like(logits)
            target[options.index(item.labels["route"])] = 1.0

            explored = logits / max(args.temperature, 0.1) + torch.randn_like(logits) * args.noise_std
            probabilities = torch.softmax(explored, dim=-1)
            action = torch.multinomial(probabilities, 1).item()
            reward = proper_reward(explored, target)
            advantage = reward.detach() - baseline
            loss = -advantage * torch.log_softmax(explored, dim=-1)[action]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            reward_value = float(reward.detach())
            baseline = 0.95 * baseline + 0.05 * reward_value
            rewards.append(reward_value)

        print(f"epoch={epoch + 1} reward={float(np.mean(rewards)):.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": f"clinevo-frontier-rlcd-{args.seed}-{args.size}",
        "architecture": "ClinevoOne-Frontier",
        "base_model": args.model_id,
        "training_method": "RLCD-style",
        "reward": "log+spherical; ordinal adds RPS",
        "exploration": {"distribution": "gaussian-logit-noise", "std": args.noise_std},
        "model": adapter_state(model) if args.unfreeze_last_n == 0 else {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "adapter_only": args.unfreeze_last_n == 0,
        "training": vars(args),
    }
    torch.save(payload, output)
    print(json.dumps({"checkpoint": str(output), "model_version": payload["model_version"], "adapter_only": payload["adapter_only"]}, indent=2))


if __name__ == "__main__":
    main()
