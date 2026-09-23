from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.owned_model import ClinevoOne, _question_text, text_to_ids
from app.decision_engine import INBOX_DECISION_QUESTIONS
from ml.synthetic_dataset import build_dataset


def log_score(probabilities: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.log((probabilities * targets).sum().clamp_min(eps))


def spherical_score(probabilities: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    norm = probabilities.square().sum().sqrt().clamp_min(eps)
    return (probabilities * targets).sum() / norm


def ranked_probability_score(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    cdf_p = probabilities.cumsum(dim=-1)
    cdf_t = targets.cumsum(dim=-1)
    return 1.0 - (cdf_p[:-1] - cdf_t[:-1]).square().mean()


def proper_reward(logits: torch.Tensor, target: torch.Tensor, ordinal: bool = False) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    if ordinal:
        return log_score(probs, target) + spherical_score(probs, target) + ranked_probability_score(probs, target)
    return log_score(probs, target) + spherical_score(probs, target)


def run_epoch(model: ClinevoOne, records, optimizer, temperature: float, noise_std: float):
    rewards = []
    losses = []
    question = INBOX_DECISION_QUESTIONS["route"]
    choices = list(question["criteria"])
    baseline = 0.0
    for item in records:
        with torch.no_grad():
            state = model.encode_state(text_to_ids(item.text), torch.tensor(item.audio), torch.tensor(item.accel))
        question_ids = text_to_ids(_question_text("route", question))
        logits = model.choice_head(model._pair(state, model.question(question_ids)))
        noise = torch.randn_like(logits) * noise_std
        explored = logits / max(temperature, 0.1) + noise
        target = torch.zeros_like(logits)
        target[choices.index(item.labels["route"])] = 1.0
        reward = proper_reward(explored.unsqueeze(0).squeeze(0), target)
        advantage = reward.detach() - baseline
        loss = -(advantage * torch.log_softmax(explored / max(temperature, 0.1), dim=-1)[target.bool()].sum())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        reward_value = float(reward.detach())
        baseline = 0.95 * baseline + 0.05 * reward_value
        rewards.append(reward_value)
        losses.append(float(loss.detach()))
    return float(np.mean(rewards)), float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(description="RLCD-style fine-tuning of ClinevoOne with proper scoring rules")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--init", default="")
    parser.add_argument("--output", default="/cache/clinevo-owned/rlcd.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = build_dataset(args.size, args.seed)
    model = ClinevoOne()
    if args.init:
        payload = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        reward, loss = run_epoch(model, data, optimizer, args.temperature, args.noise_std)
        print(f"epoch={epoch + 1} reward={reward:.5f} loss={loss:.5f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    version = f"clinevoone-rlcd-{args.seed}-{args.size}"
    torch.save({
        "model_version": version,
        "architecture": "ClinevoOne-v1",
        "training_method": "RLCD-style",
        "reward": "log+spherical; ordinal adds RPS",
        "model": model.state_dict(),
        "training": vars(args),
    }, output)
    print(json.dumps({"checkpoint": str(output), "model_version": version}, indent=2))


if __name__ == "__main__":
    main()
