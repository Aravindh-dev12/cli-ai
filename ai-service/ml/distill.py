from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.decision_engine import INBOX_DECISION_QUESTIONS
from app.owned_model import ClinevoOne, text_to_ids, _question_text


def _route_item(model: ClinevoOne, text: str):
    question = INBOX_DECISION_QUESTIONS["route"]
    output = model(
        text_to_ids(text),
        text_to_ids(_question_text("route", question)),
        "choice",
        choice_count=len(question["criteria"]),
    )
    return output["choice_logits"], list(question["criteria"])


def distill(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = [json.loads(line) for line in Path(args.teacher_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("teacher JSONL is empty")

    model = ClinevoOne()
    if args.init:
        payload = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)

    if args.lora:
        for module in model.modules():
            if hasattr(module, "freeze_base") and module.__class__.__name__ == "LoRALinear":
                module.freeze_base()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    temperature = max(args.temperature, 0.1)

    for epoch in range(args.epochs):
        random.shuffle(rows)
        losses = []
        for row in rows:
            teacher = row.get("route_probabilities") or row.get("teacher_probabilities")
            if not isinstance(teacher, dict):
                continue
            logits, choices = _route_item(model, str(row["text"]))
            target = torch.tensor([float(teacher.get(choice, 0.0)) for choice in choices], dtype=torch.float32)
            total = float(target.sum())
            if total <= 0:
                continue
            target = target / total
            log_probs = torch.log_softmax(logits / temperature, dim=-1)
            loss = nn.functional.kl_div(log_probs, target, reduction="batchmean") * (temperature ** 2)

            hard_label = row.get("label")
            if hard_label in choices:
                hard = nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([choices.index(hard_label)]))
                loss = (1.0 - args.hard_weight) * loss + args.hard_weight * hard

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        mean_loss = sum(losses) / max(len(losses), 1)
        print(f"epoch={epoch + 1} distill_loss={mean_loss:.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    version = f"clinevoone-distilled-{args.seed}-{len(rows)}"
    torch.save({
        "model_version": version,
        "architecture": "ClinevoOne-v1",
        "teacher_distillation": True,
        "teacher_dataset": str(args.teacher_jsonl),
        "distillation_temperature": temperature,
        "model": model.state_dict(),
        "training": vars(args),
    }, output)
    print(json.dumps({"checkpoint": str(output), "model_version": version, "teacher_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill Laya/Jev teacher decisions into ClinevoOne")
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--output", default="/cache/clinevo-owned/distilled.pt")
    parser.add_argument("--init", default="")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hard-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lora", action="store_true")
    distill(parser.parse_args())
