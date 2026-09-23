from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from app.decision_engine import INBOX_DECISION_QUESTIONS
from app.frontier_model import ClinevoOneFrontier
from ml.synthetic_dataset import build_dataset


def _set_backbone_trainable(model, last_n: int) -> None:
    for parameter in model.text_backbone.encoder.parameters():
        parameter.requires_grad = False
    if last_n <= 0:
        return
    encoder = model.text_backbone.encoder
    layers = getattr(getattr(encoder, "encoder", None), "layers", None)
    if layers is None:
        layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    if layers is None:
        raise RuntimeError("could not locate transformer layers for selective unfreezing")
    for block in list(layers)[-last_n:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def _adapter_state(model):
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("text_backbone.encoder.")
    }


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = build_dataset(args.size, args.seed)
    model = ClinevoOneFrontier(args.model_id)

    if args.init:
        payload = torch.load(args.init, map_location="cpu", weights_only=False)
        weights = payload.get("model", payload.get("adapter"))
        if weights:
            model.load_state_dict(weights, strict=False)

    _set_backbone_trainable(model, args.unfreeze_last_n)

    if args.freeze_signal:
        for module in (model.audio, model.accel):
            for parameter in module.parameters():
                parameter.requires_grad = False

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).train()

    tokenizer = None
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    question = INBOX_DECISION_QUESTIONS[args.task]
    options = list(question.get("criteria", {}).keys())
    if not options:
        raise ValueError("frontier supervised trainer currently requires a choice task")
    option_batch = tokenizer(options, return_tensors="pt", padding=True, truncation=True, max_length=args.max_tokens)
    option_ids = option_batch["input_ids"].to(device)
    option_mask = option_batch["attention_mask"].to(device)

    for step in range(args.steps):
        item = data[step % len(data)]
        state_batch = tokenizer(
            [_state_text_for_training(item.text)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_tokens,
        )
        q_text = f"{args.task}. {question['instructions']}. " + " ".join(f"{k}: {v}" for k, v in question["criteria"].items())
        q_batch = tokenizer([q_text], return_tensors="pt", padding=True, truncation=True, max_length=args.max_tokens)

        optimizer.zero_grad(set_to_none=True)
        state = model.encode_batch(
            state_batch["input_ids"].to(device),
            state_batch["attention_mask"].to(device),
            torch.tensor(item.audio, device=device),
            torch.tensor(item.accel, device=device),
        )
        q_emb = model.text_backbone(q_batch["input_ids"].to(device), q_batch["attention_mask"].to(device))
        with torch.no_grad():
            option_emb = model.text_backbone(option_ids, option_mask).unsqueeze(0)
        output = model.decide_from_state(state, q_emb, "choice", option_emb)
        target = torch.tensor([options.index(item.labels[args.task])], device=device)
        loss = torch.nn.functional.cross_entropy(output["choice_logits"], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if (step + 1) % max(args.log_every, 1) == 0:
            print(f"step={step + 1} loss={float(loss.detach()):.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": f"clinevo-frontier-sft-{args.seed}-{args.steps}",
        "architecture": "ClinevoOne-Frontier",
        "base_model": args.model_id,
        "training_method": "supervised_finetuning",
        "task": args.task,
        "model": _adapter_state(model) if args.unfreeze_last_n == 0 else {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "adapter_only": args.unfreeze_last_n == 0,
        "training": vars(args),
    }
    torch.save(payload, output)
    print(json.dumps({"checkpoint": str(output), "model_version": payload["model_version"], "adapter_only": payload["adapter_only"]}, indent=2))


def _state_text_for_training(text: str) -> str:
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ClinevoOne-Frontier on a labeled decision dataset")
    parser.add_argument("--model-id", default="answerdotai/ModernBERT-large")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--task", choices=("route",), default="route")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--freeze-signal", action="store_true")
    parser.add_argument("--init", default="")
    parser.add_argument("--output", default="/cache/clinevo-owned/frontier.pt")
    parser.add_argument("--log-every", type=int, default=25)
    train(parser.parse_args())
