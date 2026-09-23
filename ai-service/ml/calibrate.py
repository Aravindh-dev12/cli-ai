from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from app.decision_engine import INBOX_DECISION_QUESTIONS
from app.owned_model import ClinevoOne, _question_text, text_to_ids
from ml.synthetic_dataset import build_dataset, split_dataset


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, binary: bool) -> float:
    log_temp = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=60, line_search_fn="strong")

    def closure():
        optimizer.zero_grad()
        temperature = log_temp.exp().clamp(0.05, 20.0)
        if binary:
            loss = nn.functional.binary_cross_entropy_with_logits(logits / temperature, targets.float())
        else:
            loss = nn.functional.cross_entropy(logits / temperature, targets.long())
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temp.exp().clamp(0.05, 20.0).detach())


def ece_multiclass(logits: torch.Tensor, targets: torch.Tensor, temperature: float, bins: int = 10) -> float:
    probs = torch.softmax(logits / temperature, dim=-1)
    confidence, prediction = probs.max(dim=-1)
    correct = (prediction == targets).float()
    edges = torch.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        if mask.any():
            value += float(mask.float().mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return value


def ece_binary(logits: torch.Tensor, targets: torch.Tensor, temperature: float, bins: int = 10) -> float:
    probs = torch.sigmoid(logits / temperature)
    confidence = torch.maximum(probs, 1.0 - probs)
    prediction = (probs >= 0.5).long()
    correct = (prediction == targets.long()).float()
    edges = torch.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        if mask.any():
            value += float(mask.float().mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return value


def collect(model: ClinevoOne, data):
    logits: dict[str, list[torch.Tensor]] = {name: [] for name in INBOX_DECISION_QUESTIONS}
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in INBOX_DECISION_QUESTIONS}
    binary: dict[str, bool] = {}
    for item in data:
        for name, question in INBOX_DECISION_QUESTIONS.items():
            qtype = str(question["type"])
            binary[name] = qtype == "noul"
            output = model(
                text_to_ids(item.text),
                text_to_ids(_question_text(name, question)),
                qtype,
                choice_count=len(question.get("criteria", {})) if isinstance(question.get("criteria"), dict) else 0,
                audio_waveform=torch.tensor(item.audio),
                accel=torch.tensor(item.accel),
            )
            if qtype == "noul":
                logits[name].append(output["noul_logit"].detach().reshape(()))
                targets[name].append(torch.tensor(float(item.labels[name])))
            else:
                values = output["score_logits"] if qtype == "score" else output["choice_logits"]
                logits[name].append(values.detach())
                if qtype == "score":
                    targets[name].append(torch.tensor(int(item.labels[name])))
                else:
                    choices = list(question["criteria"])
                    targets[name].append(torch.tensor(choices.index(item.labels[name])))
    return (
        {name: torch.stack(values) for name, values in logits.items()},
        {name: torch.stack(values) for name, values in targets.items()},
        binary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit temperature scaling on an independent subject-held-out calibration partition")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--out", default="/cache/clinevo-owned/calibration.json")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ClinevoOne()
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    records = build_dataset(args.size * 2, seed=2026)
    _, calibration_set, _ = split_dataset(records, seed=2026)
    if not calibration_set:
        raise ValueError("subject-held-out calibration split is empty; increase --size")

    logits, targets, binary = collect(model, calibration_set)
    temperatures: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}

    for name in logits:
        temperature = fit_temperature(logits[name], targets[name], binary[name])
        temperatures[name] = temperature
        if binary[name]:
            pre = ece_binary(logits[name], targets[name], 1.0)
            post = ece_binary(logits[name], targets[name], temperature)
            accuracy = float(((torch.sigmoid(logits[name]) >= 0.5).long() == targets[name].long()).float().mean())
        else:
            pre = ece_multiclass(logits[name], targets[name], 1.0)
            post = ece_multiclass(logits[name], targets[name], temperature)
            accuracy = float((logits[name].argmax(dim=-1) == targets[name].long()).float().mean())
        metrics[name] = {"accuracy": accuracy, "ece_before": pre, "ece_after": post}

    result = {
        "model_version": payload.get("model_version"),
        "temperatures": temperatures,
        "metrics": metrics,
        "calibration_method": "global_temperature_scaling",
        "dataset": {
            "version": "synthetic-v1",
            "seed": 2026,
            "records_generated": len(records),
            "calibration_example_count": len(calibration_set),
            "calibration_subject_count": len({item.subject_id for item in calibration_set}),
            "split": "subject-held-out",
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
