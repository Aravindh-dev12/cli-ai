from __future__ import annotations
import argparse, json, os, time
from collections import Counter
from pathlib import Path
import numpy as np, torch
from app.decision_engine import INBOX_DECISION_QUESTIONS, LayaProvider, JevProvider
from app.owned_model import ClinevoOne, _question_text, text_to_ids
from ml.synthetic_dataset import build_dataset, split_dataset

def f1_macro(y, p, labels):
    vals = []
    for c in labels:
        tp = sum(t == c and q == c for t, q in zip(y, p))
        fp = sum(t != c and q == c for t, q in zip(y, p))
        fn = sum(t == c and q != c for t, q in zip(y, p))
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        vals.append(2 * pr * rc / max(pr + rc, 1e-12))
    return float(np.mean(vals))

def ece(conf, correct, bins=10):
    if not conf:
        return 0.0
    p = np.asarray(conf)
    y = np.asarray(correct)
    total = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            total += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return total

def main():
    ap = argparse.ArgumentParser(description="Live subject-held-out benchmark for CLI, Laya and Jev")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=91023)
    ap.add_argument("--out", default="/tmp/cli-laya-jev-v2.json")
    a = ap.parse_args()
    if a.size <= 0:
        raise ValueError("--size must be positive")

    records = build_dataset(a.size * 2, seed=a.seed)
    _, _, data = split_dataset(records, seed=a.seed)
    if not data:
        raise ValueError("empty subject-held-out test set")

    q = INBOX_DECISION_QUESTIONS["route"]
    labels = list(q["criteria"])
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = ClinevoOne().eval()
    model.load_state_dict(payload["model"], strict=True)
    laya = LayaProvider()
    jev = JevProvider() if os.getenv("JEV_API_KEY", "").strip() else None

    def cli(item):
        with torch.inference_mode():
            o = model(
                text_to_ids(item.text),
                text_to_ids(_question_text("route", q)),
                "choice",
                choice_count=len(labels),
                audio_waveform=torch.tensor(item.audio),
                accel=torch.tensor(item.accel),
            )
            prob = torch.softmax(o["choice_logits"], dim=-1)
            i = int(prob.argmax())
            return labels[i], float(prob[i])

    def remote(provider, item):
        _, ans, _ = provider.predict(item.text, {"route": q})
        answer = ans["route"]
        return answer.get("choice"), float(answer.get("confidence", 0.0))

    def run(name, fn):
        warmup_ms = None
        try:
            warmup_started = time.perf_counter()
            fn(data[0])
            warmup_ms = (time.perf_counter() - warmup_started) * 1000
        except Exception as exc:
            print(f"{name} warmup error: {type(exc).__name__}: {exc}")

        y, p, c, ok, lat = [], [], [], [], []
        err = 0
        for item in data:
            st = time.perf_counter()
            try:
                pred, conf = fn(item)
                y.append(item.labels["route"])
                p.append(pred)
                c.append(conf)
                ok.append(int(pred == item.labels["route"]))
            except Exception as exc:
                err += 1
                print(f"{name} error: {type(exc).__name__}: {exc}")
            lat.append((time.perf_counter() - st) * 1000)
        return {
            "accuracy": float(np.mean(ok)) if ok else 0.0,
            "macro_f1": f1_macro(y, p, labels) if y else 0.0,
            "error_rate": err / max(len(data), 1),
            "ece_correctness": ece(c, ok),
            "mean_confidence": float(np.mean(c)) if c else 0.0,
            "warmup_ms": warmup_ms,
            "p50_latency_ms": float(np.percentile(lat, 50)) if lat else None,
            "p95_latency_ms": float(np.percentile(lat, 95)) if lat else None,
            "sample_size": len(data),
            "evaluated": len(y),
            "truth_counts": dict(Counter(y)),
            "prediction_counts": dict(Counter(p)),
        }

    models = {"cli": run("cli", cli), "laya": run("laya", lambda x: remote(laya, x))}
    if jev:
        models["jev"] = run("jev", lambda x: remote(jev, x))

    out = {
        "protocol": "cli-vs-laya-vs-jev-live-v2",
        "dataset": {
            "version": "synthetic-v1",
            "seed": a.seed,
            "records_generated": len(records),
            "subject_held_out_test": True,
            "test_examples": len(data),
            "test_subject_count": len({item.subject_id for item in data}),
        },
        "models": models,
        "jev_live": bool(jev),
        "latency_protocol": "one warmup request per provider; reported p50/p95 exclude warmup",

        "limitations": [
            "Synthetic data only",
            "CLI and benchmark distributions are synthetic",
            "Latency is hardware/runtime dependent",
            "Remote provider latency includes network path but excludes the initial warmup request from p50/p95",
        ],
    }
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
