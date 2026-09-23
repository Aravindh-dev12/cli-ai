# CLI v2 live benchmark

Run the GitHub Actions workflow **CLI v2 live benchmark** to compare the in-house ClinevoOne-v1 route model with live Laya and, when configured, live Jev. Pushes to `main` and manual dispatches now use the same benchmark protocol.

## Protocol

The CLI checkpoint is trained on synthetic seed 7. The benchmark is generated from a different seed and then evaluated only on the subject-held-out test partition.

Metrics: route accuracy, macro F1, confidence ECE against correctness, request error rate, mean confidence, and p50/p95 steady-state latency. Each provider receives one warmup request; warmup time is reported separately and excluded from p50/p95.

Jev is invoked only when the repository has a valid `TYPESAFE_API_KEY` secret. Jev outputs are comparison-only and are not used for training or distillation.

## Current v1 live baseline

GitHub Actions run 35822076537, 8 synthetic test examples:

| System | Accuracy | p50 latency | p95 latency |
|---|---:|---:|---:|
| CLI / ClinevoOne-v1 | 0.250 | 2.65 ms | 4.20 ms |
| Laya live | 0.875 | 428.82 ms | 8,535.59 ms |
| Jev | not run | — | — |

These figures are an integration smoke result, not production evidence. The model was trained for one epoch on a tiny synthetic corpus.

## Current Laya reference

The current Laya model card describes the English checkpoint as a 421M-parameter ModernBERT-large decision model with request-time option scoring, single-forward-pass question batching, and RLCD training. Its published speed table reports 39.5 ms for one question and 158.6 ms for ten questions on a Tesla T4. Those are Laya's published measurements, not this repository's hardware benchmark.

The TypeSafe API documents `POST /v1/systemone` and bearer-token authentication for Jev.

## v2 CI fix

The first v2 push run on 2026-09-23 reached model training successfully but failed before comparison because the workflow expanded the optional manual-dispatch input to an empty `--size` argument. The workflow now resolves the benchmark size to 64 on both push and manual runs, so the benchmark executes from the same configuration.

## Hugging Face

The canonical target is `Aravindhan11/cli-ai`.

The repository already contains a one-click publication workflow and a correct ClinevoOne exporter. The connected Hugging Face session currently has read-only repository scopes, so the model repository has not been created or uploaded by this session.

Once a Hugging Face token with repository write access is supplied as the GitHub `HF_TOKEN` secret, the publication workflow can upload the packaged CLI checkpoint and model card.
