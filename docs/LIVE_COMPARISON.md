# CLI live comparison record

## Run

GitHub Actions run 35822076537 on commit `5290c1949000d18407c8b7e2e7b057b1371da071`.

The live smoke harness used the same eight synthetic route examples for CLI and Laya. CLI used the locally trained ClinevoOne-v1 route head; Laya used its live `Router` runtime. Jev was not called because `JEV_API_KEY` was not configured in GitHub Actions.

| System | Accuracy | Error rate | Confidence ECE* | Mean confidence | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| CLI / ClinevoOne-v1 | 0.250 | 0.000 | 0.0548 | 0.3048 | 2.65 ms | 4.20 ms |
| Laya live | 0.875 | 0.000 | 0.5466 | 0.3284 | 428.82 ms | 8,535.59 ms |
| Jev live | not run | — | — | — | — | — |

* This ECE is calculated against per-example correctness using the model-reported confidence. It is a small smoke-test diagnostic, not a calibration benchmark.

## Interpretation

This run validates integration and gives a reproducible baseline. It does **not** show production superiority: the CLI checkpoint was trained for one epoch on eight synthetic training examples, and the comparison set also contains only eight synthetic examples.

The latency numbers are also not apples-to-apples hardware benchmarks. CLI ran as a small local CPU neural model; Laya loaded and executed its production checkpoint locally through its Router.

## Published external context

The current Laya model card describes Laya as a non-autoregressive typed-decision model based on ModernBERT-large, with RLCD-style proper-scoring training and request-time typed choices. Its published typed-decisions evaluation reports 0.766 accuracy and Jev 1.13.0 at 0.727 on that separate 400-case/2,000-decision test. Those figures are published external results, not measurements from this repository.

The Laya card also reports GPU latency of 39.5 ms for one question and 158.6 ms for ten questions on a Tesla T4, while explicitly noting that its Jev numbers are third-party published rather than measured in that run.

## Jev live requirement

The TypeSafe API documentation requires an API key in the Authorization Bearer header for `POST /v1/systemone`. This repository therefore keeps the Jev live lane opt-in through the `TYPESAFE_API_KEY` GitHub secret.

## Reproduction

Run the workflow from GitHub Actions or locally:

```bash
cd ai-service
python -m ml.live_compare --checkpoint /path/to/cli-checkpoint.pt --size 8
```

The workflow artifact is named `cli-laya-jev-live-comparison`.

Jev results, when a valid key is supplied, are comparison-only. The repository does not distill Jev outputs or use them as training labels.
