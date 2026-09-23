# ClinevoOne research track

This document records the research direction for building an in-house decision model rather than treating Laya or Jev as the production model.

## What the external System One pattern teaches us

TypeSafe describes Jev as a machine-facing decision primitive: application state goes in, typed `choice`, `score`, or yes/no decisions come out, accompanied by probabilities/confidence. Its launch material also describes a dedicated training objective, Reinforcement Learning for Calibrated Decisions (RLCD), and parallel output generation rather than sequential text generation.

Laya is a public open-weight implementation in the same broad design space. The Hugging Face model is approximately 421M parameters and is distributed under Apache-2.0. Its model metadata is tagged for calibrated decisions, routing, scoring, guardrails, and reinforcement learning.

ClinevoOne keeps the interface contract but does not clone the external model:

- one shared state representation per request
- many typed questions evaluated against the same state
- explicit output domains rather than free-form text
- confidence and probability distributions as first-class outputs
- post-hoc calibration on a held-out validation set
- software-owned thresholds and human-review gates
- optional teacher supervision from Laya/Jev, followed by local distillation

## Why the multimodal architecture is different

The target application also needs signal intelligence. ClinevoOne has three independent encoders:

1. Text: hashed lexical representation, multi-scale 1-D convolutions, bidirectional GRU, and LoRA projection.
2. Audio: log-magnitude STFT followed by a compact 2-D CNN and LoRA projection.
3. Accelerometer: raw XYZ plus magnitude, 1-D CNN, bidirectional GRU, and LoRA projection.

The encoders are fused with explicit modality-presence masks. Missing sensors are therefore an explicit input condition rather than an invisible zero-vector substitution.

## Calibration

A model can be accurate and still be poorly calibrated. ClinevoOne now fits global temperature scaling on a held-out calibration set and loads the resulting temperatures from a sidecar artifact.

Run:

```bash
make owned-calibrate
```

Do not fit calibration temperatures on the final test set. The calibration artifact records the model version and dataset identity.

## Teacher-student path

The local workflow supports teacher labeling and distillation:

```bash
make owned-teacher-label
make owned-distill
```

`owned-teacher-label` records teacher probability distributions, and multiple teachers can be averaged:

```bash
OWNED_TEACHERS=laya,jev make owned-teacher-label
```

The distillation stage minimizes temperature-scaled KL divergence and can blend hard labels. Teacher probabilities are supervision signals, not ground truth.

## Pet-wellness data strategy

Public datasets are useful for engineering validation but should not be confused with veterinary outcome data.

- DogSpeak provides 77,202 bark sequences from 156 dogs across five breeds and is intended for in-the-wild canine vocalization research. Its Hugging Face card lists CC BY-NC-SA 4.0.
- Barkopedia contains 8,924 labeled bark clips from 60 individual dogs and supports subject-disjoint audio experiments. Its Hugging Face card lists an MIT license.
- A Royal Veterinary College / UC Irvine Dryad release provides activity-analysis data and code for domestic dogs collected with accelerometry.
- A recent dog-motion study reports a 45-dog wearable accelerometer/gyroscope dataset covering controlled activity tasks.

The repository must not silently download or package these sources. Each adapter should record license, provenance, subject identifiers, sampling details, preprocessing version, label semantics, and checksums before training.

## Experimental protocol

The owned model should be promoted only after measurement on a representative, subject-held-out evaluation set:

- macro F1 and per-class recall for discrete routes
- score MAE for ordinal urgency outputs
- Brier score or log loss for probabilistic yes/no outputs
- ECE and reliability diagrams for confidence
- p50/p95 and cold-start latency
- peak memory and artifact size
- missing-modality robustness
- sensor-noise robustness
- cross-device and cross-environment generalization

External models and the owned model must receive the same state, question definitions, and evaluation labels. The synthetic corpus remains a plumbing/regression fixture and cannot establish veterinary efficacy or superiority over Laya or Jev.

## Next research milestone

The highest-value next experiment is a subject-held-out multimodal dataset with aligned audio and IMU windows, explicit outcome labels, and a fixed evaluation protocol. Compare the deterministic baseline, Laya, Jev, supervised ClinevoOne, distilled ClinevoOne, and calibrated ClinevoOne on that exact test set.
