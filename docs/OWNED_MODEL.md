# ClinevoOne in-house model

ClinevoOne is the owned-model track for the Clinevo platform. It is designed as a domain-specialized, multimodal typed-decision network rather than a generic chat model.

Architecture:
Text -> hashed token embeddings -> multi-scale 1D CNN -> BiGRU -> text embedding
Audio -> STFT/log spectrum -> 2D CNN -> audio embedding
Accelerometer -> 3-axis + magnitude -> 1D CNN -> BiGRU -> IMU embedding
All present modalities -> gated fusion -> shared state embedding
Question text -> shared question encoder -> typed decision head

Typed heads:
- NOUL binary probability
- SCORE four-level ordinal decision
- CHOICE categorical decision

LoRA support is implemented directly as low-rank adapters on the custom projection layers. A full checkpoint can therefore be adapted to a new dataset while freezing its base weights.

Training lifecycle:
raw data -> schema checks -> subject-aware split -> train -> validation -> calibration -> frozen benchmark -> registry -> promotion gate -> inference

Local commands:
make owned-train
make owned-evaluate
make owned-benchmark
make owned-calibrate
make owned-teacher-label
make owned-distill

Model artifacts are kept under /cache/clinevo-owned inside the persistent local model volume.

Benchmark policy:
1. Use one frozen held-out dataset.
2. Measure route accuracy/macro-F1, NOUL Brier/ECE, SCORE error, p50/p95 latency, cold start, memory and fallback/error rate.
3. Compare ClinevoOne against the deterministic baseline, Laya, and Jev when available.
4. Do not declare an overall winner. Promotion uses pre-declared workload thresholds.

The initial synthetic dataset exists only to prove the model and MLOps plumbing. It is not evidence of real-world pet-wellness, audio, accelerometer, or pharmacovigilance performance.

For a real pet-wellness product, add representative audio/IMU recordings, subject-level splits, sensor-quality checks, augmentation, class-imbalance handling, external validation, and a labeled evaluation set before promotion. See `docs/RESEARCH_TRACK.md` for the public-dataset inventory, licensing notes, and the proposed experiment matrix.

## Connection to Clinevo

The same provider interface is exposed through the AI decision service. Local routing is:
owned checkpoint -> Laya -> optional Jev -> deterministic fallback.

After a ClinevoOne checkpoint is trained and placed at the configured path, DECISION_BACKEND=auto automatically gives the owned model first opportunity. Existing human-review and provenance controls remain unchanged.