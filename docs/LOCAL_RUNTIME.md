# Clinevo Local Agent Runtime

Clinevo now has a local-first control/data/evidence split inspired by the operational shape of LARRI and the typed-decision approach exposed by Laya and Jev.

## Runtime graph

```
                    +---------------------------+
                    |   Clinevo Agent Control   |
                    | pause/resume/reconcile     |
                    | durable state + events     |
                    +-------------+-------------+
                                  |
                                  v
[mailbox] -> [Spring durable jobs] -> [AI service]
                                  |       |
                                  |       +--> System One decision
                                  |       |    Laya (local)
                                  |       |    or Jev (hosted)
                                  |       |
                                  |       +--> OCR/PDF/tables/images
                                  |       +--> optional structured LLM
                                  |
                                  v
                         [Oracle evidence store]
                                  |
                                  v
                        [Angular reviewer UI]
```

## System One decision layer

The AI service sends one state plus a set of typed questions to the configured decision provider.

The built-in inbox questions cover:

- ICSR signal
- PQC signal
- MI signal
- Not-relevant signal
- review urgency
- human-review requirement
- operational route

All questions are evaluated in one request and returned as typed values with probabilities/confidence metadata where the provider exposes them.

## Local Laya mode

Docker Compose defaults to:

```text
DECISION_BACKEND=laya
DECISION_FALLBACK=true
LAYA_LOAD_MODE=lazy
LAYA_MAX_LOADED=2
```

Laya is an open-weight Apache-2.0 decision model family. The local adapter uses the published `Router` API. Laya can automatically route multilingual state to the appropriate checkpoint; the cache is mounted as a Docker volume so weights do not need to be downloaded into the image on every rebuild.

The Docker image uses a single Uvicorn worker deliberately. A multi-worker configuration would initialize a separate Python process and can multiply model memory usage.

For a faster but higher-memory developer demo, set:

```text
LAYA_LOAD_MODE=preload
```

For CPU-constrained machines, keep lazy loading and `LAYA_MAX_LOADED=1`.

## Optional Jev mode

The same adapter contract can call the TypeSafe API:

```text
DECISION_BACKEND=jev
JEV_API_KEY=<your-key>
JEV_MODEL=jev-latest
```

Jev is hosted rather than locally self-hosted; there are no public weights or supported offline binary. The Clinevo architecture therefore treats Jev as a remote provider, not as the local model.

## Provider policy

```
DECISION_BACKEND=laya
        |
        +-- success --> typed System One decision
        |
        +-- failure and DECISION_FALLBACK=true
                         |
                         v
                    deterministic
```

The decision trace is returned on every AI text/PDF processing result with provider, model, latency, routing metadata and whether fallback was used.

The System One signal is intentionally not used to invent clinical facts. Existing source-grounded extraction and provenance rules remain authoritative, and the human reviewer remains required under the current pv-1 policy. The latest provider/model/fallback metadata is persisted in Oracle as AGENT_DECISION evidence and included in the agent's review-required event.

## One-command local run

From repository root:

```bash
make local-up
```

This:

1. creates `.env` from `.env.example` if needed
2. builds/starts Oracle, ClamAV, AI, backend and frontend
3. waits for AI/backend readiness
4. calls the decision endpoint once to warm/validate the configured decision provider
5. prints the local URLs

Other commands:

```bash
make local-down
make status
make decision-health
bash scripts/clinevo-local.sh doctor
bash scripts/clinevo-local.sh logs
```

## Provider switching

Local Laya:

```text
DECISION_BACKEND=laya
DECISION_FALLBACK=true
```

Jev with deterministic fallback:

```text
DECISION_BACKEND=jev
DECISION_FALLBACK=true
JEV_API_KEY=...
```

Strict offline deterministic mode:

```text
DECISION_BACKEND=deterministic
DECISION_FALLBACK=false
```

The last mode is the CI/fixture baseline because it is repeatable and requires no model weights or network provider.

## Operational lifecycle

The Clinevo agent lifecycle is deliberately explicit:

```
OBSERVED
   -> ANALYZING
   -> WAITING_REVIEW
   -> FINALIZED
```

Failures terminate in `FAILED`. Pause/resume only controls the agent control plane; durable inbox and processing jobs are retained. `FINALIZED` means the reviewer workflow reached a terminal decision, not that Clinevo transmitted a regulatory submission.

This is the local equivalent of the LARRI pattern: explicit state, durable work, reconciliation, operator controls, and a stable runtime boundary around pluggable model providers.
