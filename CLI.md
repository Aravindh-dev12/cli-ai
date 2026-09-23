# CLI

**CLI (formerly Clinevo Smart Inbox Assistant)** is the production-oriented agent and decision runtime in this repository.

The codebase contains the ClinevoOne family, including the heavy **ClinevoOne-Frontier** research model, System-One-style typed decisions, bounded agent actions, RLCD-style calibration training, Laya interoperability, and an optional Jev comparison lane.

> Repository URL remains `Aravindh-dev12/clinevo-inbox-ai` until the GitHub repository administrator rename is performed. The product/model name is now **CLI**.

## Live comparison

The repository includes a GitHub Actions workflow that runs a reproducible synthetic comparison against local Laya and Jev when `TYPESAFE_API_KEY` is configured as a repository secret.

The benchmark measures accuracy, error rate, calibration error on confidence/correctness, and p50/p95 latency. It does not use Jev output for training or distillation.
