import os

from app.decision_engine import evaluate_inbox


os.environ.setdefault("USE_TF", "0")


def test_deterministic_fallback_has_typed_answers(monkeypatch):
    monkeypatch.setenv("DECISION_BACKEND", "deterministic")
    monkeypatch.setenv("DECISION_FALLBACK", "false")

    result = evaluate_inbox({
        "subject": "Quality complaint",
        "body": "The vial is leaking and the seal is broken.",
    })

    assert result.provider == "deterministic"
    assert result.answers["pqc"]["type"] == "noul"
    assert 0.0 <= result.answers["pqc"]["noul"] <= 1.0
    assert result.answers["route"]["choice"] in {"safety", "quality", "medical_information", "general"}


def test_deterministic_path_requires_human_review(monkeypatch):
    monkeypatch.setenv("DECISION_BACKEND", "deterministic")
    monkeypatch.setenv("DECISION_FALLBACK", "false")

    result = evaluate_inbox("Please tell me the dose for Clinevex 10 mg.")
    assert result.answers["requires_human_review"]["noul"] == 1.0
