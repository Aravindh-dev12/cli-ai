from __future__ import annotations

import os
import time
from typing import Any, Protocol

import httpx

from .classifier import classify
from .models import DecisionTrace


INBOX_DECISION_QUESTIONS: dict[str, dict[str, Any]] = {
    "icsr": {
        "type": "noul",
        "instructions": "Is this content a potential individual case safety report requiring pharmacovigilance review?",
        "criteria": {
            "true": "The content indicates an identifiable patient, a reporter or source, a suspect medicinal product, and an adverse event/reaction.",
            "false": "One or more minimum ICSR elements are absent or the content is not an adverse-event case.",
        },
    },
    "pqc": {
        "type": "noul",
        "instructions": "Is this content a potential product quality complaint?",
        "criteria": {
            "true": "The content reports a defect, damage, contamination, packaging problem, unexpected product appearance, counterfeit concern, or similar quality issue.",
            "false": "No supported product-quality issue is described.",
        },
    },
    "mi": {
        "type": "noul",
        "instructions": "Is this content primarily a medical-information request?",
        "criteria": {
            "true": "The content asks for medical/product information such as dose, administration, interactions, precautions, or use instructions without being primarily a safety or quality case.",
            "false": "The content is not primarily a medical-information question.",
        },
    },
    "not_relevant": {
        "type": "noul",
        "instructions": "Is this content unrelated to pharmacovigilance, product quality, or medical-information handling?",
        "criteria": {
            "true": "The message does not contain supported safety, quality, or medical-information content.",
            "false": "The message contains at least one supported pharmacovigilance, quality, or medical-information signal.",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgently should this item be reviewed by a human operator?",
        "criteria": [
            "Routine; no time-sensitive signal.",
            "Soon; review during normal queue handling.",
            "High; prioritize review because of material safety, quality, or escalation signals.",
            "Critical; immediate review is warranted because of a potentially severe or time-sensitive signal.",
        ],
    },
    "requires_human_review": {
        "type": "noul",
        "instructions": "Should a human reviewer remain in the loop before a terminal pharmacovigilance decision?",
        "criteria": {
            "true": "The item involves regulated safety/quality/medical-information interpretation, uncertain evidence, translation, OCR, or any decision that should remain supervised.",
            "false": "No regulated interpretation is required and an explicit local policy permits autonomous handling.",
        },
    },
    "route": {
        "type": "choice",
        "instructions": "Which operational workstream should own this item first?",
        "criteria": {
            "safety": "Potential ICSR/adverse-event content.",
            "quality": "Potential product-quality complaint.",
            "medical_information": "Medical-information question.",
            "general": "No supported regulated workstream.",
        },
    },
}


class DecisionProvider(Protocol):
    name: str

    def predict(self, state: Any, questions: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        ...


def _deterministic_answers(text: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    classifications = classify(text)
    by_category = {item.category: item.confidence for item in classifications}
    icsr = by_category.get("ICSR", 0.02)
    pqc = by_category.get("PQC", 0.02)
    mi = by_category.get("MI", 0.02)
    not_relevant = by_category.get("NOT_RELEVANT", 0.02)

    if "ICSR" in by_category:
        route = "safety"
    elif "PQC" in by_category:
        route = "quality"
    elif "MI" in by_category:
        route = "medical_information"
    else:
        route = "general"

    urgency = 3.0 if any(token in text.casefold() for token in ("death", "life-threatening", "hospitalized", "hospitalised")) else (
        2.0 if any(token in text.casefold() for token in ("serious", "urgent", "immediate", "critical")) else 1.0
    )

    answers = {
        "icsr": {"type": "noul", "noul": float(icsr)},
        "pqc": {"type": "noul", "noul": float(pqc)},
        "mi": {"type": "noul", "noul": float(mi)},
        "not_relevant": {"type": "noul", "noul": float(not_relevant)},
        "urgency": {
            "type": "score",
            "score": urgency,
            "confidence": 0.50,
            "legend": {"0": "Routine", "1": "Soon", "2": "High", "3": "Critical"},
            "probabilities": {},
        },
        "requires_human_review": {"type": "noul", "noul": 1.0},
        "route": {
            "type": "choice",
            "choice": route,
            "confidence": 0.75,
            "probabilities": {route: 1.0},
        },
    }
    return "deterministic", answers, {"model": "rule-engine", "reason": "local deterministic fallback"}


class LayaProvider:
    name = "laya"

    def __init__(self) -> None:
        self._router: Any | None = None

    def _router_instance(self) -> Any:
        if self._router is not None:
            return self._router

        os.environ.setdefault("USE_TF", "0")
        try:
            from laya import Router
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Laya is not installed; install the local decision dependencies") from exc

        load_mode = os.getenv("LAYA_LOAD_MODE", "lazy").strip().lower()
        if load_mode == "preload":
            self._router = Router(preload=True)
        else:
            max_loaded = max(1, int(os.getenv("LAYA_MAX_LOADED", "2")))
            self._router = Router(max_loaded=max_loaded)
        return self._router

    def predict(self, state: Any, questions: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        router = self._router_instance()
        model_override = os.getenv("LAYA_MODEL", "").strip()
        if model_override:
            result = router.predict(state, questions, model=model_override)
        else:
            result = router.predict(state, questions)
        return str(result.get("routing", {}).get("model", "laya")), result.get("answers", {}), result.get("routing", {})


class JevProvider:
    name = "jev"

    def predict(self, state: Any, questions: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        api_key = os.getenv("JEV_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("JEV_API_KEY is not configured")

        url = os.getenv("JEV_API_URL", "https://api.typesafe.ai/v1/systemone").strip()
        model = os.getenv("JEV_MODEL", "jev-latest").strip()
        timeout = float(os.getenv("JEV_TIMEOUT_SECONDS", "15"))

        response = httpx.post(
            url,
            json={"model": model, "state": state, "questions": questions},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        return str(body.get("model", model)), body.get("answers", {}), {"model": body.get("model", model)}


class DecisionEngine:
    def __init__(self) -> None:
        self.backend = os.getenv("DECISION_BACKEND", "laya").strip().lower()
        self.fallback_enabled = os.getenv("DECISION_FALLBACK", "true").strip().lower() == "true"
        self._providers: dict[str, DecisionProvider] = {
            "laya": LayaProvider(),
            "jev": JevProvider(),
            "deterministic": _DeterministicProvider(),
        }

    def evaluate(self, state: Any, questions: dict[str, Any] | None = None) -> DecisionTrace:
        question_set = questions or INBOX_DECISION_QUESTIONS
        provider_names = self._provider_order()

        failures: list[str] = []
        for provider_name in provider_names:
            provider = self._providers[provider_name]
            started = time.perf_counter()
            try:
                model, answers, routing = provider.predict(state, question_set)
                return DecisionTrace(
                    provider=provider_name,
                    model=model,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    answers=answers,
                    model_routing=routing,
                    fallback_used=bool(failures),
                    fallback_reason="; ".join(failures) if failures else None,
                )
            except Exception as exc:  # pragma: no cover - model/network dependent
                failures.append(f"{provider_name}: {str(exc)[:300]}")

        if self.fallback_enabled:
            provider = self._providers["deterministic"]
            started = time.perf_counter()
            model, answers, routing = provider.predict(state if isinstance(state, str) else _state_text(state), question_set)
            return DecisionTrace(
                provider="deterministic",
                model=model,
                latency_ms=int((time.perf_counter() - started) * 1000),
                answers=answers,
                model_routing=routing,
                fallback_used=True,
                fallback_reason="; ".join(failures) if failures else "configured deterministic fallback",
            )

        raise RuntimeError("No decision provider succeeded: " + "; ".join(failures))

    def _provider_order(self) -> list[str]:
        if self.backend == "auto":
            order = ["laya"]
            if os.getenv("JEV_API_KEY", "").strip():
                order.append("jev")
            order.append("deterministic")
            return order
        if self.backend not in self._providers:
            raise RuntimeError(f"Unsupported DECISION_BACKEND: {self.backend}")
        order = [self.backend]
        if self.backend != "deterministic" and self.fallback_enabled:
            order.append("deterministic")
        return order


class _DeterministicProvider:
    name = "deterministic"

    def predict(self, state: Any, questions: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        return _deterministic_answers(_state_text(state))


_ENGINE = DecisionEngine()


def _state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        parts: list[str] = []
        for key, value in state.items():
            if value in (None, ""):
                continue
            if isinstance(value, list):
                value = "\n".join(str(item) for item in value)
            elif isinstance(value, dict):
                value = " ".join(f"{k}: {v}" for k, v in value.items())
            parts.append(f"{key}: {value}")
        return "\n".join(parts)
    return str(state)


def evaluate_inbox(state: Any) -> DecisionTrace:
    return _ENGINE.evaluate(state, INBOX_DECISION_QUESTIONS)


def evaluate_custom(state: Any, questions: dict[str, Any]) -> DecisionTrace:
    return _ENGINE.evaluate(state, questions)


def decision_health() -> dict[str, Any]:
    return {
        "backend": _ENGINE.backend,
        "fallbackEnabled": _ENGINE.fallback_enabled,
        "layaInstalled": "laya" in __import__("sys").modules,
        "jevConfigured": bool(os.getenv("JEV_API_KEY", "").strip()),
        "questions": list(INBOX_DECISION_QUESTIONS.keys()),
    }
