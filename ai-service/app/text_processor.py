from __future__ import annotations

import re
import time

from .classifier import classify, extract_facts
from .decision_engine import evaluate_inbox
from .llm import StructuredLlmClient, validate_provenance
from .models import AiDecision, TextProcessingResult


def _summary(text: str, classifications: list) -> str:
    cleaned = " ".join(text.split())
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", cleaned) if len(item.strip()) > 20]
    selected = sentences[:7]
    labels = ", ".join(item.category for item in classifications)
    selected.extend([
        f"The automated first-pass classification is {labels}.",
        "This email-body result is intended for human review and is not a final pharmacovigilance decision.",
        "Missing facts are left as Not stated rather than inferred.",
    ])
    while len(selected) < 10:
        selected.append("The reviewer should verify extracted values against the original email evidence before acceptance.")
    return " ".join(selected[:10])


def _as_email_provenance(facts: dict, source_name: str) -> dict:
    for group in facts.values():
        for value in group.values():
            if value.source is not None:
                value.source = value.source.model_copy(update={
                    "source_type": "EMAIL",
                    "source_name": source_name,
                    "page": None,
                })
    return facts


def process_text(source_name: str, text: str) -> TextProcessingResult:
    started = time.perf_counter()
    classifications = classify(text)
    system_one = evaluate_inbox({
        "source_type": "EMAIL",
        "source_name": source_name,
        "body": text[:12000],
    })
    decision = AiDecision(
        classifications=classifications,
        summary=_summary(text, classifications),
        extracted_facts=_as_email_provenance(extract_facts([text], source_name), source_name),
    )

    llm_decision = StructuredLlmClient().decide(f"[EMAIL:{source_name}]\n{text}")
    if llm_decision is not None:
        decision = validate_provenance(llm_decision, file_name="", pages=[], email_text=text)

    return TextProcessingResult(
        classifications=decision.classifications,
        summary=decision.summary,
        extracted_facts=decision.extracted_facts,
        processing_ms=int((time.perf_counter() - started) * 1000),
        decision=system_one,
    )
