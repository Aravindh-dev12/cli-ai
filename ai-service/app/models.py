from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

Category = Literal["ICSR", "PQC", "MI", "NOT_RELEVANT"]
PdfType = Literal["DIGITAL", "SCANNED_OR_HANDWRITTEN", "PUBLISHED_ARTICLE", "NON_ENGLISH"]
TranslationStatus = Literal["NOT_REQUIRED", "TRANSLATED", "PARTIAL", "UNAVAILABLE"]
TranslationMethod = Literal["NOT_REQUIRED", "LLM", "SYNTHETIC_RULES", "UNAVAILABLE"]
ImageDescriptionMethod = Literal["OCR_TEXT", "METADATA_ONLY"]


class SourceRef(BaseModel):
    source_type: Literal["EMAIL", "PDF"]
    source_name: str
    page: int | None = None
    evidence: str | None = None


class ExtractedValue(BaseModel):
    value: str = "Not stated"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: SourceRef | None = None


class Classification(BaseModel):
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class TableData(BaseModel):
    page: int
    rows: list[list[str | None]]


class TranslationPage(BaseModel):
    page: int
    original_text: str
    translated_text: str


class TranslationInfo(BaseModel):
    applied: bool = False
    status: TranslationStatus = "NOT_REQUIRED"
    source_language: str = "unknown"
    target_language: str = "en"
    method: TranslationMethod = "NOT_REQUIRED"
    rationale: str
    requires_human_review: bool = False
    pages: list[TranslationPage] = Field(default_factory=list)


class ImageFinding(BaseModel):
    page: int
    description: str
    requires_human_review: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    method: ImageDescriptionMethod = "METADATA_ONLY"
    evidence_text: str | None = None
    width: int | None = None
    height: int | None = None


class DecisionTrace(BaseModel):
    provider: str
    model: str
    latency_ms: int = Field(ge=0)
    answers: dict[str, Any] = Field(default_factory=dict)
    model_routing: dict[str, Any] = Field(default_factory=dict)
    fallback_used: bool = False
    fallback_reason: str | None = None


class DecisionRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Any] | None = None


class AiDecision(BaseModel):
    classifications: list[Classification]
    summary: str
    extracted_facts: dict[str, dict[str, ExtractedValue]] = Field(default_factory=dict)


class ProcessingResult(BaseModel):
    file_name: str
    pdf_type: PdfType
    detected_language: str
    ocr_confidence: float | None = None
    classifications: list[Classification]
    summary: str
    translation: TranslationInfo
    tables: list[TableData] = Field(default_factory=list)
    images: list[ImageFinding] = Field(default_factory=list)
    extracted_facts: dict[str, dict[str, ExtractedValue]] = Field(default_factory=dict)
    processing_ms: int
    decision: DecisionTrace | None = None


class TextProcessingRequest(BaseModel):
    source_name: str = "email"
    text: str = ""


class TextProcessingResult(BaseModel):
    classifications: list[Classification]
    summary: str
    extracted_facts: dict[str, dict[str, ExtractedValue]] = Field(default_factory=dict)
    processing_ms: int
    decision: DecisionTrace | None = None


class LiteratureCaseResult(BaseModel):
    case_id: str
    reportable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    relevance_reason: str
    summary: str
    classifications: list[Classification] = Field(default_factory=list)
    extracted_facts: dict[str, dict[str, ExtractedValue]] = Field(default_factory=dict)
    source_pages: list[int] = Field(default_factory=list)


class LiteratureDocumentResult(BaseModel):
    file_name: str
    pdf_type: PdfType
    detected_language: str
    cases: list[LiteratureCaseResult] = Field(default_factory=list)
    case_count: int
    processing_ms: int


class LiteratureBatchResult(BaseModel):
    documents: list[LiteratureDocumentResult] = Field(default_factory=list)
    total_documents: int
    total_cases: int
    processing_ms: int
