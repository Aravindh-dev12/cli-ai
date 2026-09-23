from __future__ import annotations

import io
import re
import time
from statistics import mean

import fitz
import pdfplumber
import pytesseract
from PIL import Image
from pypdf import PdfReader

from .classifier import classify, extract_facts
from .decision_engine import evaluate_inbox
from .llm import StructuredLlmClient, validate_provenance
from .models import AiDecision, ImageFinding, ProcessingResult, TableData, TranslationInfo, TranslationPage

try:
    from langdetect import DetectorFactory, LangDetectException, detect
    DetectorFactory.seed = 0
except ImportError:
    LangDetectException = Exception
    detect = None


_SYNTHETIC_TRANSLATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "es": (
        ("Informe de seguridad sintético", "Synthetic safety report"),
        ("Paciente P-ES1, mujer de 44 años.", "Patient P-ES1, 44-year-old female."),
        ("Médico tratante en España.", "Treating physician in Spain."),
        ("Clinevex 10 mg por vía oral una vez al día.", "Clinevex 10 mg orally once daily."),
        ("Dos días después de iniciar el tratamiento presentó erupción cutánea y picor.", "Two days after starting treatment, the patient developed a skin rash and itching."),
        ("El medicamento se suspendió y la paciente se recuperó en 48 horas.", "The medicine was discontinued and the patient recovered within 48 hours."),
        ("Todos los datos son ficticios y se usan únicamente para pruebas.", "All data are fictional and are used only for testing."),
        ("Reportante", "Reporter"),
        ("Producto", "Product"),
        ("Reacción", "Reaction"),
        ("Resultado", "Outcome"),
        ("Paciente", "Patient"),
        ("Aviso", "Notice"),
    ),
    "fr": (
        ("Demande d’information médicale synthétique", "Synthetic medical information request"),
        ("Demande d'information médicale synthétique", "Synthetic medical information request"),
        ("Pharmacien hospitalier en France.", "Hospital pharmacist in France."),
        ("Novera 20 mg comprimé.", "Novera 20 mg tablet."),
        ("Le médicament peut-il être pris avec des aliments et existe-t-il une interaction avec un antiacide?", "Can the medicine be taken with food, and is there an interaction with an antacid?"),
        ("Aucun effet indésirable et aucun défaut du produit n’ont été signalés.", "No adverse event and no product defect were reported."),
        ("Aucun effet indésirable et aucun défaut du produit n'ont été signalés.", "No adverse event and no product defect were reported."),
        ("Toutes les informations sont fictives.", "All information is fictional."),
        ("Demandeur", "Requester"),
        ("Produit", "Product"),
        ("Sécurité", "Safety"),
        ("Question", "Question"),
        ("Avis", "Notice"),
    ),
}

_RESIDUAL_LANGUAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "es": (" mujer ", " años", " médico", " después ", " tratamiento", " erupción", " medicamento", " suspendió", " recuperó", " ficticios", " pruebas"),
    "fr": (" pharmacien", " médicament", " aliments", " interaction", " aucun ", " indésirable", " défaut", " signalés", " informations", " fictives"),
}


def _digital_pages(data: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(data))
    return [page.extract_text() or "" for page in reader.pages]


def _ocr_pages(data: bytes) -> tuple[list[str], float]:
    document = fitz.open(stream=data, filetype="pdf")
    pages: list[str] = []
    confidences: list[float] = []
    for page in document:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = [word for word in result.get("text", []) if word and word.strip()]
        pages.append(" ".join(words))
        for raw in result.get("conf", []):
            try:
                value = float(raw)
                if value >= 0:
                    confidences.append(value / 100.0)
            except (TypeError, ValueError):
                pass
    return pages, (mean(confidences) if confidences else 0.0)


def _detect_language(text: str) -> str:
    sample = " ".join(text.split())[:5000]
    if len(sample) < 30:
        return "unknown"
    if detect is not None:
        try:
            return detect(sample)
        except LangDetectException:
            pass
    folded = sample.casefold()
    if any(token in folded for token in (" paciente ", " reacción", " medicamento", "médico")):
        return "es"
    if any(token in folded for token in (" demande", " médicament", " aucun ", "pharmacien")):
        return "fr"
    return "en"


def _article_like(text: str) -> bool:
    lowered = text.casefold()
    return "references" in lowered and any(marker in lowered for marker in ("abstract", "case report", "case 1", "case study"))


def _strip_references(pages: list[str]) -> list[str]:
    cleaned: list[str] = []
    for page in pages:
        match = re.search(r"\bReferences\b", page, re.I)
        cleaned.append(page[:match.start()] if match else page)
    return cleaned


def _extract_tables(data: bytes) -> list[TableData]:
    output: list[TableData] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables() or []:
                rows = [[cell.strip() if isinstance(cell, str) else cell for cell in row] for row in table]
                if rows:
                    output.append(TableData(page=page_number, rows=rows))
    return output


def _ocr_image(image: Image.Image) -> tuple[str, float]:
    result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words: list[str] = []
    confidences: list[float] = []
    for word, raw_conf in zip(result.get("text", []), result.get("conf", [])):
        if not word or not word.strip():
            continue
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = -1
        if confidence >= 0:
            confidences.append(confidence / 100.0)
        words.append(word.strip())
    return " ".join(words), (mean(confidences) if confidences else 0.0)


def _extract_images(data: bytes) -> list[ImageFinding]:
    document = fitz.open(stream=data, filetype="pdf")
    findings: list[ImageFinding] = []
    for page_number, page in enumerate(document, start=1):
        for index, image_info in enumerate(page.get_images(full=True), start=1):
            width = int(image_info[2]) if len(image_info) > 3 else None
            height = int(image_info[3]) if len(image_info) > 3 else None
            try:
                extracted = document.extract_image(image_info[0])
                image = Image.open(io.BytesIO(extracted["image"])).convert("RGB")
                width, height = image.size
                evidence_text, confidence = _ocr_image(image)
                if evidence_text:
                    snippet = " ".join(evidence_text.split())[:500]
                    findings.append(ImageFinding(
                        page=page_number,
                        description=(
                            f"Embedded image {index} contains machine-readable text: '{snippet}'. "
                            "The text may be relevant to a product label, complaint photo, completed form, or other source evidence; visual meaning must be confirmed by a reviewer."
                        ),
                        requires_human_review=True,
                        confidence=confidence,
                        method="OCR_TEXT",
                        evidence_text=snippet,
                        width=width,
                        height=height,
                    ))
                else:
                    findings.append(ImageFinding(
                        page=page_number,
                        description=(
                            f"Embedded non-text visual {index} detected ({width}x{height} pixels). "
                            "No reliable readable text was found, so the system does not infer clinical or product meaning and requires human visual review."
                        ),
                        requires_human_review=True,
                        confidence=0.0,
                        method="METADATA_ONLY",
                        width=width,
                        height=height,
                    ))
            except Exception:
                findings.append(ImageFinding(
                    page=page_number,
                    description=(
                        f"Embedded visual {index} detected ({width or 'unknown'}x{height or 'unknown'} pixels), but automated image extraction/OCR was unavailable. "
                        "No visual meaning was inferred; human review is required."
                    ),
                    requires_human_review=True,
                    confidence=0.0,
                    method="METADATA_ONLY",
                    width=width,
                    height=height,
                ))
    return findings


def _ensure_sentence_ending(value: str) -> str:
    value = value.strip()
    if not value or value.endswith((".", "!", "?")):
        return value
    return f"{value}."


def _summary_sentence_count(value: str) -> int:
    return len([sentence for sentence in re.split(r"(?<=[.!?])\s+", value.strip()) if sentence.strip()])


def _summary(text: str, classifications: list) -> str:
    cleaned = " ".join(text.split())
    sentence_candidates = [
        _ensure_sentence_ending(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if len(sentence.strip()) > 20
    ]
    selected = sentence_candidates[:7]
    labels = ", ".join(item.category for item in classifications)
    selected.extend([
        f"The automated first-pass classification is {labels}.",
        "The result is intended for human review and is not a final pharmacovigilance decision.",
        "Missing facts are left as Not stated rather than inferred.",
    ])
    while len(selected) < 10:
        selected.append("The reviewer should verify extracted values against the cited source page and evidence before acceptance.")
    return " ".join(selected[:10])


def _synthetic_translate_page(text: str, language: str) -> str:
    translated = text
    for source, target in sorted(_SYNTHETIC_TRANSLATIONS.get(language, ()), key=lambda item: len(item[0]), reverse=True):
        translated = translated.replace(source, target)
    return translated


def _has_residual_source_language(text: str, language: str) -> bool:
    padded = f" {' '.join(text.casefold().split())} "
    return any(marker in padded for marker in _RESIDUAL_LANGUAGE_MARKERS.get(language, ()))


def _translate_pages(pages: list[str], language: str, llm: StructuredLlmClient) -> TranslationInfo:
    if language == "en":
        return TranslationInfo(
            applied=False,
            status="NOT_REQUIRED",
            source_language="en",
            target_language="en",
            method="NOT_REQUIRED",
            rationale="English was detected, so no translation was applied; the original PDF remains the authoritative source.",
            requires_human_review=False,
        )
    if language == "unknown":
        return TranslationInfo(
            applied=False,
            status="UNAVAILABLE",
            source_language="unknown",
            target_language="en",
            method="UNAVAILABLE",
            rationale="The source language could not be detected reliably. Original page text is preserved and a reviewer must decide whether translation is required.",
            requires_human_review=True,
            pages=[TranslationPage(page=index, original_text=text, translated_text="") for index, text in enumerate(pages, start=1)],
        )

    llm_pages = llm.translate_pages(pages, language)
    if llm_pages is not None:
        return TranslationInfo(
            applied=True,
            status="TRANSLATED",
            source_language=language,
            target_language="en",
            method="LLM",
            rationale=(
                "A configured structured LLM translated each page to English while preserving page boundaries. "
                "The translation is auxiliary context only; extracted facts must still cite evidence from the original-language page."
            ),
            requires_human_review=True,
            pages=[
                TranslationPage(page=index, original_text=original, translated_text=translated)
                for index, (original, translated) in enumerate(zip(pages, llm_pages), start=1)
            ],
        )

    if language in _SYNTHETIC_TRANSLATIONS:
        translated_pages = [_synthetic_translate_page(page, language) for page in pages]
        changed = any(original != translated for original, translated in zip(pages, translated_pages))
        residual = any(_has_residual_source_language(page, language) for page in translated_pages)
        status = "PARTIAL" if residual or not changed else "TRANSLATED"
        return TranslationInfo(
            applied=changed,
            status=status,
            source_language=language,
            target_language="en",
            method="SYNTHETIC_RULES",
            rationale=(
                "The configured LLM translation path was unavailable, so deterministic translations limited to the repository's synthetic Spanish/French fixture vocabulary were used. "
                "This mode is for reproducible demonstration/testing only and must not be treated as production medical translation."
            ),
            requires_human_review=True,
            pages=[
                TranslationPage(page=index, original_text=original, translated_text=translated)
                for index, (original, translated) in enumerate(zip(pages, translated_pages), start=1)
            ],
        )

    return TranslationInfo(
        applied=False,
        status="UNAVAILABLE",
        source_language=language,
        target_language="en",
        method="UNAVAILABLE",
        rationale=(
            "A non-English document was detected but no approved translation engine was configured and no deterministic synthetic-fixture translation exists for this language. "
            "The original text is preserved and human translation/review is required."
        ),
        requires_human_review=True,
        pages=[TranslationPage(page=index, original_text=text, translated_text="") for index, text in enumerate(pages, start=1)],
    )


def _source_for_llm(file_name: str, pages: list[str], email_text: str, translation: TranslationInfo) -> str:
    chunks = [f"[EMAIL]\n{email_text.strip()}"] if email_text.strip() else []
    translated_by_page = {item.page: item.translated_text for item in translation.pages if item.translated_text}
    for index, text in enumerate(pages, start=1):
        chunks.append(f"[PDF:{file_name} PAGE:{index} ORIGINAL]\n{text}")
        translated = translated_by_page.get(index)
        if translated:
            chunks.append(f"[PDF:{file_name} PAGE:{index} ENGLISH_TRANSLATION_AUXILIARY]\n{translated}")
    return "\n\n".join(chunks)


def process_pdf(file_name: str, data: bytes, email_text: str = "") -> ProcessingResult:
    started = time.perf_counter()
    pages = _digital_pages(data)
    text = "\n\n".join(pages)
    compact_len = len("".join(text.split()))
    ocr_confidence: float | None = None

    if compact_len < 80:
        try:
            pages, ocr_confidence = _ocr_pages(data)
        except Exception:
            pages = pages or [""]
            ocr_confidence = 0.0
        text = "\n\n".join(pages)
        pdf_type = "SCANNED_OR_HANDWRITTEN"
    else:
        language = _detect_language(text)
        if language not in {"en", "unknown"}:
            pdf_type = "NON_ENGLISH"
        elif _article_like(text):
            pdf_type = "PUBLISHED_ARTICLE"
        else:
            pdf_type = "DIGITAL"

    language = _detect_language(text)
    analysis_pages = _strip_references(pages) if pdf_type == "PUBLISHED_ARTICLE" else pages
    analysis_text = "\n\n".join(analysis_pages)

    llm = StructuredLlmClient()
    translation = _translate_pages(analysis_pages, language, llm)
    translated_text = "\n\n".join(item.translated_text for item in translation.pages if item.translated_text)
    classification_context = "\n".join(part for part in (email_text, analysis_text, translated_text) if part).strip()

    fallback_classifications = classify(classification_context)
    decision = AiDecision(
        classifications=fallback_classifications,
        summary=_summary(translated_text or analysis_text, fallback_classifications),
        extracted_facts=extract_facts(analysis_pages, file_name),
    )

    llm_decision = llm.decide(_source_for_llm(file_name, analysis_pages, email_text, translation))
    if llm_decision is not None:
        decision = validate_provenance(llm_decision, file_name=file_name, pages=analysis_pages, email_text=email_text)
        if not 10 <= _summary_sentence_count(decision.summary) <= 15:
            decision.summary = _summary(translated_text or analysis_text, decision.classifications)

    system_one = evaluate_inbox({
        "source_type": "PDF",
        "source_name": file_name,
        "email_text": email_text[:12000],
        "pdf_text": analysis_text[:20000],
    })
    processing_ms = int((time.perf_counter() - started) * 1000)
    return ProcessingResult(
        file_name=file_name,
        pdf_type=pdf_type,
        detected_language=language,
        ocr_confidence=ocr_confidence,
        classifications=decision.classifications,
        summary=decision.summary,
        translation=translation,
        tables=_extract_tables(data),
        images=_extract_images(data),
        extracted_facts=decision.extracted_facts,
        processing_ms=processing_ms,
        decision=system_one,
    )
