import os
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .decision_engine import decision_health, evaluate_custom, evaluate_inbox
from .literature import screen_literature_pdf
from .models import DecisionRequest, DecisionTrace, LiteratureBatchResult, ProcessingResult, TextProcessingRequest, TextProcessingResult
from .pdf_processor import process_pdf
from .text_processor import process_text

app = FastAPI(title="Clinevo Inbox AI Service", version="0.4.0", description="Synthetic-data-only document understanding and literature screening service for the candidate assignment.")


def _max_pdf_bytes() -> int:
    return int(os.getenv("MAX_PDF_BYTES", str(20 * 1024 * 1024)))


def _validate_pdf(file: UploadFile) -> None:
    content_type = (file.content_type or "").lower()
    if content_type != "application/pdf" and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported by the document AI service.")


async def _read_pdf(file: UploadFile) -> bytes:
    _validate_pdf(file)
    limit = _max_pdf_bytes()
    data = await file.read(limit + 1)
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty PDF: {file.filename or 'unnamed'}")
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"PDF exceeds the configured processing limit: {file.filename or 'unnamed'}")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail=f"File does not contain a valid PDF signature: {file.filename or 'unnamed'}")
    return data


@app.get("/decision/health")
def decision_provider_health() -> dict:
    return decision_health()


@app.post("/decision", response_model=DecisionTrace)
def decide(request: DecisionRequest) -> DecisionTrace:
    return evaluate_inbox(request.state) if request.questions is None else evaluate_custom(request.state, request.questions)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "clinevo-ai-service"}


@app.post("/process", response_model=ProcessingResult)
async def process_document(file: UploadFile = File(...), email_text: str = Form(default="")) -> ProcessingResult:
    data = await _read_pdf(file)
    try:
        return process_pdf(file.filename or "attachment.pdf", data, email_text=email_text)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Document processing failed") from exc


@app.post("/process-text", response_model=TextProcessingResult)
def process_email_body(request: TextProcessingRequest) -> TextProcessingResult:
    return process_text(request.source_name, request.text)


@app.post("/literature/screen", response_model=LiteratureBatchResult)
async def screen_literature(files: list[UploadFile] = File(...)) -> LiteratureBatchResult:
    if not files:
        raise HTTPException(status_code=400, detail="At least one article PDF is required")
    if len(files) > 25:
        raise HTTPException(status_code=400, detail="A maximum of 25 article PDFs can be screened in one batch")

    started = time.perf_counter()
    documents = []
    for file in files:
        data = await _read_pdf(file)
        try:
            documents.append(screen_literature_pdf(file.filename or "article.pdf", data))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Literature screening failed for {file.filename or 'article.pdf'}") from exc

    return LiteratureBatchResult(
        documents=documents,
        total_documents=len(documents),
        total_cases=sum(item.case_count for item in documents),
        processing_ms=int((time.perf_counter() - started) * 1000),
    )
