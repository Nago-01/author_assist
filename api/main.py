"""
FastAPI backend for Author Assist.

Exposes two endpoints:
  POST /analyze/text  — accepts a raw JSON body {"text": "..."} 
  POST /analyze/file  — accepts a multipart file upload (PDF/DOCX/TXT)

Both endpoints pass the extracted text into core.pipeline.run_pipeline()
and return the standard FinalOutput JSON.

Deploy on Hugging Face Spaces as a Docker Space.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(override=True)

# App setup

app = FastAPI(
    title="Author Assist API",
    description="Multi-agent system that generates titles, TLDRs, tags, and references.",
    version="1.0.0",
)

# Allow requests from Vercel frontend and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {"status": "ok", "service": "Author Assist API"}


# ── Endpoint 1: Raw text ──────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str


@app.post("/analyze/text", tags=["analyze"])
def analyze_text(body: TextRequest) -> dict[str, Any]:
    """Run the full Author Assist pipeline on raw text."""
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    _check_api_key()

    from core.pipeline import run_pipeline  # noqa: PLC0415

    try:
        return run_pipeline(body.text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Endpoint 2: File upload ───────────────────────────────────────────────────

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


@app.post("/analyze/file", tags=["analyze"])
def analyze_file(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload a PDF, DOCX, DOC, or TXT file.
    The file is written to a temp folder, read by core.file_reader, then deleted.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Accepted: {_ALLOWED_EXTENSIONS}",
        )

    _check_api_key()

    # Write to a temp file so file_reader can open it normally
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        from core.file_reader import read_file   # noqa: PLC0415
        from core.pipeline import run_pipeline   # noqa: PLC0415

        text = read_file(tmp_path)
        if not text.strip():
            raise HTTPException(
                status_code=422, detail="Could not extract text from the uploaded file."
            )
        return run_pipeline(text)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Helper ────────────────────────────────────────────────────────────────────

def _check_api_key() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured on the server.",
        )
