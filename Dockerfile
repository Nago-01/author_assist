# ── Hugging Face Spaces Docker deployment ─────────────────────────────────────
# Build context: project root (author_assist/)
# Runs: uvicorn api.main:app on port 7860 (HF Spaces default)

FROM python:3.11-slim

WORKDIR /app

# System deps for PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "fastapi[standard]" uvicorn python-multipart

# Download spaCy model
RUN python -m spacy download en_core_web_md

# Copy project source
COPY api/       ./api/
COPY core/      ./core/
COPY agents/    ./agents/

# HF Spaces exposes port 7860
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
