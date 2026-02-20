# Tag Extractor — AI Publication Tagging Subsystem

Part of the **Agentic Authoring Assistant** project.

## Overview

The Tag Extractor uses a LangGraph workflow with three parallel extraction methods — fused by an LLM aggregator — to produce the top 10 publication tags for any AI or healthcare article.

```
START → [Gazetteer | spaCy NER | LLM Extractor] (parallel) → Aggregator → END
```

| Node | Method | Strength |
|------|--------|----------|
| `gazetteer_node` | Dictionary lookup | High precision, domain-curated |
| `spacy_node` | `en_core_web_md` NER | Catches novel proper nouns |
| `llm_extractor_node` | Groq `llama-3.3-70b-versatile` | Semantic understanding |
| `aggregator_node` | Groq LLM | Deduplication & ranking |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download spaCy model
```bash
python -m spacy download en_core_web_md
```

### 3. Set your Groq API key
Create a `.env` file in the `tag_extractor/` directory:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your key at: https://console.groq.com

## Usage

```bash
# Run on the built-in sample article
python main.py

# Run on a PDF (research paper, report, etc.)
python main.py --file paper.pdf

# Run on a Word document
python main.py --file manuscript.docx

# Run on a legacy Word document
python main.py --file old_report.doc

# Run on plain text
python main.py --file article.txt

# Run on inline text
python main.py --text "Your article content here..."
```

## Supported File Formats

| Format | Extension | Library Used | Notes |
|--------|-----------|-------------|-------|
| PDF | `.pdf` | `pdfplumber` (primary), `pypdf` (fallback) | Layout-aware extraction; image-only PDFs return empty |
| Word (modern) | `.docx` | `python-docx` | Extracts paragraphs + table cells |
| Word (legacy) | `.doc` | `mammoth` | No LibreOffice required |
| Plain text | `.txt` | Built-in | UTF-8 with latin-1 fallback |

## Output

Results are printed to the console and saved to `tag_extractor_output.json`:

```json
{
  "candidate_counts": {
    "gazetteer": 24,
    "spacy": 18,
    "llm": 31,
    "total_deduped": 52
  },
  "final_tags": [
    {
      "tag": "Retrieval-Augmented Generation",
      "category": "AI/ML Concept",
      "rationale": "Core method of the paper, referenced across all extractors."
    },
    ...
  ]
}
```

## Gazetteer Coverage

The curated gazetteer covers 5 domains:
- **AI/ML Terminology** — transformers, RAG, LoRA, RLHF, MoE, etc.
- **Notable AI Researchers** — Hinton, LeCun, Bengio, Karpathy, etc.
- **Organizations & Labs** — OpenAI, DeepMind, Anthropic, Hugging Face, etc.
- **Conferences & Journals** — NeurIPS, ICML, ICLR, JMLR, arXiv, etc.
- **Healthcare Journals** — NEJM, The Lancet, JAMA, BMJ, Nature Medicine, etc.

## Project Structure

```
tag_extractor/
├── main.py           # Entry point & CLI
├── graph.py          # LangGraph graph definition (wiring)
├── nodes.py          # All 6 node implementations
├── state.py          # TagState TypedDict schema
├── gazetteer.py      # Curated domain dictionary
├── file_reader.py    # PDF, DOCX, DOC, TXT ingestion layer
├── requirements.txt
└── README.md
```

## Extending

- **Add gazetteer terms**: Edit `gazetteer.py` — add to the `GAZETTEER` dict
- **Change top-N**: Set `top_n` in the initial state dict passed to `graph.invoke()`
- **Swap LLM**: Change the `model` param in `nodes.py` `ChatGroq(...)` calls
- **Add an extractor**: Add a node to `nodes.py`, register in `graph.py`, wire edges
