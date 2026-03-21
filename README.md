# Author Assist

**Coordinated multi-agent system for academic publication metadata generation.**

Automatically generates titles, TLDRs, tags, and formatted references from any article —
using a Manager → Parallel Agents → Reviewer architecture with targeted feedback loops.

---

## Architecture

```
        START
          │
       Manager          ← reads raw text once; builds SharedContext
          │
    ┌─────┼──────────────────┐
    ▼     ▼         ▼        ▼
 Title  TLDR      Tags   References   ← run in parallel; all share same context
    │     │         │        │
    └─────┴──────────────────┘
                   │
               Reviewer              ← per-agent verdicts (not pass/fail)
                   │
        ┌──────────┴──────────┐
        │   needs revision?   │
        │  only those agents  │      ← selective re-run loop (max 3 rounds)
        │   re-run with       │
        │   targeted feedback │
        └──────────┬──────────┘
                   │
                  END
```

### How it works

1. **Manager** — reads the article once, extracts key themes, audience, domain, and main
   message into a `SharedContext`. Every agent receives this context before starting.
   No agent ever reads the raw text in isolation.

2. **Parallel agents** — all four agents run concurrently via `asyncio.gather`.
   Each is independent, stateless, and ignorant of the others.

3. **Reviewer** — evaluates every agent's output against the `SharedContext` and returns
   a per-agent verdict: either `"approved"` or specific targeted feedback
   (e.g. *"The TLDR does not mention the benchmarking methodology"*).

4. **Selective re-run** — only agents that received feedback re-run, with that feedback
   passed directly into their next execution. Approved agents are never re-run.
   This loop repeats up to `MAX_REVISIONS = 3` times.

---

## Project Structure

```
author_assist/
│
├── core/                        ← shared contracts & orchestration
│   ├── base_agent.py            │  SharedContext, AgentResult, BaseAgent ABC
│   ├── state.py                 │  PipelineState TypedDict (top-level graph)
│   ├── manager.py               │  Manager node → builds SharedContext
│   ├── reviewer.py              │  Reviewer node → per-agent verdicts
│   ├── pipeline.py              │  Orchestrator: parallel fan-out + re-run loop
│   ├── file_reader.py           │  PDF / DOCX / DOC / TXT ingestion
│   └── __init__.py              │
│
├── agents/                      ← four independent, reusable agents
│   ├── tags/
│   │   ├── agent.py             │  BaseAgent wrapper (public interface)
│   │   ├── graph.py             │  LangGraph wiring (parallel extraction)
│   │   ├── nodes.py             │  gazetteer / spaCy NER / LLM / aggregator nodes
│   │   ├── state.py             │  TagState TypedDict
│   │   ├── gazetteer.py         │  Curated domain dictionary
│   │   └── __init__.py          │
│   │
│   ├── title/
│   │   ├── agent.py             │  BaseAgent wrapper
│   │   ├── graph.py             │  LangGraph wiring
│   │   ├── nodes.py             │  candidate_generator + title_selector nodes
│   │   ├── state.py             │  TitleState TypedDict
│   │   └── __init__.py          │
│   │
│   ├── tldr/
│   │   ├── agent.py             │  BaseAgent wrapper
│   │   ├── graph.py             │  LangGraph wiring
│   │   ├── nodes.py             │  key_points + drafter + refiner nodes
│   │   ├── state.py             │  TLDRState TypedDict
│   │   └── __init__.py          │
│   │
│   └── references/
│       ├── agent.py             │  BaseAgent wrapper
│       ├── graph.py             │  LangGraph wiring
│       ├── nodes.py             │  citation_extractor + parser + formatter nodes
│       ├── state.py             │  ReferencesState TypedDict
│       └── __init__.py          │
│
├── main.py                      ← CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/Nago-01/author_assist.git
cd author_assist
pip install -r requirements.txt
```

### 2. Download spaCy model

```bash
python -m spacy download en_core_web_md
```

### 3. Set your Groq API key

```bash
cp .env.example .env
# Edit .env and add your key
```

Get your free key at: https://console.groq.com

---

## Usage

```bash
# Run on a PDF
python main.py --file paper.pdf

# Run on a Word document
python main.py --file manuscript.docx

# Run on plain text
python main.py --file article.txt

# Run on inline text
python main.py --text "Your article content here..."

# Verbose mode — shows extraction details and review verdicts
python main.py --file paper.pdf --verbose

# Custom output file
python main.py --file paper.pdf --output my_results.json
```

---

## Output

Results are printed to the console and saved to `author_assist_output.json`:

```json
{
  "title": {
    "final_title": "Benchmarking LLMs on Combinatorial Optimisation: A Systematic Survey",
    "rationale": "Precise, signals methodology (benchmarking) and domain.",
    "alternative_titles": ["...", "..."],
    "all_candidates": ["...", "...", "...", "...", "..."]
  },
  "tldr": {
    "final_tldr": "This survey evaluates 12 large language models...",
    "one_liner": "Systematic benchmark of LLMs on combinatorial optimisation tasks.",
    "key_points": ["...", "...", "..."]
  },
  "tags": {
    "final_tags": [
      {"tag": "Large Language Model", "category": "AI/ML Concept", "rationale": "..."},
      {"tag": "Benchmarking",         "category": "AI/ML Concept", "rationale": "..."}
    ],
    "candidate_counts": {"gazetteer": 18, "spacy": 12, "llm": 27, "total_deduped": 41}
  },
  "references": {
    "final_references": [
      {"formatted": "Brown, T. et al. (2020). Language models are few-shot learners...", ...}
    ],
    "citation_style": "APA",
    "total_references": 34,
    "raw_citations_found": 36
  },
  "meta": {
    "revision_rounds": 1,
    "review_verdicts": {
      "title_generator":      "approved",
      "tldr_generator":       "The TLDR should more clearly state the benchmarking methodology used.",
      "tags_generator":       "approved",
      "references_generator": "approved"
    },
    "shared_context": {
      "key_themes": ["LLM benchmarking", "combinatorial optimisation", "survey methodology"],
      "target_audience": "ML researchers and operations researchers",
      "main_message": "LLMs underperform specialised solvers on NP-hard combinatorial problems.",
      "domain": "AI/ML",
      "article_type": "survey"
    },
    "timestamp": "2026-03-21T10:00:00Z"
  }
}
```

---

## Reusing Agents in Other Projects

Every agent is fully self-contained. You can lift any one agent into a separate project:

```python
from agents.tags.agent import TagsAgent
from core.base_agent import SharedContext

agent = TagsAgent()

context = SharedContext(
    raw_text="Your article text here...",
    key_themes=["machine learning", "transformers"],
    target_audience="ML researchers",
    main_message="We propose a new attention mechanism.",
    domain="AI/ML",
)

result = agent.run(context)
print(result.output["final_tags"])
```

The agent's internal LangGraph graph (`graph.py`, `nodes.py`, `state.py`) is completely
unaware of the outer multi-agent system. Only `agent.py` bridges the two worlds.

---

## Extending the System

### Add a new agent

1. Create `agents/your_agent/` with `state.py`, `nodes.py`, `graph.py`, `agent.py`, `__init__.py`
2. Implement `BaseAgent` in `agent.py` (see any existing agent as a template)
3. Register in `core/pipeline.py` → `_build_registry()`
4. Add a key to `core/reviewer.py` → `_SYSTEM_PROMPT` and `reviewer_node`

No other files need to change.

### Extend the gazetteer

Edit `agents/tags/gazetteer.py` — add entries to the `GAZETTEER` dict. No code changes needed elsewhere.

### Swap the LLM

Change the `model` string in any `nodes.py` file. All agents use Groq by default
(`llama-3.3-70b-versatile`). Any Groq-compatible model works.

---

## LLM & Model

All agents use **Groq** with `llama-3.3-70b-versatile` by default.
- Fast inference (sub-second per node for most operations)
- Free tier available at https://console.groq.com
- Swap by changing the `model` parameter in any `nodes.py`
