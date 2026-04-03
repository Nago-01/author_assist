# Author Assist

**Coordinated multi-agent system for publication metadata generation.**

Automatically generates titles, TLDRs, tags, and formatted references from any article using a Manager → Parallel Agents → Reviewer architecture with targeted feedback loops. Now completely accessible through a full-stack Next.js web app.

---

## ⚡️ Features

- **Multi-Format Ingestion**: Drag and drop PDF, DOCX, DOC, or TXT files.
- **Agentic Pipeline**: Employs an orchestration of 6 independent LLM agents working concurrently.
- **Reviewer Loop**: Includes an automated review phase that kicks back inadequate outputs (e.g., misformatted citations) with targeted feedback until they pass quality checks.
- **Cost-Optimized**: 
  - *Fast extraction*: Bypasses full-document scans by isolating reference sections.
  - *Revision short-circuits*: Bypasses re-extraction on minor formatting corrections, cutting retry API calls by 95%.
- **Client History**: Remembers your last 3 analyzed articles securely in your browser's local storage.
- **One-Click Deploy**: Production ready for Hugging Face Spaces (backend) and Vercel (frontend).

---

## Architecture

At the backend level (`core/pipeline.py`), the system coordinates:

```
        START
          │
       Manager          ← reads raw text once; builds SharedContext
          │
    ┌─────┼──────────────────┐
    ▼     ▼         ▼        ▼
 Title  TLDR      Tags   References   ← run in parallel via asyncio; all share context
    │     │         │        │
    └─────┴──────────────────┘
                   │
               Reviewer               ← per-agent verdicts
                   │
        ┌──────────┴──────────┐
        │   needs revision?   │
        │  only those agents  │       ← selective re-run loop
        │   re-run with       │
        │   targeted feedback │
        └──────────┬──────────┘
                   │
                  END
```

---

## Project Structure

```
author_assist/
│
├── api/                         ← FastAPI Backend
│   └── main.py                  │  Exposes /analyze/text and /analyze/file
│
├── frontend/                    ← Next.js App Router Web UI
│   ├── src/app/                 │  Main dashboard (page.tsx, globals.css)
│   ├── src/components/          │  Glassmorphic Dashboard & History UI
│   └── src/lib/                 │  localStorage history tracker
│
├── core/                        ← Shared contracts & orchestration (LangGraph)
│   ├── pipeline.py              │  Orchestrator: parallel fan-out + re-run loop
│   ├── manager.py               │  Manager node → builds SharedContext
│   ├── reviewer.py              │  Reviewer node → per-agent verdicts
│   └── file_reader.py           │  PDF / DOCX / DOC / TXT ingestion
│
├── agents/                      ← Four independent, reusable AI agents
│   ├── tags/                    │  gazetteer / spaCy NER / LLM / aggregator
│   ├── title/                   │  candidate_generator + title_selector
│   ├── tldr/                    │  key_points + drafter + refiner
│   └── references/              │  citation_extractor + parser + formatter
│
├── Dockerfile                   ← Deployment configuration for HF Spaces
├── main.py                      ← CLI entry point
└── requirements.txt
```

---

##  Running the Web App Locally

You will need two terminals running simultaneously.

### 1. Start the API (Terminal 1)
```bash
# Clone the repository and install requirements
git clone https://github.com/Nago-01/author_assist.git
cd author_assist
pip install -r requirements.txt
python -m spacy download en_core_web_md

# Configure environment keys
cp .env.example .env
# Edit .env and supply your GROQ_API_KEY

# Start FastAPI
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the UI (Terminal 2)
```bash
cd frontend
npm install
npm run dev
```

Open **[http://localhost:3000](http://localhost:3000)** in your browser to use the app.

---

## 🛠 Command Line (CLI) Usage

For developers preferring the terminal, Author Assist still works purely offline locally:

```bash
# Run on a PDF
python main.py --file paper.pdf

# Run on a Word document
python main.py --file manuscript.docx

# Verbose mode (shows agent extraction details and review verdicts)
python main.py --file paper.pdf --verbose

# Save direct to JSON
python main.py --file paper.pdf --output my_results.json
```

---

## Cloud Deployment

Author Assist is purposefully designed to be hosted for free on enterprise providers:

### Backend: Hugging Face Spaces (Docker)
1. Create a Blank Docker Space on Hugging Face.
2. Provide your `GROQ_API_KEY` as a Space Secret.
3. Push the `main` branch to Hugging Face. The `Dockerfile` natively builds the API on port `7860`.
*(Tip: I have included a GitHub Action in `.github/workflows/huggingface.yml` to automate pushes to your space!).*

### Frontend: Vercel
1. Import the repository into Vercel.
2. Select `frontend` as the Root Directory.
3. Add `NEXT_PUBLIC_API_URL` to Vercel's Environment Variables, pointing it to your Hugging Face space URL (e.g. `https://your-user-author-assist-api.hf.space`).
4. Deploy!

---

## Reusing Agents in Other Projects

Every core agent is fully decoupled. You can easily lift any individual agent into a separate project:

```python
from agents.tags.agent import TagsAgent
from core.base_agent import SharedContext

agent = TagsAgent()
context = SharedContext(
    raw_text="Your article...",
    key_themes=["machine learning"],
    target_audience="Researchers",
    main_message="Proposing a new system",
    domain="AI",
)

result = agent.run(context)
print(result.output["final_tags"])
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
