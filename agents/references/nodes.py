"""
Node functions for the References agent LangGraph graph.

Nodes:
  citation_extractor_node  — section-aware extraction + LLM, capped at 40 raw citations
  reference_parser_node    — structure each citation into a dict; SKIPPED on revisions
  reference_formatter_node — normalise, deduplicate, detect style, finalise list

Cost optimisations vs the original:
  1. Section-aware extraction: we detect a References/Bibliography header first and
     only fall back to the last 3 000 chars if no header is found.  This prevents the
     regex from matching numbered list items throughout the full document.
  2. Hard-cap: raw_citations is truncated to MAX_RAW_CITATIONS before the parser loop
     runs, so the batch-loop never exceeds ceil(40/20) = 2 API calls.
  3. Revision short-circuit: when reviewer_feedback is present AND structured_refs is
     already populated, citation_extractor_node and reference_parser_node both return
     immediately without making any API calls.  Only reference_formatter_node re-runs—
     which is a single call—to act on the reviewer's formatting feedback.
"""

from __future__ import annotations

import json
import os
import re

from groq import Groq

from agents.references.state import ReferencesState

_CLIENT: Groq | None = None

MAX_RAW_CITATIONS = 40          # hard-cap before batch-parser loop


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _CLIENT


# ── Helpers ───────────────────────────────────────────────────────────────────

# Compile once at import time
_SECTION_HEADER = re.compile(
    r"(?:^|\n)\s*(?:references|bibliography|works cited|literature cited|citations)"
    r"\s*[\n:]",
    re.IGNORECASE,
)

_BIBLIOGRAPHY_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\[\d+\]|\d+\.)\s+(.{30,250})",
    re.MULTILINE,
)


def _extract_reference_section(text: str) -> str:
    """
    Return the slice of *text* most likely to contain the bibliography.

    Strategy (in order):
      1. Find the last occurrence of a References/Bibliography section header
         and take everything from there to the end.
      2. Fall back to the last 3 000 characters.
    """
    match = None
    for m in _SECTION_HEADER.finditer(text):
        match = m          # keep the *last* match in case there are several

    if match:
        return text[match.start():]

    return text[-3000:] if len(text) > 3000 else text


# ── Citation Extractor ────────────────────────────────────────────────────────

_EXTRACTOR_SYSTEM = """You are a citation extraction specialist.
Given a reference section from an academic article, extract ALL reference / bibliography
entries you can find.  Look for: numbered references, in-text citations, bibliography
sections.

Return ONLY a JSON array of strings, each string being one raw citation as it appears.
If you find no citations, return an empty array [].
No markdown, no preamble, no commentary."""


def citation_extractor_node(state: ReferencesState) -> ReferencesState:
    # ── Revision short-circuit ────────────────────────────────────────────────
    # If this is a re-run triggered by the reviewer AND we already have
    # structured_refs from the first pass, skip extraction entirely.
    if state.get("reviewer_feedback") and state.get("structured_refs"):
        return state          # pass through; reference_parser_node will also skip

    text = state.get("text", "")
    ref_section = _extract_reference_section(text)

    # Step 1: regex pre-scan (scoped to the reference section only)
    regex_hits = [m.group(1).strip() for m in _BIBLIOGRAPHY_PATTERN.finditer(ref_section)]

    # Step 2: LLM extraction on the same section
    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _EXTRACTOR_SYSTEM},
            {"role": "user", "content": f"Reference section:\n\n{ref_section}"},
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        llm_hits = json.loads(raw)
        if not isinstance(llm_hits, list):
            llm_hits = []
    except json.JSONDecodeError:
        llm_hits = []

    # Merge regex + LLM hits, deduplicate by first 60 chars
    seen: set[str] = set()
    merged: list[str] = []
    for citation in regex_hits + llm_hits:
        key = str(citation)[:60].lower().strip()
        if key and key not in seen:
            seen.add(key)
            merged.append(str(citation).strip())

    # ── Hard-cap ──────────────────────────────────────────────────────────────
    merged = merged[:MAX_RAW_CITATIONS]

    return {**state, "raw_citations": merged}


# ── Reference Parser ──────────────────────────────────────────────────────────

_PARSER_SYSTEM = """You are a bibliographic data parser.
Given a list of raw citation strings, parse each into a structured object.

For each citation return:
{
  "authors": ["Last, F.", ...],   // list of author strings; [] if unknown
  "year": "2023",                 // string or null
  "title": "Paper title",        // string or null
  "venue": "Journal / Conference name",  // string or null
  "volume": "12",                // string or null
  "pages": "100-110",            // string or null
  "doi": "10.xxxx/...",          // string or null
  "raw": "original citation string"
}

Return ONLY a JSON array of these objects (one per input citation).
No markdown, no preamble."""


def reference_parser_node(state: ReferencesState) -> ReferencesState:
    # ── Revision short-circuit ────────────────────────────────────────────────
    if state.get("reviewer_feedback") and state.get("structured_refs"):
        return state          # structured_refs already available; skip to formatter

    raw_citations = state.get("raw_citations", [])
    if not raw_citations:
        return {**state, "structured_refs": []}

    client = _get_client()

    # Process in batches of 20 — with the 40-item cap, this is at most 2 calls
    structured: list[dict] = []
    batch_size = 20

    for i in range(0, len(raw_citations), batch_size):
        batch = raw_citations[i : i + batch_size]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _PARSER_SYSTEM},
                {"role": "user", "content": json.dumps(batch)},
            ],
            temperature=0.0,
            max_tokens=2048,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                structured.extend(parsed)
        except json.JSONDecodeError:
            for citation in batch:
                structured.append({"raw": citation})

    return {**state, "structured_refs": structured}


# ── Reference Formatter ───────────────────────────────────────────────────────

_FORMATTER_SYSTEM = """You are a reference list editor.
Given structured reference data, do three things:
1. Detect the citation style ("APA", "IEEE", "Vancouver", "MLA", "Chicago", or "Mixed/Unknown").
2. Deduplicate references (same title or same DOI = duplicate; keep one).
3. Format each reference cleanly in the detected style.

Return ONLY a JSON object:
{
  "citation_style": "APA",
  "references": [
    {
      "formatted": "Full formatted reference string",
      "authors": [...],
      "year": "...",
      "title": "...",
      "venue": "...",
      "doi": "..."
    },
    ...
  ]
}
No markdown, no preamble."""


def reference_formatter_node(state: ReferencesState) -> ReferencesState:
    structured_refs = state.get("structured_refs", [])
    if not structured_refs:
        return {**state, "final_references": [], "citation_style": "Unknown"}

    client = _get_client()
    feedback = state.get("reviewer_feedback")

    system = _FORMATTER_SYSTEM
    if feedback:
        system += f"\n\nREVIEWER FEEDBACK (apply this to the formatting): {feedback}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(structured_refs[:MAX_RAW_CITATIONS])},
        ],
        temperature=0.1,
        max_tokens=4096,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}

    return {
        **state,
        "final_references": parsed.get("references", []),
        "citation_style": parsed.get("citation_style", "Unknown"),
    }
