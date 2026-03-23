"""
Node functions for the References agent LangGraph graph.

Nodes:
citation_extractor_node  — regex + LLM to pull raw citation strings from text
reference_parser_node    — structure each citation into a dict
reference_formatter_node — normalise, deduplicate, detect style, finalise list
"""

from __future__ import annotations

import json
import os
import re

from groq import Groq

from agents.references.state import ReferencesState

_CLIENT: Groq | None = None


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _CLIENT


# Citation Extractor

# Common patterns: [1], [Author, 2023], (Author et al., 2023), numbered bibliography lines
_INLINE_PATTERNS = [
    r"\[[\w\s,\.]+\d{4}[a-z]?\]",          # [Smith, 2023] / [Smith et al., 2023a]
    r"\(\w[\w\s,\.]+,\s*\d{4}[a-z]?\)",     # (Smith, 2023)
    r"\[\d+\]",                              # [1], [23]
]

_BIBLIOGRAPHY_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\[\d+\]|\d+\.)\s+(.{30,250})",
    re.MULTILINE,
)

_EXTRACTOR_SYSTEM = """You are a citation extraction specialist.
Given an article, extract ALL reference / bibliography entries you can find.
Look for: numbered references, in-text citations, bibliography sections.

Return ONLY a JSON array of strings, each string being one raw citation as it appears.
If you find no citations, return an empty array [].
No markdown, no preamble, no commentary."""


def citation_extractor_node(state: ReferencesState) -> ReferencesState:
    text = state.get("text", "")

    # Step 1: regex pre-scan for bibliography lines
    regex_hits = [m.group(1).strip() for m in _BIBLIOGRAPHY_PATTERN.finditer(text)]

    # Step 2: LLM extraction on the last 3000 chars (references usually at end)
    tail = text[-3000:] if len(text) > 3000 else text
    client = _get_client()

    feedback = state.get("reviewer_feedback")
    system = _EXTRACTOR_SYSTEM
    if feedback:
        system += f"\n\nREVIEWER FEEDBACK: {feedback}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Article tail section:\n\n{tail}"},
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

    return {**state, "raw_citations": merged}


# Reference Parser

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
    raw_citations = state.get("raw_citations", [])
    if not raw_citations:
        return {**state, "structured_refs": []}

    client = _get_client()

    # Process in batches of 20 to stay within token limits
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
            # fallback: keep as raw-only objects
            for citation in batch:
                structured.append({"raw": citation})

    return {**state, "structured_refs": structured}


# Reference Formatter

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
        system += f"\n\nREVIEWER FEEDBACK: {feedback}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(structured_refs[:40])},  # cap at 40
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
