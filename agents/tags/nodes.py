"""
agents/tags/nodes.py
--------------------
All node functions for the Tags LangGraph graph.

Nodes
-----
gazetteer_node   — dictionary-based, high precision
spacy_node       — NER-based, catches novel proper nouns
llm_extractor_node — semantic extraction via Groq LLM
aggregator_node  — deduplication, ranking, final top-10
"""

from __future__ import annotations

import json
import os

from groq import Groq

from agents.tags.gazetteer import lookup
from agents.tags.state import TagState

_CLIENT: Groq | None = None


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _CLIENT


# ── Node 1: Gazetteer ────────────────────────────────────────────────────────

def gazetteer_node(state: TagState) -> TagState:
    """Dictionary lookup — deterministic, high precision."""
    hits = lookup(state.get("text", ""))
    candidates = [name for name, _ in hits]
    return {"gazetteer_candidates": candidates}


# ── Node 2: spaCy NER ────────────────────────────────────────────────────────

def spacy_node(state: TagState) -> TagState:
    """Named Entity Recognition — catches proper nouns not in the gazetteer."""
    try:
        import spacy  # noqa: PLC0415

        try:
            nlp = spacy.load("en_core_web_md")
        except OSError:
            # Fallback: small model
            nlp = spacy.load("en_core_web_sm")

        doc = nlp(state.get("text", "")[:10000])  # limit for speed

        relevant_labels = {"ORG", "PRODUCT", "PERSON", "GPE", "EVENT", "WORK_OF_ART", "LAW"}
        seen: set[str] = set()
        candidates: list[str] = []

        for ent in doc.ents:
            if ent.label_ in relevant_labels:
                clean = ent.text.strip()
                if clean and clean.lower() not in seen and len(clean) > 2:
                    seen.add(clean.lower())
                    candidates.append(clean)

        return {"spacy_candidates": candidates[:30]}  # cap at 30

    except Exception:  # noqa: BLE001
        # spaCy not installed or model missing — skip gracefully
        return {"spacy_candidates": []}


# ── Node 3: LLM Extractor ────────────────────────────────────────────────────

_LLM_EXTRACTOR_SYSTEM = """You are a tag extraction specialist for academic and technical publications.
Given an article, extract the most important technical and conceptual tags.

Consider: key methods, datasets, frameworks, algorithms, application domains, and concepts.

Return ONLY a JSON array of objects. Each object: {"tag": "...", "category": "..."}
Category options: "AI/ML Concept", "Method", "Dataset", "Framework", "Application Domain",
"Organization", "Researcher", "Conference", "Healthcare", "Other"

Extract 15-25 tags. No markdown, no preamble."""


def llm_extractor_node(state: TagState) -> TagState:
    """LLM-based semantic extraction — understands context."""
    client = _get_client()
    text = state.get("text", "")[:5000]
    themes = state.get("key_themes", [])
    domain = state.get("domain", "General")
    feedback = state.get("reviewer_feedback")

    system_prompt = _LLM_EXTRACTOR_SYSTEM
    if feedback:
        system_prompt += f"\n\nIMPORTANT REVIEWER FEEDBACK: {feedback}"
    if themes:
        system_prompt += f"\n\nKey themes to prioritise: {', '.join(themes)}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Domain: {domain}\n\nArticle:\n{text}"},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        candidates = json.loads(raw)
        if not isinstance(candidates, list):
            candidates = []
    except json.JSONDecodeError:
        candidates = []

    return {"llm_candidates": candidates}


# ── Node 4: Aggregator ───────────────────────────────────────────────────────

_AGGREGATOR_SYSTEM = """You are a publication tag editor. You receive three lists of candidate tags
from different extraction methods. Your job: deduplicate, rank by relevance, and select the
best 10 tags for the publication.

Input format:
- gazetteer: list of tag names (high precision, from curated dictionary)
- spacy: list of entity names (from NER)
- llm: list of {tag, category} objects (from semantic analysis)

Return ONLY a JSON array of exactly 10 objects:
[{"tag": "...", "category": "...", "rationale": "one sentence why this tag matters"}, ...]

Prioritise tags that appear in multiple sources. No markdown, no preamble."""


def aggregator_node(state: TagState) -> TagState:
    """Merge all three extraction outputs into a ranked top-10 list."""
    client = _get_client()

    input_data = {
        "gazetteer": state.get("gazetteer_candidates", []),
        "spacy": state.get("spacy_candidates", []),
        "llm": state.get("llm_candidates", []),
    }

    feedback = state.get("reviewer_feedback")
    system_prompt = _AGGREGATOR_SYSTEM
    if feedback:
        system_prompt += f"\n\nREVIEWER FEEDBACK (apply this): {feedback}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(input_data)},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        final_tags = json.loads(raw)
        if not isinstance(final_tags, list):
            final_tags = []
    except json.JSONDecodeError:
        final_tags = []

    return {"final_tags": final_tags[:10]}
