"""
agents/tags/nodes.py — All nodes for the Tags extraction agent.

Node inventory:
  1. tags_start_node       — validates input, sets defaults
  2. gazetteer_node        — curated dictionary lookup
  3. spacy_node            — spaCy NER extraction
  4. llm_extractor_node    — LLM semantic extraction (context-aware)
  5. aggregator_node       — dedup + LLM ranking of final top-N tags
  6. tags_end_node         — logs summary, wraps into AgentResult
"""

import re
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ─── 1. START ─────────────────────────────────────────────────────────────────

def tags_start_node(state: dict) -> dict:
    """Validates article text and initialises all list fields."""
    article_text = state.get("article_text", "").strip()

    if not article_text:
        return {
            "error": "No article text provided.",
            "gazetteer_tags": [],
            "spacy_tags": [],
            "llm_tags": [],
            "all_candidate_tags": [],
            "final_tags": [],
            "top_n": 10,
        }

    logger.info(f"[TAGS:START] Article received — {len(article_text.split())} words.")
    revision = state.get("revision_note", "")
    if revision:
        logger.info(f"[TAGS:START] Revision note: {revision}")

    return {
        "article_text": article_text,
        "top_n": state.get("top_n", 10),
        "gazetteer_tags": [],
        "spacy_tags": [],
        "llm_tags": [],
        "all_candidate_tags": [],
        "final_tags": [],
        "error": "",
    }


# ─── 2. GAZETTEER ─────────────────────────────────────────────────────────────

def gazetteer_node(state: dict) -> dict:
    """Scans the article against the curated GAZETTEER_INDEX."""
    import sys
    from pathlib import Path
    # Allow import of gazetteer from this package
    sys.path.insert(0, str(Path(__file__).parent))
    from gazetteer import GAZETTEER_INDEX

    text = state.get("article_text", "")
    text_lower = text.lower()
    matched: list[dict] = []
    seen_terms: set[str] = set()

    for lower_term, canonical_term, category in GAZETTEER_INDEX:
        if canonical_term in seen_terms:
            continue
        pattern = r'(?<!\w)' + re.escape(lower_term) + r'(?!\w)'
        if re.search(pattern, text_lower):
            matched.append({
                "term": canonical_term,
                "category": category,
                "source": "gazetteer",
                "confidence": 1.0,
            })
            seen_terms.add(canonical_term)

    logger.info(f"[TAGS:GAZETTEER] {len(matched)} matches.")
    return {"gazetteer_tags": matched}


# ─── 3. SPACY ─────────────────────────────────────────────────────────────────

SPACY_LABEL_MAP = {
    "ORG": "Organization", "PERSON": "Person", "GPE": "Geopolitical Entity",
    "LOC": "Location", "PRODUCT": "Product", "EVENT": "Event",
    "WORK_OF_ART": "Work of Art", "LAW": "Law/Policy", "LANGUAGE": "Language",
    "NORP": "Nationality/Group", "FAC": "Facility",
}
USEFUL_LABELS = {"ORG", "PERSON", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "NORP"}


def spacy_node(state: dict) -> dict:
    """Runs spaCy NER over the article."""
    import spacy

    text = state.get("article_text", "")
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        logger.warning("[TAGS:SPACY] Model not found. Run: python -m spacy download en_core_web_md")
        return {"spacy_tags": []}

    doc = nlp(text)
    seen: set[str] = set()
    tags: list[dict] = []

    for ent in doc.ents:
        if ent.label_ not in USEFUL_LABELS:
            continue
        term = ent.text.strip()
        if not term or term in seen or len(term) < 2 or term.isdigit():
            continue
        seen.add(term)
        tags.append({
            "term": term,
            "category": SPACY_LABEL_MAP.get(ent.label_, ent.label_),
            "source": "spacy",
            "confidence": 0.85,
        })

    logger.info(f"[TAGS:SPACY] {len(tags)} entities extracted.")
    return {"spacy_tags": tags}


# ─── 4. LLM EXTRACTOR ────────────────────────────────────────────────────────

_LLM_EXTRACTION_SYSTEM = """\
You are an expert scientific editor specialising in AI and healthcare publications.
Extract meaningful tags from the article text, informed by the shared context provided.

Tags should capture: key AI/ML concepts, methods, architectures, named models, datasets,
benchmarks, organisations, labs, researchers, application domains, conferences, journals.

Shared context:
  Key themes    : {key_themes}
  Target audience: {target_audience}
  Main message  : {main_message}
  Domain        : {domain}

Return ONLY a valid JSON array. Each element must be an object with:
  "term"      : the tag text (canonical, concise)
  "category"  : one of [AI/ML Concept, AI Model, Dataset, Benchmark, Organization,
                Person, Application Domain, Conference, Journal, Method, Other]
  "confidence": a float 0.0–1.0

Do not include any text outside the JSON array. Do not truncate."""

_LLM_EXTRACTION_USER = """\
{revision_prefix}Extract all meaningful publication tags from the following article:

---
{article_text}
---

Return a JSON array only."""


def llm_extractor_node(state: dict) -> dict:
    """LLM semantic extraction, enriched with SharedContext."""
    from shared.llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    text = state.get("article_text", "")
    ctx = state.get("shared_context", {})
    revision = state.get("revision_note", "")

    max_chars = 12_000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Article truncated for extraction]"

    system_prompt = _LLM_EXTRACTION_SYSTEM.format(
        key_themes=", ".join(ctx.get("key_themes", [])) or "not specified",
        target_audience=ctx.get("target_audience", "general"),
        main_message=ctx.get("main_message", ""),
        domain=ctx.get("domain", ""),
    )
    revision_prefix = f"REVISION REQUEST: {revision}\n\n" if revision else ""
    user_msg = _LLM_EXTRACTION_USER.format(
        revision_prefix=revision_prefix,
        article_text=text,
    )

    llm = get_llm(temperature=0.1)
    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        extracted = json.loads(raw)

        tags = []
        for item in extracted:
            if not isinstance(item, dict) or "term" not in item:
                continue
            tags.append({
                "term": item.get("term", "").strip(),
                "category": item.get("category", "Other"),
                "source": "llm",
                "confidence": float(item.get("confidence", 0.8)),
            })

        logger.info(f"[TAGS:LLM] {len(tags)} tags extracted.")
        return {"llm_tags": tags}

    except Exception as e:
        logger.error(f"[TAGS:LLM] Error: {e}")
        return {"llm_tags": []}


# ─── 5. AGGREGATOR ────────────────────────────────────────────────────────────

_AGGREGATION_SYSTEM = """\
You are a senior editorial AI for a publications platform.
Select the BEST {top_n} final tags from the candidate list below.

Shared context:
  Key themes    : {key_themes}
  Main message  : {main_message}
  Target audience: {target_audience}

Selection criteria:
  - Prefer specific, meaningful, publication-relevant tags
  - Deduplicate synonyms; keep the more canonical/precise form
  - Favour tags appearing across multiple extraction sources
  - Balance technical concepts AND named entities (orgs, people, events)
  {revision_note}

Return ONLY a valid JSON array of exactly {top_n} objects. Each must have:
  "tag"       : final tag text
  "category"  : tag category
  "rationale" : one sentence explaining selection

No text outside the JSON array."""

_AGGREGATION_USER = """\
Article snippet (first 2000 chars):
---
{article_snippet}
---

Candidate tags:
{candidates_json}

Select the best {top_n} tags. Return a JSON array only."""


def aggregator_node(state: dict) -> dict:
    """Deduplicates candidates and selects top-N with LLM ranking."""
    from shared.llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    gz   = state.get("gazetteer_tags", [])
    sp   = state.get("spacy_tags", [])
    llm_ = state.get("llm_tags", [])
    top_n = state.get("top_n", 10)
    ctx  = state.get("shared_context", {})
    revision = state.get("revision_note", "")

    all_candidates = gz + sp + llm_
    logger.info(f"[TAGS:AGGREGATOR] Candidates: g={len(gz)} s={len(sp)} l={len(llm_)} total={len(all_candidates)}")

    if not all_candidates:
        return {"all_candidate_tags": [], "final_tags": []}

    # Deduplicate by term (case-insensitive)
    seen: set[str] = set()
    deduped: list[dict] = []
    for tag in all_candidates:
        key = tag["term"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(tag)

    revision_note_line = f"- REVISION: {revision}" if revision else ""
    system_prompt = _AGGREGATION_SYSTEM.format(
        top_n=top_n,
        key_themes=", ".join(ctx.get("key_themes", [])) or "not specified",
        main_message=ctx.get("main_message", ""),
        target_audience=ctx.get("target_audience", "general"),
        revision_note=revision_note_line,
    )

    llm = get_llm(temperature=0.2)
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=_AGGREGATION_USER.format(
                article_snippet=state.get("article_text", "")[:2000],
                candidates_json=json.dumps(deduped, indent=2),
                top_n=top_n,
            )),
        ])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        final = json.loads(raw)

        final_tags = [
            {
                "tag": item.get("tag", "").strip(),
                "category": item.get("category", "Other"),
                "rationale": item.get("rationale", ""),
            }
            for item in final
            if isinstance(item, dict) and "tag" in item
        ]

        logger.info(f"[TAGS:AGGREGATOR] {len(final_tags)} final tags selected.")
        return {"all_candidate_tags": deduped, "final_tags": final_tags}

    except Exception as e:
        logger.error(f"[TAGS:AGGREGATOR] Error: {e}")
        sorted_c = sorted(deduped, key=lambda x: x.get("confidence", 0), reverse=True)
        fallback = [
            {"tag": t["term"], "category": t["category"], "rationale": "Fallback (aggregator error)"}
            for t in sorted_c[:top_n]
        ]
        return {"all_candidate_tags": deduped, "final_tags": fallback}


# ─── 6. END ───────────────────────────────────────────────────────────────────

def tags_end_node(state: dict) -> dict:
    """Wraps final tags into a standardised AgentResult."""
    final_tags = state.get("final_tags", [])
    error = state.get("error", "")

    if error:
        status = "failed"
        logger.error(f"[TAGS:END] Completed with error: {error}")
    elif not final_tags:
        status = "partial"
        logger.warning("[TAGS:END] No final tags produced.")
    else:
        status = "success"
        logger.info(f"[TAGS:END] {len(final_tags)} tags produced.")

    return {
        "agent_result": {
            "agent": "tags",
            "output": {
                "final_tags": final_tags,
                "candidate_counts": {
                    "gazetteer": len(state.get("gazetteer_tags", [])),
                    "spacy": len(state.get("spacy_tags", [])),
                    "llm": len(state.get("llm_tags", [])),
                    "total_deduped": len(state.get("all_candidate_tags", [])),
                },
            },
            "status": status,
            "error": error,
        }
    }
