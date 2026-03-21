"""
agents/references/nodes.py — Nodes for the References extraction agent.

Node inventory:
  1. references_start_node      — validates input
  2. extract_raw_references_node — regex + heuristics to pull citation strings
  3. format_references_node      — LLM normalises into APA-like style
  4. references_end_node         — wraps output into AgentResult
"""

import re
import json
import logging

logger = logging.getLogger(__name__)


# ─── 1. START ─────────────────────────────────────────────────────────────────

def references_start_node(state: dict) -> dict:
    article_text = state.get("article_text", "").strip()
    if not article_text:
        return {"error": "No article text provided.", "raw_references": [], "references": [], "count": 0}

    revision = state.get("revision_note", "")
    if revision:
        logger.info(f"[REFS:START] Revision note: {revision}")

    logger.info(f"[REFS:START] {len(article_text.split())} words received.")
    return {"error": "", "raw_references": [], "references": [], "count": 0}


# ─── 2. EXTRACT RAW REFERENCES ────────────────────────────────────────────────

# Patterns to locate the references / bibliography section
_SECTION_HEADERS = re.compile(
    r'\n\s*(References|Bibliography|Works Cited|Citations|Literature)\s*\n',
    re.IGNORECASE,
)

# A line that looks like a numbered or bracketed citation entry
_REF_LINE = re.compile(
    r'^\s*(\[\d+\]|\d+\.|\d+\))\s+.{10,}',
)

# Author-year inline citation: "Smith et al. (2023)" or "Smith & Jones, 2022"
_AUTHOR_YEAR = re.compile(
    r'[A-Z][a-z]+(?: et al\.?| & [A-Z][a-z]+)?,?\s*\(?(?:19|20)\d{2}\)?',
)


def extract_raw_references_node(state: dict) -> dict:
    """
    Extracts raw reference strings from the article text using two strategies:

    1. Section-based: looks for a References/Bibliography section heading and
       pulls numbered/bulleted lines beneath it.
    2. Inline fallback: collects "Author et al. (Year)" patterns from the full text.
    """
    text = state.get("article_text", "")
    raw_refs: list[str] = []

    # Strategy 1: find a references section
    match = _SECTION_HEADERS.search(text)
    if match:
        refs_section = text[match.end():]
        lines = refs_section.split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Stop if we hit another major section heading
            if re.match(r'^[A-Z][A-Za-z\s]{2,30}$', stripped) and len(stripped.split()) <= 4:
                break
            if _REF_LINE.match(line) or len(stripped) > 40:
                raw_refs.append(stripped)
        logger.info(f"[REFS:EXTRACT] Found references section — {len(raw_refs)} entries.")

    # Strategy 2: inline citation fallback (if section strategy found nothing)
    if not raw_refs:
        logger.info("[REFS:EXTRACT] No bibliography section found — using inline citation extraction.")
        seen: set[str] = set()
        for m in _AUTHOR_YEAR.finditer(text):
            # Expand to grab the surrounding sentence fragment
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 100)
            snippet = text[start:end].strip().replace("\n", " ")
            key = m.group(0)
            if key not in seen:
                seen.add(key)
                raw_refs.append(snippet)
        logger.info(f"[REFS:EXTRACT] Inline strategy — {len(raw_refs)} citations found.")

    return {"raw_references": raw_refs}


# ─── 3. FORMAT REFERENCES ────────────────────────────────────────────────────

_FORMAT_SYSTEM = """\
You are a scholarly editor responsible for normalising academic references.
Format each raw reference string into a clean APA 7th edition citation.

Rules:
  - Author(s). (Year). Title. Venue (Journal/Conference). DOI/URL if present.
  - If information is missing, include what is available and omit the rest.
  - Do not invent authors, titles, or years that are not present in the raw string.
  - Deduplicate: if two raw strings clearly refer to the same source, include only one.
  {revision_note}

Return ONLY a valid JSON array of formatted citation strings. No other text."""

_FORMAT_USER = """\
Raw reference strings extracted from the article:
{raw_refs_json}

Return a JSON array of formatted APA-style citations."""


def format_references_node(state: dict) -> dict:
    from shared.llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    raw_refs = state.get("raw_references", [])
    revision = state.get("revision_note", "")

    if not raw_refs:
        logger.warning("[REFS:FORMAT] No raw references to format.")
        return {"references": [], "count": 0}

    # Cap at 50 references to stay within LLM context limits
    refs_to_format = raw_refs[:50]
    system_prompt = _FORMAT_SYSTEM.format(
        revision_note=f"- REVISION: {revision}" if revision else "",
    )

    llm = get_llm(temperature=0.1)
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=_FORMAT_USER.format(
                raw_refs_json=json.dumps(refs_to_format, indent=2),
            )),
        ])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        formatted = json.loads(raw)

        if not isinstance(formatted, list):
            formatted = []

        logger.info(f"[REFS:FORMAT] {len(formatted)} references formatted.")
        return {"references": formatted, "count": len(formatted)}

    except Exception as e:
        logger.error(f"[REFS:FORMAT] Error: {e}")
        # Fallback: return raw strings as-is
        return {"references": refs_to_format, "count": len(refs_to_format), "error": str(e)}


# ─── 4. END ───────────────────────────────────────────────────────────────────

def references_end_node(state: dict) -> dict:
    refs = state.get("references", [])
    error = state.get("error", "")
    status = "failed" if error and not refs else ("partial" if not refs else "success")
    logger.info(f"[REFS:END] status={status} count={len(refs)}")

    return {
        "agent_result": {
            "agent": "references",
            "output": {
                "references": refs,
                "count": len(refs),
                "raw_count": len(state.get("raw_references", [])),
            },
            "status": status,
            "error": error,
        }
    }
