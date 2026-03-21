"""
agents/tldr/nodes.py — Nodes for the TLDR Generation agent.

Node inventory:
  1. tldr_start_node   — validates input
  2. draft_tldr_node   — LLM writes an initial full-length TLDR draft
  3. refine_tldr_node  — LLM tightens it to ≤ 80 words, audience-aligned
  4. tldr_end_node     — wraps output into AgentResult
"""

import re
import json
import logging

logger = logging.getLogger(__name__)


# ─── 1. START ─────────────────────────────────────────────────────────────────

def tldr_start_node(state: dict) -> dict:
    article_text = state.get("article_text", "").strip()
    if not article_text:
        return {"error": "No article text provided.", "draft_tldr": "", "tldr": "", "word_count": 0}

    revision = state.get("revision_note", "")
    if revision:
        logger.info(f"[TLDR:START] Revision note: {revision}")

    logger.info(f"[TLDR:START] {len(article_text.split())} words received.")
    return {"error": "", "draft_tldr": "", "tldr": "", "word_count": 0}


# ─── 2. DRAFT ─────────────────────────────────────────────────────────────────

_DRAFT_SYSTEM = """\
You are a scientific communicator writing a TLDR for an article.
Produce a comprehensive 3–5 sentence draft that captures:
  1. The core problem or question
  2. The method or approach used
  3. The key result or contribution

Shared context:
  Key themes    : {key_themes}
  Main message  : {main_message}
  Target audience: {target_audience}
  Domain        : {domain}

Write in a style appropriate for the audience. Do not use the phrase "TLDR:".
{revision_prefix}
Return ONLY the TLDR draft as plain text."""

_DRAFT_USER = """\
Article (first 4000 characters):
---
{article_snippet}
---

Write the TLDR draft."""


def draft_tldr_node(state: dict) -> dict:
    from shared.llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    ctx = state.get("shared_context", {})
    revision = state.get("revision_note", "")
    article_snippet = state.get("article_text", "")[:4000]

    system_prompt = _DRAFT_SYSTEM.format(
        key_themes=", ".join(ctx.get("key_themes", [])) or "not specified",
        main_message=ctx.get("main_message", ""),
        target_audience=ctx.get("target_audience", "general readers"),
        domain=ctx.get("domain", ""),
        revision_prefix=f"REVISION REQUEST: {revision}" if revision else "",
    )

    llm = get_llm(temperature=0.3)
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=_DRAFT_USER.format(article_snippet=article_snippet)),
        ])
        draft = response.content.strip()
        logger.info(f"[TLDR:DRAFT] Draft produced ({len(draft.split())} words).")
        return {"draft_tldr": draft}

    except Exception as e:
        logger.error(f"[TLDR:DRAFT] Error: {e}")
        return {"draft_tldr": "", "error": str(e)}


# ─── 3. REFINE ────────────────────────────────────────────────────────────────

_REFINE_SYSTEM = """\
You are a precision editor. Your job is to tighten the TLDR draft below into a
final version of exactly 2–3 sentences, with a MAXIMUM of 80 words.

Rules:
  - Keep the single most important insight from each of: problem, method, result
  - Match the target audience's level: {target_audience}
  - Remove all filler phrases ("In this paper", "We present", "Our work shows")
  - Do not start with "TLDR" or similar labels
  - Use active voice where possible
  {revision_note}

Return ONLY the polished TLDR as plain text. No labels, no JSON."""

_REFINE_USER = """\
Draft TLDR:
---
{draft_tldr}
---

Refined version (≤ 80 words):"""


def refine_tldr_node(state: dict) -> dict:
    from shared.llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    draft = state.get("draft_tldr", "")
    ctx = state.get("shared_context", {})
    revision = state.get("revision_note", "")

    if not draft:
        return {"tldr": "", "word_count": 0, "error": "No draft to refine."}

    system_prompt = _REFINE_SYSTEM.format(
        target_audience=ctx.get("target_audience", "general readers"),
        revision_note=f"- REVISION: {revision}" if revision else "",
    )

    llm = get_llm(temperature=0.1)
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=_REFINE_USER.format(draft_tldr=draft)),
        ])
        tldr = response.content.strip()
        word_count = len(tldr.split())
        logger.info(f"[TLDR:REFINE] Final TLDR: {word_count} words.")
        return {"tldr": tldr, "word_count": word_count}

    except Exception as e:
        logger.error(f"[TLDR:REFINE] Error: {e}")
        # Fallback: use draft, truncate roughly
        fallback = " ".join(draft.split()[:80])
        return {"tldr": fallback, "word_count": len(fallback.split()), "error": str(e)}


# ─── 4. END ───────────────────────────────────────────────────────────────────

def tldr_end_node(state: dict) -> dict:
    tldr = state.get("tldr", "")
    error = state.get("error", "")
    status = "failed" if error and not tldr else ("partial" if not tldr else "success")
    logger.info(f"[TLDR:END] status={status} words={state.get('word_count', 0)}")

    return {
        "agent_result": {
            "agent": "tldr",
            "output": {
                "tldr": tldr,
                "word_count": state.get("word_count", 0),
                "draft": state.get("draft_tldr", ""),
            },
            "status": status,
            "error": error,
        }
    }
