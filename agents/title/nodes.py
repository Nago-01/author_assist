"""
agents/title/nodes.py — Nodes for the Title Generation agent.

Node inventory:
  1. title_start_node       — validates input
  2. generate_candidates_node — LLM generates 5 candidate titles
  3. rank_titles_node        — LLM selects the best + 2 alternates
  4. title_end_node          — wraps output into AgentResult
"""

import re
import json
import logging

logger = logging.getLogger(__name__)


# ─── 1. START ─────────────────────────────────────────────────────────────────

def title_start_node(state: dict) -> dict:
    article_text = state.get("article_text", "").strip()
    if not article_text:
        return {"error": "No article text provided.", "candidates": [], "primary_title": "", "alternates": [], "rationale": ""}

    revision = state.get("revision_note", "")
    if revision:
        logger.info(f"[TITLE:START] Revision note: {revision}")

    logger.info(f"[TITLE:START] {len(article_text.split())} words received.")
    return {"error": "", "candidates": [], "primary_title": "", "alternates": [], "rationale": ""}


# ─── 2. GENERATE CANDIDATES ───────────────────────────────────────────────────

_CANDIDATES_SYSTEM = """\
You are an expert science communicator and editor.
Generate exactly 5 strong candidate titles for an article, guided by the shared context.

Context:
  Key themes    : {key_themes}
  Target audience: {target_audience}
  Main message  : {main_message}
  Domain        : {domain}
  Language style: {language_style}

Title guidelines:
  - Specific, informative, and accurate to the article content
  - Appropriate for the indicated language style and audience
  - Vary the approach across candidates (e.g., question, declarative, method-focused, result-focused)
  - No clickbait; no vague filler words
  {revision_prefix}

Return ONLY a valid JSON array of exactly 5 title strings. No other text."""

_CANDIDATES_USER = """\
Article (first 3000 characters):
---
{article_snippet}
---

Return a JSON array of 5 candidate titles."""


def generate_candidates_node(state: dict) -> dict:
    from shared.llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    ctx = state.get("shared_context", {})
    revision = state.get("revision_note", "")
    text_snippet = state.get("article_text", "")[:3000]

    system_prompt = _CANDIDATES_SYSTEM.format(
        key_themes=", ".join(ctx.get("key_themes", [])) or "not specified",
        target_audience=ctx.get("target_audience", "general readers"),
        main_message=ctx.get("main_message", ""),
        domain=ctx.get("domain", ""),
        language_style=ctx.get("language_style", "academic"),
        revision_prefix=f"REVISION REQUEST: {revision}" if revision else "",
    )

    llm = get_llm(temperature=0.7)   # Higher temp for creative title diversity
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=_CANDIDATES_USER.format(article_snippet=text_snippet)),
        ])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        candidates = json.loads(raw)
        if not isinstance(candidates, list):
            candidates = []

        logger.info(f"[TITLE:CANDIDATES] {len(candidates)} candidates generated.")
        return {"candidates": candidates}

    except Exception as e:
        logger.error(f"[TITLE:CANDIDATES] Error: {e}")
        return {"candidates": [], "error": str(e)}


# ─── 3. RANK TITLES ───────────────────────────────────────────────────────────

_RANK_SYSTEM = """\
You are a senior editor evaluating candidate article titles.
Select the BEST primary title and 2 runner-up alternates from the list.

Shared context:
  Main message  : {main_message}
  Target audience: {target_audience}
  Language style: {language_style}

Criteria:
  - Clarity, accuracy, and informativeness
  - Appeal to the target audience
  - Specificity (avoids vague generalities)
  {revision_note}

Return ONLY a valid JSON object with these keys:
  "primary"   : the single best title string
  "alternates": array of exactly 2 runner-up title strings
  "rationale" : one sentence explaining why primary was chosen

No text outside the JSON object."""

_RANK_USER = """\
Candidate titles:
{candidates_json}

Choose the best. Return JSON only."""


def rank_titles_node(state: dict) -> dict:
    from shared.llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    candidates = state.get("candidates", [])
    ctx = state.get("shared_context", {})
    revision = state.get("revision_note", "")

    if not candidates:
        return {"primary_title": "", "alternates": [], "rationale": "No candidates available.", "error": "No candidates to rank."}

    system_prompt = _RANK_SYSTEM.format(
        main_message=ctx.get("main_message", ""),
        target_audience=ctx.get("target_audience", "general readers"),
        language_style=ctx.get("language_style", "academic"),
        revision_note=f"- REVISION: {revision}" if revision else "",
    )

    llm = get_llm(temperature=0.1)
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=_RANK_USER.format(candidates_json=json.dumps(candidates, indent=2))),
        ])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)

        primary = result.get("primary", "")
        alternates = result.get("alternates", [])[:2]
        rationale = result.get("rationale", "")

        logger.info(f"[TITLE:RANK] Primary: '{primary}'")
        return {"primary_title": primary, "alternates": alternates, "rationale": rationale}

    except Exception as e:
        logger.error(f"[TITLE:RANK] Error: {e}")
        first = candidates[0] if candidates else ""
        return {"primary_title": first, "alternates": candidates[1:3], "rationale": "Fallback (ranker error)", "error": str(e)}


# ─── 4. END ───────────────────────────────────────────────────────────────────

def title_end_node(state: dict) -> dict:
    primary = state.get("primary_title", "")
    error = state.get("error", "")

    status = "failed" if error and not primary else ("partial" if not primary else "success")
    logger.info(f"[TITLE:END] status={status} title='{primary}'")

    return {
        "agent_result": {
            "agent": "title",
            "output": {
                "primary": primary,
                "alternates": state.get("alternates", []),
                "rationale": state.get("rationale", ""),
                "all_candidates": state.get("candidates", []),
            },
            "status": status,
            "error": error,
        }
    }
