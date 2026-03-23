"""
Manager node for the Author Assist orchestrator.

The Manager is the first node in the top-level graph. It reads the article text
and produces a SharedContext — a rich structured understanding of:
  • Key themes
  • Target audience
  • Main message / core contribution
  • Domain
  • Language style

All four parallel worker agents receive this SharedContext before they begin.
"""

import re
import json
import logging

logger = logging.getLogger(__name__)

_MANAGER_SYSTEM = """\
You are an expert editorial analyst. Your task is to read an article and produce
a structured understanding of it that will guide a team of specialist agents
(title generator, TLDR writer, tag extractor, reference formatter).

Analyse the article and return ONLY a valid JSON object with these exact keys:

{
  "key_themes"     : ["theme1", "theme2", ...],   // 3–6 specific themes from the article
  "target_audience": "...",                        // one phrase: e.g. "ML researchers", "clinicians", "general tech readers"
  "main_message"   : "...",                        // single sentence: the core claim or contribution
  "domain"         : "...",                        // one of: "AI/ML", "Healthcare", "Mixed", "Other"
  "language_style" : "..."                         // one of: "academic", "technical blog", "popular science", "clinical"
}

Be specific. key_themes should be concrete concepts from this article, NOT generic phrases.
Return ONLY the JSON object — no markdown, no explanation."""

_MANAGER_USER = """\
Article text (first 5000 characters):
---
{article_snippet}
---

Produce the SharedContext JSON object."""


def manager_node(state: dict) -> dict:
    """
    Analyses the article and produces SharedContext.
    Populates state["shared_context"].
    """
    from shared.llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    article_text = state.get("article_text", "").strip()

    if not article_text:
        logger.error("[MANAGER] No article text provided.")
        return {
            "error": "No article text provided.",
            "shared_context": {
                "article_text": "",
                "key_themes": [],
                "target_audience": "unknown",
                "main_message": "",
                "domain": "Other",
                "language_style": "academic",
            },
        }

    article_snippet = article_text[:5000]
    llm = get_llm(temperature=0.1)

    try:
        response = llm.invoke([
            SystemMessage(content=_MANAGER_SYSTEM),
            HumanMessage(content=_MANAGER_USER.format(article_snippet=article_snippet)),
        ])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)

        shared_context = {
            "article_text": article_text,       # pass full text through for agents
            "key_themes":     parsed.get("key_themes", []),
            "target_audience": parsed.get("target_audience", "general readers"),
            "main_message":   parsed.get("main_message", ""),
            "domain":         parsed.get("domain", "Other"),
            "language_style": parsed.get("language_style", "academic"),
        }

        logger.info(f"[MANAGER] SharedContext produced.")
        logger.info(f"  Domain        : {shared_context['domain']}")
        logger.info(f"  Audience      : {shared_context['target_audience']}")
        logger.info(f"  Main message  : {shared_context['main_message'][:80]}...")
        logger.info(f"  Key themes    : {', '.join(shared_context['key_themes'])}")

        return {"shared_context": shared_context, "error": ""}

    except Exception as e:
        logger.error(f"[MANAGER] Failed to produce SharedContext: {e}")
        # Graceful fallback — agents will still run with minimal context
        return {
            "shared_context": {
                "article_text": article_text,
                "key_themes": [],
                "target_audience": "general readers",
                "main_message": article_text[:200],
                "domain": "Other",
                "language_style": "academic",
            },
            "error": f"Manager warning: {e}",
        }
