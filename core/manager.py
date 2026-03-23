"""
Manager node — the ONLY node that reads raw input text.

Responsibilities:
1. Call the LLM once to extract key themes, audience, domain, main message.
2. Produce a SharedContext that is broadcast to all agents.
3. Never run any agent logic itself.
"""

from __future__ import annotations

import json
import os

from groq import Groq

from core.base_agent import SharedContext
from core.state import PipelineState

_CLIENT: Groq | None = None


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _CLIENT


_SYSTEM_PROMPT = """You are a document analyst. Given an article, extract:
- key_themes: list of 3-6 core topics/themes
- target_audience: brief phrase (e.g. "ML researchers", "clinical practitioners")
- main_message: one sentence summary of the article's central argument or contribution
- domain: primary domain ("AI/ML", "Healthcare", "Finance", "General", etc.)
- article_type: one of "research", "tutorial", "opinion", "survey", "news"

Return ONLY a valid JSON object with exactly these five keys. No markdown, no preamble."""


def manager_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: reads raw_text → writes shared_context.
    """
    raw_text = state.get("raw_text", "")
    if not raw_text.strip():
        return {**state, "error": "Manager received empty text."}

    client = _get_client()

    # Truncate to ~6000 chars to stay within context limits while keeping cost low
    truncated = raw_text[:6000]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Article:\n\n{truncated}"},
        ],
        temperature=0.1,
        max_tokens=512,
    )

    raw_json = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    raw_json = raw_json.strip()

    parsed = json.loads(raw_json)

    context = SharedContext(
        raw_text=raw_text,
        key_themes=parsed.get("key_themes", []),
        target_audience=parsed.get("target_audience", "General audience"),
        main_message=parsed.get("main_message", ""),
        domain=parsed.get("domain", "General"),
        article_type=parsed.get("article_type", "research"),
    )

    return {**state, "shared_context": context}
