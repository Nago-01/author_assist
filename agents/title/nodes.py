"""
agents/title/nodes.py
---------------------
Node functions for the Title agent LangGraph graph.

Nodes
-----
candidate_generator_node  — generate 5 candidate titles via LLM
title_selector_node       — pick the best one, explain why
"""

from __future__ import annotations

import json
import os

from groq import Groq

from agents.title.state import TitleState

_CLIENT: Groq | None = None


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _CLIENT


# ── Node 1: Candidate Generator ──────────────────────────────────────────────

_CANDIDATE_SYSTEM = """You are an academic publication title specialist.
Given an article's text and metadata, generate exactly 5 candidate titles.

Each title should be:
- Precise and informative (not vague or clickbait)
- Appropriate for the target audience
- Reflect the article's main contribution or finding
- Vary in style: one formal, one question-based, one colon-separated, one brief, one descriptive

Return ONLY a JSON array of 5 title strings. No markdown, no preamble, no numbering."""


def candidate_generator_node(state: TitleState) -> TitleState:
    client = _get_client()
    text = state.get("text", "")[:4000]
    themes = ", ".join(state.get("key_themes", []))
    audience = state.get("target_audience", "")
    message = state.get("main_message", "")
    domain = state.get("domain", "General")
    article_type = state.get("article_type", "research")
    feedback = state.get("reviewer_feedback")

    system = _CANDIDATE_SYSTEM
    if feedback:
        system += f"\n\nREVIEWER FEEDBACK — apply this when generating titles: {feedback}"

    user_content = (
        f"Domain: {domain} | Type: {article_type}\n"
        f"Key themes: {themes}\n"
        f"Target audience: {audience}\n"
        f"Main message: {message}\n\n"
        f"Article excerpt:\n{text}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=512,
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

    return {**state, "candidate_titles": candidates[:5]}


# ── Node 2: Title Selector ────────────────────────────────────────────────────

_SELECTOR_SYSTEM = """You are a senior journal editor selecting the best title from candidates.
Given the article context and 5 candidate titles, select the single best title.

Consider:
1. Clarity and precision
2. Relevance to the main contribution
3. Appeal to the target audience
4. Academic appropriateness

Return ONLY a JSON object:
{
  "final_title": "The selected best title",
  "rationale": "One sentence explaining why this is the best choice.",
  "alternatives": ["second best", "third best"]
}
No markdown, no preamble."""


def title_selector_node(state: TitleState) -> TitleState:
    client = _get_client()
    candidates = state.get("candidate_titles", [])
    themes = ", ".join(state.get("key_themes", []))
    audience = state.get("target_audience", "")
    message = state.get("main_message", "")

    user_content = (
        f"Key themes: {themes}\n"
        f"Target audience: {audience}\n"
        f"Main message: {message}\n\n"
        f"Candidate titles:\n{json.dumps(candidates, indent=2)}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SELECTOR_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=512,
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
        "final_title": parsed.get("final_title", candidates[0] if candidates else ""),
        "title_rationale": parsed.get("rationale", ""),
        "alternative_titles": parsed.get("alternatives", []),
    }
