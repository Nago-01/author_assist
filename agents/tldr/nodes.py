"""
Node functions for the TLDR agent LangGraph graph.

Nodes:
key_points_node   — extract 3-5 core take-aways from the article
tldr_drafter_node — write a coherent TLDR paragraph from the key points
tldr_refiner_node — polish, apply feedback, produce one-liner
"""

from __future__ import annotations

import json
import os

from groq import Groq

from agents.tldr.state import TLDRState

_CLIENT: Groq | None = None


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _CLIENT


# Key Points Extractor

_KEY_POINTS_SYSTEM = """You are a research analyst. Extract exactly 3-5 key takeaways from the
article. Each takeaway should be a complete, specific sentence — not a vague platitude.

Prioritise: main contribution, methodology, results/findings, implications.

Return ONLY a JSON array of 3-5 strings. No markdown, no numbering, no preamble."""


def key_points_node(state: TLDRState) -> TLDRState:
    client = _get_client()
    text = state.get("text", "")[:5000]
    themes = ", ".join(state.get("key_themes", []))
    domain = state.get("domain", "General")
    feedback = state.get("reviewer_feedback")

    system = _KEY_POINTS_SYSTEM
    if feedback:
        system += f"\n\nREVIEWER FEEDBACK — prioritise these aspects: {feedback}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Domain: {domain}\nKey themes: {themes}\n\nArticle:\n{text}",
            },
        ],
        temperature=0.2,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        points = json.loads(raw)
        if not isinstance(points, list):
            points = []
    except json.JSONDecodeError:
        points = []

    return {**state, "key_points": points[:5]}


# TLDR Drafter

_DRAFTER_SYSTEM = """You are a science communicator writing a TLDR for a publication.
Given key points and article metadata, write a clear, engaging TLDR paragraph.

Requirements:
- 3-5 sentences
- Accessible to the stated target audience
- Lead with the most important finding or contribution
- Avoid jargon unless the audience is specialist
- Do NOT start with "This paper" or "In this study"

Return ONLY the TLDR paragraph as plain text. No markdown, no preamble."""


def tldr_drafter_node(state: TLDRState) -> TLDRState:
    client = _get_client()
    points = state.get("key_points", [])
    audience = state.get("target_audience", "")
    message = state.get("main_message", "")
    domain = state.get("domain", "General")

    user_content = (
        f"Target audience: {audience}\n"
        f"Main message: {message}\n"
        f"Domain: {domain}\n\n"
        f"Key points:\n" + "\n".join(f"- {p}" for p in points)
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _DRAFTER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0.4,
        max_tokens=512,
    )

    draft = response.choices[0].message.content.strip()
    return {**state, "draft_tldr": draft}


# TLDR Refiner

_REFINER_SYSTEM = """You are a copyeditor finalising a TLDR for a publication.
Given a draft TLDR, refine it and also produce a one-liner elevator pitch (≤25 words).

Return ONLY a JSON object:
{
  "final_tldr": "The polished TLDR paragraph (3-5 sentences).",
  "one_liner": "≤25 word elevator pitch."
}
No markdown, no preamble."""


def tldr_refiner_node(state: TLDRState) -> TLDRState:
    client = _get_client()
    draft = state.get("draft_tldr", "")
    feedback = state.get("reviewer_feedback")
    audience = state.get("target_audience", "")

    system = _REFINER_SYSTEM
    if feedback:
        system += f"\n\nREVIEWER FEEDBACK — apply this during refinement: {feedback}"

    user_content = (
        f"Target audience: {audience}\n\n"
        f"Draft TLDR:\n{draft}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
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
        "final_tldr": parsed.get("final_tldr", draft),
        "one_liner": parsed.get("one_liner", ""),
    }
