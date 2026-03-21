"""
agents/tags/state.py
--------------------
Internal LangGraph state for the Tags agent graph.
Completely isolated — the outer pipeline only sees AgentResult.
"""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class TagState(TypedDict, total=False):
    # Input
    text: str
    key_themes: list[str]          # from SharedContext
    domain: str                    # from SharedContext
    reviewer_feedback: Optional[str]

    # Intermediate extraction results
    gazetteer_candidates: list[str]
    spacy_candidates: list[str]
    llm_candidates: list[dict]     # [{tag, category}]

    # Final output
    final_tags: list[dict]         # [{tag, category, rationale}]
    error: Optional[str]
