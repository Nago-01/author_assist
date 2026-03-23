"""Internal LangGraph state for the References agent."""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class ReferencesState(TypedDict, total=False):
    # Input
    text: str
    key_themes: list[str]
    domain: str
    article_type: str
    reviewer_feedback: Optional[str]

    # Intermediate
    raw_citations: list[str]         # strings that look like citations in the text
    structured_refs: list[dict]      # parsed {authors, year, title, venue, ...}

    # Final
    final_references: list[dict]     # cleaned, formatted reference list
    citation_style: str              # detected or inferred: "APA", "IEEE", "Vancouver", etc.
    error: Optional[str]
