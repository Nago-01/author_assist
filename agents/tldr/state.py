"""Internal LangGraph state for the TLDR agent."""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class TLDRState(TypedDict, total=False):
    # Input
    text: str
    key_themes: list[str]
    target_audience: str
    main_message: str
    domain: str
    article_type: str
    reviewer_feedback: Optional[str]

    # Intermediate
    draft_tldr: str                  # first-pass summary
    key_points: list[str]            # bullet points extracted before drafting

    # Final
    final_tldr: str                  # polished single paragraph
    one_liner: str                   # ≤ 25 word elevator pitch
    error: Optional[str]
