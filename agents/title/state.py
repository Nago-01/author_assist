"""agents/title/state.py — internal LangGraph state for the Title agent."""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class TitleState(TypedDict, total=False):
    # Input
    text: str
    key_themes: list[str]
    target_audience: str
    main_message: str
    domain: str
    article_type: str
    reviewer_feedback: Optional[str]

    # Intermediate
    candidate_titles: list[str]          # multiple raw candidates

    # Final
    final_title: str
    title_rationale: str
    alternative_titles: list[str]        # runner-up options
    error: Optional[str]
