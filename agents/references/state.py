"""
agents/references/state.py — State schema for the References extraction agent.
"""

from typing import TypedDict


class ReferencesAgentState(TypedDict):
    """
    Internal state for the References extraction LangGraph workflow.

    Inputs:
      - article_text   : full article text
      - shared_context : SharedContext dict from the Manager
      - revision_note  : Reviewer feedback — empty on first run

    Outputs:
      - raw_references   : citation strings pulled from the text by regex/NLP
      - references       : normalised, formatted reference list (APA-like)
      - count            : number of references found
    """
    article_text: str
    shared_context: dict
    revision_note: str

    raw_references: list[str]
    references: list[str]
    count: int
    error: str
