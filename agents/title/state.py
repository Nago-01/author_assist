"""
agents/title/state.py — State schema for the Title Generation agent.
"""

from typing import TypedDict


class TitleAgentState(TypedDict):
    """
    Internal state for the Title generation LangGraph workflow.

    Inputs:
      - article_text   : full article text (truncated internally if needed)
      - shared_context : SharedContext dict from the Manager
      - revision_note  : Reviewer feedback — empty on first run

    Outputs:
      - candidates     : list of 5 candidate titles generated
      - primary_title  : the best single title selected
      - alternates     : 2 runner-up titles
      - rationale      : why the primary was chosen
    """
    article_text: str
    shared_context: dict
    revision_note: str

    candidates: list[str]
    primary_title: str
    alternates: list[str]
    rationale: str
    error: str
