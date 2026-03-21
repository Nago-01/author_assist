"""
agents/tldr/state.py — State schema for the TLDR Generation agent.
"""

from typing import TypedDict


class TLDRAgentState(TypedDict):
    """
    Internal state for the TLDR generation LangGraph workflow.

    Inputs:
      - article_text   : full article text
      - shared_context : SharedContext dict from the Manager
      - revision_note  : Reviewer feedback — empty on first run

    Outputs:
      - draft_tldr  : initial full draft before refinement
      - tldr        : final refined TLDR (2–3 sentences, ≤ 80 words)
      - word_count  : word count of the final tldr
    """
    article_text: str
    shared_context: dict
    revision_note: str

    draft_tldr: str
    tldr: str
    word_count: int
    error: str
