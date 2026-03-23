"""
Internal LangGraph state for the Tags agent graph.
Completely isolated — the outer pipeline only sees AgentResult.
"""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class TagState(TypedDict, total=False):
    # Input
    text: str
    key_themes: list[str]          
    domain: str                    
    reviewer_feedback: Optional[str]

    # Intermediate extraction results
    gazetteer_candidates: list[str]
    spacy_candidates: list[str]
    llm_candidates: list[dict]     

    # Final output
    final_tags: list[dict]         
    error: Optional[str]
