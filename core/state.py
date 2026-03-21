"""
core/state.py
-------------
Top-level PipelineState shared across the entire orchestration graph.
Each agent writes only to its own key — no agent touches another agent's output.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

from core.base_agent import SharedContext, AgentResult


class PipelineState(TypedDict, total=False):
    # ---- Input -------------------------------------------------------
    raw_text: str                          # Original article text

    # ---- Manager output ---------------------------------------------
    shared_context: Optional[SharedContext]

    # ---- Agent outputs (keyed by agent name) ------------------------
    agent_results: dict[str, AgentResult]  # populated after parallel run

    # ---- Reviewer output --------------------------------------------
    review_verdicts: dict[str, str]        # agent_name → "approved" | feedback
    all_approved: bool
    revision_round: int

    # ---- Final output -----------------------------------------------
    final_output: dict[str, Any]
    error: Optional[str]
