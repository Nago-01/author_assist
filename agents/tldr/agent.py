"""
agents/tldr/agent.py
--------------------
Thin wrapper exposing the TLDR internal graph as a BaseAgent.
"""

from __future__ import annotations

from typing import Any

from core.base_agent import BaseAgent, SharedContext
from agents.tldr.graph import build_tldr_graph


class TLDRAgent(BaseAgent):
    name = "tldr_generator"

    def __init__(self) -> None:
        self._graph = build_tldr_graph()

    def _execute(
        self,
        context: SharedContext,
        feedback: str | None,
    ) -> dict[str, Any]:
        initial_state = {
            "text": context.raw_text,
            "key_themes": context.key_themes,
            "target_audience": context.target_audience,
            "main_message": context.main_message,
            "domain": context.domain,
            "article_type": context.article_type,
            "reviewer_feedback": feedback,
        }
        result = self._graph.invoke(initial_state)

        return {
            "final_tldr": result.get("final_tldr", ""),
            "one_liner": result.get("one_liner", ""),
            "key_points": result.get("key_points", []),
        }
