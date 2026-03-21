"""
agents/references/agent.py
--------------------------
Thin wrapper exposing the References internal graph as a BaseAgent.
"""

from __future__ import annotations

from typing import Any

from core.base_agent import BaseAgent, SharedContext
from agents.references.graph import build_references_graph


class ReferencesAgent(BaseAgent):
    name = "references_generator"

    def __init__(self) -> None:
        self._graph = build_references_graph()

    def _execute(
        self,
        context: SharedContext,
        feedback: str | None,
    ) -> dict[str, Any]:
        initial_state = {
            "text": context.raw_text,
            "key_themes": context.key_themes,
            "domain": context.domain,
            "article_type": context.article_type,
            "reviewer_feedback": feedback,
        }
        result = self._graph.invoke(initial_state)

        return {
            "final_references": result.get("final_references", []),
            "citation_style": result.get("citation_style", "Unknown"),
            "total_references": len(result.get("final_references", [])),
            "raw_citations_found": len(result.get("raw_citations", [])),
        }
