"""
agents/title/agent.py
---------------------
Thin wrapper exposing the Title internal graph as a BaseAgent.
The internal graph.py / nodes.py / state.py are fully standalone.
"""

from __future__ import annotations

from typing import Any

from core.base_agent import BaseAgent, SharedContext
from agents.title.graph import build_title_graph


class TitleAgent(BaseAgent):
    name = "title_generator"

    def __init__(self) -> None:
        self._graph = build_title_graph()

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
            "final_title": result.get("final_title", ""),
            "rationale": result.get("title_rationale", ""),
            "alternative_titles": result.get("alternative_titles", []),
            "all_candidates": result.get("candidate_titles", []),
        }
