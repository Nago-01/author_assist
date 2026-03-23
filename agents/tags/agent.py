"""
Thin wrapper that makes the Tags internal LangGraph graph
conform to the BaseAgent interface.

This is the ONLY file that knows about both the internal graph and the outer
pipeline contract. graph.py, nodes.py, state.py are completely unaware of
the multi-agent system and can be used standalone.
"""

from __future__ import annotations

from typing import Any

from core.base_agent import BaseAgent, SharedContext
from agents.tags.graph import build_tag_graph


class TagsAgent(BaseAgent):
    name = "tags_generator"

    def __init__(self) -> None:
        self._graph = build_tag_graph()

    def _execute(
        self,
        context: SharedContext,
        feedback: str | None,
    ) -> dict[str, Any]:
        initial_state = {
            "text": context.raw_text,
            "key_themes": context.key_themes,
            "domain": context.domain,
            "reviewer_feedback": feedback,
        }
        result = self._graph.invoke(initial_state)

        candidate_counts = {
            "gazetteer": len(result.get("gazetteer_candidates", [])),
            "spacy": len(result.get("spacy_candidates", [])),
            "llm": len(result.get("llm_candidates", [])),
        }
        candidate_counts["total_deduped"] = sum(
            1 for _ in {
                *result.get("gazetteer_candidates", []),
                *result.get("spacy_candidates", []),
                *[t.get("tag", "") for t in result.get("llm_candidates", [])],
            }
        )

        return {
            "final_tags": result.get("final_tags", []),
            "candidate_counts": candidate_counts,
        }
