"""
LangGraph graph for the References agent.

Graph topology (sequential pipeline):
    START → citation_extractor_node → reference_parser_node → reference_formatter_node → END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END, START

from agents.references.nodes import (
    citation_extractor_node,
    reference_parser_node,
    reference_formatter_node,
)
from agents.references.state import ReferencesState


def build_references_graph() -> StateGraph:
    builder = StateGraph(ReferencesState)

    builder.add_node("citation_extractor", citation_extractor_node)
    builder.add_node("reference_parser", reference_parser_node)
    builder.add_node("reference_formatter", reference_formatter_node)

    builder.add_edge(START, "citation_extractor")
    builder.add_edge("citation_extractor", "reference_parser")
    builder.add_edge("reference_parser", "reference_formatter")
    builder.add_edge("reference_formatter", END)

    return builder.compile()
