"""
agents/tags/graph.py
--------------------
Wires the Tags agent internal LangGraph graph.

Graph topology (parallel extraction):
    START
      ↓ (fan-out)
    gazetteer_node | spacy_node | llm_extractor_node
      ↓ (fan-in)
    aggregator_node
      ↓
    END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END, START

from agents.tags.nodes import (
    gazetteer_node,
    spacy_node,
    llm_extractor_node,
    aggregator_node,
)
from agents.tags.state import TagState


def build_tag_graph() -> StateGraph:
    """Build and compile the Tags extraction graph."""
    builder = StateGraph(TagState)

    # Register nodes
    builder.add_node("gazetteer", gazetteer_node)
    builder.add_node("spacy", spacy_node)
    builder.add_node("llm_extractor", llm_extractor_node)
    builder.add_node("aggregator", aggregator_node)

    # Fan-out from START to all three extractors in parallel
    builder.add_edge(START, "gazetteer")
    builder.add_edge(START, "spacy")
    builder.add_edge(START, "llm_extractor")

    # Fan-in: all three converge on aggregator
    builder.add_edge("gazetteer", "aggregator")
    builder.add_edge("spacy", "aggregator")
    builder.add_edge("llm_extractor", "aggregator")

    # Aggregator → END
    builder.add_edge("aggregator", END)

    return builder.compile()
