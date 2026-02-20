"""
Graph definition for the Tag Extractor.

"""

from langgraph.graph import StateGraph, START, END

from state import TagState
from nodes import (
    start_node,
    gazetteer_node,
    spacy_node,
    llm_extractor_node,
    aggregator_node,
    end_node,
)


def build_graph() -> StateGraph:
    """
    Constructs and compiles the Tag Extractor LangGraph workflow.

    The three extraction nodes run in parallel (map step).
    The aggregator node collects all their outputs (reduce step).
    """
    builder = StateGraph(TagState)

    # Register all nodes
    builder.add_node("start_node",         start_node)
    builder.add_node("gazetteer_node",     gazetteer_node)
    builder.add_node("spacy_node",         spacy_node)
    builder.add_node("llm_extractor_node", llm_extractor_node)
    builder.add_node("aggregator_node",    aggregator_node)
    builder.add_node("end_node",           end_node)

    # Entry point
    builder.add_edge(START, "start_node")

    # Fan-out: start_node - three parallel extractors
    builder.add_edge("start_node", "gazetteer_node")
    builder.add_edge("start_node", "spacy_node")
    builder.add_edge("start_node", "llm_extractor_node")

    # Fan-in: all three extractors - aggregator
    builder.add_edge("gazetteer_node",     "aggregator_node")
    builder.add_edge("spacy_node",         "aggregator_node")
    builder.add_edge("llm_extractor_node", "aggregator_node")

    # Aggregator - end
    builder.add_edge("aggregator_node", "end_node")
    builder.add_edge("end_node", END)

    return builder.compile()


# Singleton graph instance
tag_extractor_graph = build_graph()