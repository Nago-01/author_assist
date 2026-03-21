"""
agents/tags/graph.py — LangGraph graph for the Tags extraction agent.

Standalone usage:
    from agents.tags.graph import build_tags_graph
    graph = build_tags_graph()
    result = graph.invoke({
        "article_text": "...",
        "shared_context": { ... },   # SharedContext dict
        "revision_note": "",         # empty on first run
        "top_n": 10,
    })
    # result["agent_result"] contains the AgentResult dict
"""

from langgraph.graph import StateGraph, START, END
from agents.tags.state import TagAgentState
from agents.tags.nodes import (
    tags_start_node,
    gazetteer_node,
    spacy_node,
    llm_extractor_node,
    aggregator_node,
    tags_end_node,
)


def build_tags_graph() -> StateGraph:
    """
    Constructs and compiles the Tags extraction workflow.

    Topology:
      START → tags_start_node
            → [gazetteer_node | spacy_node | llm_extractor_node]  (parallel)
            → aggregator_node
            → tags_end_node → END
    """
    builder = StateGraph(TagAgentState)

    builder.add_node("tags_start",      tags_start_node)
    builder.add_node("gazetteer",       gazetteer_node)
    builder.add_node("spacy",           spacy_node)
    builder.add_node("llm_extractor",   llm_extractor_node)
    builder.add_node("aggregator",      aggregator_node)
    builder.add_node("tags_end",        tags_end_node)

    builder.add_edge(START, "tags_start")

    # Fan-out to three parallel extractors
    builder.add_edge("tags_start",    "gazetteer")
    builder.add_edge("tags_start",    "spacy")
    builder.add_edge("tags_start",    "llm_extractor")

    # Fan-in to aggregator
    builder.add_edge("gazetteer",     "aggregator")
    builder.add_edge("spacy",         "aggregator")
    builder.add_edge("llm_extractor", "aggregator")

    builder.add_edge("aggregator", "tags_end")
    builder.add_edge("tags_end",   END)

    return builder.compile()
