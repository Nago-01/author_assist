"""
agents/references/graph.py — LangGraph graph for the References extraction agent.

Standalone usage:
    from agents.references.graph import build_references_graph
    graph = build_references_graph()
    result = graph.invoke({
        "article_text": "...",
        "shared_context": { ... },
        "revision_note": "",
    })
    # result["agent_result"] contains the AgentResult dict
"""

from langgraph.graph import StateGraph, START, END
from agents.references.state import ReferencesAgentState
from agents.references.nodes import (
    references_start_node,
    extract_raw_references_node,
    format_references_node,
    references_end_node,
)


def build_references_graph() -> StateGraph:
    """
    Topology:
      START → refs_start → extract_raw → format_refs → refs_end → END
    """
    builder = StateGraph(ReferencesAgentState)

    builder.add_node("refs_start",    references_start_node)
    builder.add_node("extract_raw",   extract_raw_references_node)
    builder.add_node("format_refs",   format_references_node)
    builder.add_node("refs_end",      references_end_node)

    builder.add_edge(START,           "refs_start")
    builder.add_edge("refs_start",    "extract_raw")
    builder.add_edge("extract_raw",   "format_refs")
    builder.add_edge("format_refs",   "refs_end")
    builder.add_edge("refs_end",      END)

    return builder.compile()
