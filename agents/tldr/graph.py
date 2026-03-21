"""
agents/tldr/graph.py — LangGraph graph for the TLDR Generation agent.

Standalone usage:
    from agents.tldr.graph import build_tldr_graph
    graph = build_tldr_graph()
    result = graph.invoke({
        "article_text": "...",
        "shared_context": { ... },
        "revision_note": "",
    })
    # result["agent_result"] contains the AgentResult dict
"""

from langgraph.graph import StateGraph, START, END
from agents.tldr.state import TLDRAgentState
from agents.tldr.nodes import (
    tldr_start_node,
    draft_tldr_node,
    refine_tldr_node,
    tldr_end_node,
)


def build_tldr_graph() -> StateGraph:
    """
    Topology:
      START → tldr_start → draft_tldr → refine_tldr → tldr_end → END
    """
    builder = StateGraph(TLDRAgentState)

    builder.add_node("tldr_start",   tldr_start_node)
    builder.add_node("draft_tldr",   draft_tldr_node)
    builder.add_node("refine_tldr",  refine_tldr_node)
    builder.add_node("tldr_end",     tldr_end_node)

    builder.add_edge(START,          "tldr_start")
    builder.add_edge("tldr_start",   "draft_tldr")
    builder.add_edge("draft_tldr",   "refine_tldr")
    builder.add_edge("refine_tldr",  "tldr_end")
    builder.add_edge("tldr_end",     END)

    return builder.compile()
