"""
LangGraph graph for the TLDR agent.

Graph topology (sequential pipeline):
    START → key_points_node → tldr_drafter_node → tldr_refiner_node → END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END, START

from agents.tldr.nodes import key_points_node, tldr_drafter_node, tldr_refiner_node
from agents.tldr.state import TLDRState


def build_tldr_graph() -> StateGraph:
    builder = StateGraph(TLDRState)

    builder.add_node("key_points", key_points_node)
    builder.add_node("tldr_drafter", tldr_drafter_node)
    builder.add_node("tldr_refiner", tldr_refiner_node)

    builder.add_edge(START, "key_points")
    builder.add_edge("key_points", "tldr_drafter")
    builder.add_edge("tldr_drafter", "tldr_refiner")
    builder.add_edge("tldr_refiner", END)

    return builder.compile()
