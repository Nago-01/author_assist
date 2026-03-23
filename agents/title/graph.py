"""
LangGraph graph for the Title agent.

Graph topology (sequential — selection depends on candidates):
    START → candidate_generator_node → title_selector_node → END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END, START

from agents.title.nodes import candidate_generator_node, title_selector_node
from agents.title.state import TitleState


def build_title_graph() -> StateGraph:
    builder = StateGraph(TitleState)

    builder.add_node("candidate_generator", candidate_generator_node)
    builder.add_node("title_selector", title_selector_node)

    builder.add_edge(START, "candidate_generator")
    builder.add_edge("candidate_generator", "title_selector")
    builder.add_edge("title_selector", END)

    return builder.compile()
