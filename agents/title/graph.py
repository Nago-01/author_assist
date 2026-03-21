"""
agents/title/graph.py — LangGraph graph for the Title Generation agent.

Standalone usage:
    from agents.title.graph import build_title_graph
    graph = build_title_graph()
    result = graph.invoke({
        "article_text": "...",
        "shared_context": { ... },
        "revision_note": "",
    })
    # result["agent_result"] contains the AgentResult dict
"""

from langgraph.graph import StateGraph, START, END
from agents.title.state import TitleAgentState
from agents.title.nodes import (
    title_start_node,
    generate_candidates_node,
    rank_titles_node,
    title_end_node,
)


def build_title_graph() -> StateGraph:
    """
    Topology:
      START → title_start → generate_candidates → rank_titles → title_end → END
    """
    builder = StateGraph(TitleAgentState)

    builder.add_node("title_start",          title_start_node)
    builder.add_node("generate_candidates",  generate_candidates_node)
    builder.add_node("rank_titles",          rank_titles_node)
    builder.add_node("title_end",            title_end_node)

    builder.add_edge(START,                 "title_start")
    builder.add_edge("title_start",         "generate_candidates")
    builder.add_edge("generate_candidates", "rank_titles")
    builder.add_edge("rank_titles",         "title_end")
    builder.add_edge("title_end",           END)

    return builder.compile()
