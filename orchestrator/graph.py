"""
orchestrator/graph.py — Top-level LangGraph orchestrator for Author Assist.

Architecture:
  START
    → manager_node              (analyses article, produces SharedContext)
    → [run_tags | run_title | run_tldr | run_references]   (parallel fan-out)
    → reviewer_node             (scores all 4 outputs, produces ReviewFeedback)
    → route_after_review()      (conditional edge)
         ├─ if agents need revision AND retry_count < max_retries:
         │      re-run only the failing agents → reviewer_node → loop
         └─ else:
                final_output_node → END

Selective re-execution uses LangGraph's Send API to dispatch only the
failing agents back into the graph, each carrying the Reviewer's
targeted feedback in their revision_note.
"""

import logging
from datetime import datetime
from typing import Any

from langgraph.graph import StateGraph, START, END

from shared.state import OrchestratorState
from orchestrator.manager import manager_node
from orchestrator.reviewer import reviewer_node

logger = logging.getLogger(__name__)


# ─── Agent runner nodes ───────────────────────────────────────────────────────
# Each runner is a thin wrapper that invokes the agent's compiled sub-graph and
# writes the returned AgentResult into the orchestrator state.

def run_tags_node(state: dict) -> dict:
    """Runs the Tags extraction agent."""
    from agents.tags.graph import build_tags_graph

    logger.info("[ORCH] Running Tags agent...")
    revision = _get_revision_note(state, "tags")
    graph = build_tags_graph()
    result = graph.invoke({
        "article_text":   state["shared_context"]["article_text"],
        "shared_context": state["shared_context"],
        "revision_note":  revision,
        "top_n":          10,
    })
    return {"tags_result": result.get("agent_result", _failed_result("tags", "No agent_result returned"))}


def run_title_node(state: dict) -> dict:
    """Runs the Title generation agent."""
    from agents.title.graph import build_title_graph

    logger.info("[ORCH] Running Title agent...")
    revision = _get_revision_note(state, "title")
    graph = build_title_graph()
    result = graph.invoke({
        "article_text":   state["shared_context"]["article_text"],
        "shared_context": state["shared_context"],
        "revision_note":  revision,
    })
    return {"title_result": result.get("agent_result", _failed_result("title", "No agent_result returned"))}


def run_tldr_node(state: dict) -> dict:
    """Runs the TLDR generation agent."""
    from agents.tldr.graph import build_tldr_graph

    logger.info("[ORCH] Running TLDR agent...")
    revision = _get_revision_note(state, "tldr")
    graph = build_tldr_graph()
    result = graph.invoke({
        "article_text":   state["shared_context"]["article_text"],
        "shared_context": state["shared_context"],
        "revision_note":  revision,
    })
    return {"tldr_result": result.get("agent_result", _failed_result("tldr", "No agent_result returned"))}


def run_references_node(state: dict) -> dict:
    """Runs the References extraction agent."""
    from agents.references.graph import build_references_graph

    logger.info("[ORCH] Running References agent...")
    revision = _get_revision_note(state, "references")
    graph = build_references_graph()
    result = graph.invoke({
        "article_text":   state["shared_context"]["article_text"],
        "shared_context": state["shared_context"],
        "revision_note":  revision,
    })
    return {"references_result": result.get("agent_result", _failed_result("references", "No agent_result returned"))}


def _get_revision_note(state: dict, agent_name: str) -> str:
    """Extracts revision feedback for a specific agent from the last review cycle."""
    feedback_list = state.get("review_feedback", [])
    for fb in feedback_list:
        if fb.get("agent") == agent_name and not fb.get("approved", True):
            return fb.get("feedback", "")
    return ""


def _failed_result(agent: str, error: str) -> dict:
    return {"agent": agent, "output": {}, "status": "failed", "error": error}


# ─── Routing ──────────────────────────────────────────────────────────────────

def route_after_review(state: dict) -> list[str]:
    """
    Conditional routing function called after every reviewer_node execution.

    Returns a list of node names to execute next:
      - If agents need revision AND we haven't exceeded max retries:
          → only the failing agent runners, which then feed back into reviewer_node
      - Otherwise:
          → final_output_node
    """
    needs_revision  = state.get("agents_needing_revision", [])
    retry_count     = state.get("retry_count", 0)
    max_retries     = state.get("max_retries", 3)

    if needs_revision and retry_count < max_retries:
        runner_map = {
            "tags":       "run_tags",
            "title":      "run_title",
            "tldr":       "run_tldr",
            "references": "run_references",
        }
        targets = [runner_map[a] for a in needs_revision if a in runner_map]
        logger.info(f"[ORCH:ROUTE] Retry #{retry_count} — re-running: {targets}")
        return targets

    if needs_revision:
        logger.warning(
            f"[ORCH:ROUTE] Max retries ({max_retries}) reached. "
            f"Proceeding with unresolved agents: {needs_revision}"
        )
    else:
        logger.info("[ORCH:ROUTE] All agents approved. Proceeding to final output.")

    return ["final_output"]


# ─── Final output node ────────────────────────────────────────────────────────

def final_output_node(state: dict) -> dict:
    """
    Assembles the unified output dict from all agent results.
    This is what main.py reads and saves to disk.
    """
    ctx = state.get("shared_context", {})
    feedback = state.get("review_feedback", [])

    # Build a clean review summary
    review_summary = {
        "cycles":   state.get("retry_count", 1),
        "feedback": [
            {
                "agent":    fb["agent"],
                "approved": fb["approved"],
                "score":    fb["score"],
                "feedback": fb.get("feedback", ""),
            }
            for fb in feedback
        ],
        "unresolved_agents": state.get("agents_needing_revision", []),
    }

    final_output = {
        "source":         state.get("source_name", "unknown"),
        "processed_at":   datetime.now().isoformat(),
        "shared_context": {
            k: v for k, v in ctx.items() if k != "article_text"   # omit full text from output
        },
        "tags":           state.get("tags_result",        {}).get("output", {}),
        "title":          state.get("title_result",       {}).get("output", {}),
        "tldr":           state.get("tldr_result",        {}).get("output", {}),
        "references":     state.get("references_result",  {}).get("output", {}),
        "review":         review_summary,
    }

    logger.info("[ORCH:FINAL] Output assembled.")
    return {"final_output": final_output}


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_orchestrator_graph() -> Any:
    """
    Constructs and compiles the top-level Author Assist orchestrator graph.

    Returns a compiled LangGraph runnable.
    """
    builder = StateGraph(OrchestratorState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    builder.add_node("manager",       manager_node)
    builder.add_node("run_tags",      run_tags_node)
    builder.add_node("run_title",     run_title_node)
    builder.add_node("run_tldr",      run_tldr_node)
    builder.add_node("run_references", run_references_node)
    builder.add_node("reviewer",      reviewer_node)
    builder.add_node("final_output",  final_output_node)

    # ── Edges ──────────────────────────────────────────────────────────────
    # Entry: manager first
    builder.add_edge(START, "manager")

    # After manager: fan-out to all 4 agents in parallel
    builder.add_edge("manager", "run_tags")
    builder.add_edge("manager", "run_title")
    builder.add_edge("manager", "run_tldr")
    builder.add_edge("manager", "run_references")

    # After each agent: fan-in to reviewer
    builder.add_edge("run_tags",        "reviewer")
    builder.add_edge("run_title",       "reviewer")
    builder.add_edge("run_tldr",        "reviewer")
    builder.add_edge("run_references",  "reviewer")

    # After reviewer: conditional routing
    # Selective agents re-run OR go to final_output
    builder.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "run_tags":      "run_tags",
            "run_title":     "run_title",
            "run_tldr":      "run_tldr",
            "run_references": "run_references",
            "final_output":  "final_output",
        },
    )

    builder.add_edge("final_output", END)

    return builder.compile()
