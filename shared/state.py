"""
shared/state.py — Master state schemas for the Author Assist multi-agent system.

Defines:
  • SharedContext   — produced by the Manager, consumed by all parallel agents
  • AgentResult     — produced by each worker agent, consumed by the Reviewer
  • ReviewFeedback  — per-agent verdict from the Reviewer
  • OrchestratorState — the single state object flowing through the top-level graph
"""

from typing import TypedDict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Manager output / Agent input
# ─────────────────────────────────────────────────────────────────────────────

class SharedContext(TypedDict):
    """
    Rich understanding of the article produced by the Manager node.
    All parallel worker agents receive this before they begin work.
    """
    article_text: str           # full raw text passed through for agent use
    key_themes: list[str]       # e.g. ["LLM benchmarking", "combinatorial optimisation"]
    target_audience: str        # e.g. "ML researchers", "general tech readers"
    main_message: str           # one-sentence core claim or contribution
    domain: str                 # "AI/ML" | "Healthcare" | "Mixed" | "Other"
    language_style: str         # "academic" | "technical blog" | "popular science"


# ─────────────────────────────────────────────────────────────────────────────
# Agent outputs
# ─────────────────────────────────────────────────────────────────────────────

class AgentResult(TypedDict):
    """Standardised result wrapper returned by every worker agent."""
    agent: str          # "tags" | "title" | "tldr" | "references"
    output: dict        # agent-specific structured result (see each agent's state.py)
    status: str         # "success" | "partial" | "failed"
    error: str          # empty string on success; error message on failure


# ─────────────────────────────────────────────────────────────────────────────
# Reviewer output
# ─────────────────────────────────────────────────────────────────────────────

class ReviewFeedback(TypedDict):
    """Per-agent verdict from the Reviewer node."""
    agent: str          # which agent this feedback targets
    approved: bool      # True if score >= APPROVAL_THRESHOLD
    score: float        # 0.0–1.0 quality score
    feedback: str       # specific, actionable improvement instruction (empty if approved)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestrator state
# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    """
    The single state object flowing through the top-level LangGraph orchestrator.

    Fields are populated progressively:
      1. article_text + source_name  — set by the caller
      2. shared_context              — filled by the Manager node
      3. *_result fields             — filled by parallel worker agents
      4. review_feedback             — filled by the Reviewer node
      5. agents_needing_revision     — routing key: which agents to re-run
      6. retry_count                 — guard against infinite re-run loops
      7. final_output                — assembled by the final_output_node
    """
    # ── Input ────────────────────────────────────────────────────────────────
    article_text: str
    source_name: str

    # ── Manager ──────────────────────────────────────────────────────────────
    shared_context: SharedContext

    # ── Worker agent results ─────────────────────────────────────────────────
    tags_result: AgentResult
    title_result: AgentResult
    tldr_result: AgentResult
    references_result: AgentResult

    # ── Reviewer ─────────────────────────────────────────────────────────────
    review_feedback: list[ReviewFeedback]
    agents_needing_revision: list[str]   # e.g. ["tldr", "title"]

    # ── Control flow ─────────────────────────────────────────────────────────
    retry_count: int                     # incremented each reviewer cycle
    max_retries: int                     # default 3 — set by caller or start node

    # ── Final ────────────────────────────────────────────────────────────────
    final_output: dict
    error: str
