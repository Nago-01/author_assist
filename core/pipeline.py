"""
core/pipeline.py
----------------
Main orchestration layer.

Flow (mirrors the architecture diagram exactly):
    START
      ↓
    Manager          — builds SharedContext once
      ↓
    [parallel fan-out]
    Title | TLDR | Tags | References  — all run concurrently
      ↓
    Reviewer         — per-agent targeted verdicts
      ↓ (loop: only failed agents re-run, up to MAX_REVISIONS)
    END              — assemble FinalOutput

Design decisions
----------------
- asyncio.gather drives true parallel execution (each agent is async-wrapped).
- The Reviewer returns a per-agent verdict map; only "needs_revision" agents
  re-enter the next round.
- MAX_REVISIONS caps the loop to avoid infinite cycling.
- Adding a new agent = register it in AGENT_REGISTRY. Nothing else changes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from core.base_agent import AgentResult, SharedContext
from core.manager import manager_node
from core.reviewer import reviewer_node
from core.state import PipelineState

logger = logging.getLogger(__name__)

MAX_REVISIONS = 3  # maximum reviewer → re-run cycles


# ---------------------------------------------------------------------------
# Agent registry  (import here to keep pipeline.py as the single wiring point)
# ---------------------------------------------------------------------------

def _build_registry() -> dict[str, Any]:
    """Lazy import so agents can be used standalone without importing pipeline."""
    from agents.title.agent import TitleAgent
    from agents.tldr.agent import TLDRAgent
    from agents.tags.agent import TagsAgent
    from agents.references.agent import ReferencesAgent

    return {
        "title_generator": TitleAgent(),
        "tldr_generator": TLDRAgent(),
        "tags_generator": TagsAgent(),
        "references_generator": ReferencesAgent(),
    }


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def _run_agent_async(
    agent_name: str,
    agent: Any,
    context: SharedContext,
    feedback: str | None,
    revision_count: int,
) -> AgentResult:
    """Wrap a synchronous agent.run() call in a thread so it doesn't block the
    event loop while the other agents are running in parallel."""
    loop = asyncio.get_event_loop()
    result: AgentResult = await loop.run_in_executor(
        None,
        lambda: agent.run(context, feedback),
    )
    result.revision_count = revision_count
    return result


async def _run_agents_parallel(
    agent_registry: dict[str, Any],
    context: SharedContext,
    feedback_map: dict[str, str],          # agent_name → feedback (or "")
    revision_count: int,
) -> dict[str, AgentResult]:
    """Fan-out: run every agent in agent_registry concurrently."""
    tasks = [
        _run_agent_async(
            name,
            agent,
            context,
            feedback_map.get(name) or None,
            revision_count,
        )
        for name, agent in agent_registry.items()
    ]
    results_list: list[AgentResult] = await asyncio.gather(*tasks)
    return {r.agent_name: r for r in results_list}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(text: str) -> dict[str, Any]:
    """
    Synchronous entry point — wraps the async pipeline for CLI / script use.

    Returns a FinalOutput dict:
    {
        "title":       {...},
        "tldr":        {...},
        "tags":        {...},
        "references":  {...},
        "meta": {
            "revision_rounds": int,
            "review_verdicts": {...},
            "timestamp": str,
        }
    }
    """
    return asyncio.run(_async_pipeline(text))


async def _async_pipeline(text: str) -> dict[str, Any]:
    """Core async pipeline. Called by run_pipeline()."""
    agent_registry = _build_registry()

    # ------------------------------------------------------------------ #
    # 1. Manager — build SharedContext
    # ------------------------------------------------------------------ #
    state: PipelineState = {"raw_text": text, "revision_round": 0}
    state = manager_node(state)

    if state.get("error"):
        raise RuntimeError(f"Manager failed: {state['error']}")

    context: SharedContext = state["shared_context"]
    logger.info("Manager complete — domain=%s, themes=%s", context.domain, context.key_themes)

    # ------------------------------------------------------------------ #
    # 2. Initial parallel run — all agents, no feedback yet
    # ------------------------------------------------------------------ #
    feedback_map: dict[str, str] = {}
    current_results: dict[str, AgentResult] = await _run_agents_parallel(
        agent_registry, context, feedback_map, revision_count=0
    )
    logger.info("Initial parallel run complete — %d agents", len(current_results))

    # ------------------------------------------------------------------ #
    # 3. Reviewer → selective re-run loop
    # ------------------------------------------------------------------ #
    final_verdicts: dict[str, str] = {}
    revision_round = 0

    for round_num in range(1, MAX_REVISIONS + 2):  # +2: initial review + up to MAX re-runs
        state = {
            **state,
            "agent_results": current_results,
            "revision_round": revision_round,
        }
        state = reviewer_node(state)

        verdicts: dict[str, str] = state["review_verdicts"]
        all_approved: bool = state["all_approved"]
        final_verdicts = verdicts

        logger.info(
            "Review round %d — all_approved=%s, verdicts=%s",
            round_num, all_approved, verdicts,
        )

        if all_approved or revision_round >= MAX_REVISIONS:
            break

        # ---- selective re-run: only failed agents -------------------- #
        agents_to_rerun = {
            name: agent_registry[name]
            for name, verdict in verdicts.items()
            if verdict != "approved" and name in agent_registry
        }

        if not agents_to_rerun:
            break

        revision_round += 1
        feedback_map = {
            name: verdict
            for name, verdict in verdicts.items()
            if verdict != "approved"
        }

        revised = await _run_agents_parallel(
            agents_to_rerun, context, feedback_map, revision_count=revision_round
        )

        # Patch only the revised agents into the result map
        current_results = {**current_results, **revised}
        logger.info(
            "Re-run round %d — revised agents: %s", revision_round, list(revised.keys())
        )

    # ------------------------------------------------------------------ #
    # 4. Assemble final output
    # ------------------------------------------------------------------ #
    final_output: dict[str, Any] = {
        "title": current_results.get("title_generator", AgentResult("title_generator", {})).output,
        "tldr": current_results.get("tldr_generator", AgentResult("tldr_generator", {})).output,
        "tags": current_results.get("tags_generator", AgentResult("tags_generator", {})).output,
        "references": current_results.get("references_generator", AgentResult("references_generator", {})).output,
        "meta": {
            "revision_rounds": revision_round,
            "review_verdicts": final_verdicts,
            "shared_context": {
                "key_themes": context.key_themes,
                "target_audience": context.target_audience,
                "main_message": context.main_message,
                "domain": context.domain,
                "article_type": context.article_type,
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    }

    return final_output
