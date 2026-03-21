"""
core/reviewer.py
----------------
Reviewer node — evaluates ALL agent outputs and issues PER-AGENT verdicts.

This is NOT a pass/fail gate. It returns targeted feedback for each agent:
    - "approved"             → agent output is good, no re-run needed
    - "<specific feedback>"  → what exactly needs to change (agent re-runs)

Only agents that receive feedback are re-run in the next round.
"""

from __future__ import annotations

import json
import os

from groq import Groq

from core.base_agent import AgentResult, SharedContext
from core.state import PipelineState

_CLIENT: Groq | None = None


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _CLIENT


_SYSTEM_PROMPT = """You are a strict but fair publication reviewer. You will be given:
1. A SharedContext (themes, audience, main_message) produced by the Manager.
2. Outputs from four agents: title_generator, tldr_generator, tags_generator, references_generator.

Your job: evaluate each agent's output against the SharedContext and return a JSON object
with exactly these four keys:
  "title_generator", "tldr_generator", "tags_generator", "references_generator"

For each key, write EITHER:
  - The string "approved"  (output is good and coherent with the context)
  - A short, specific feedback string explaining what must be fixed (e.g.
    "The title does not reflect the benchmarking angle — make it more specific.")

Return ONLY the JSON object. No markdown, no preamble. No extra keys."""


def reviewer_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: reads agent_results + shared_context → writes review_verdicts.
    """
    context: SharedContext = state["shared_context"]
    results: dict[str, AgentResult] = state.get("agent_results", {})

    # Build a compact summary of each agent's output for the reviewer
    outputs_summary: dict[str, object] = {}
    for name, result in results.items():
        if result.status == "error":
            outputs_summary[name] = f"ERROR: {result.error}"
        else:
            outputs_summary[name] = result.output

    context_summary = {
        "key_themes": context.key_themes,
        "target_audience": context.target_audience,
        "main_message": context.main_message,
        "domain": context.domain,
        "article_type": context.article_type,
    }

    user_content = (
        f"SharedContext:\n{json.dumps(context_summary, indent=2)}\n\n"
        f"Agent Outputs:\n{json.dumps(outputs_summary, indent=2)}"
    )

    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=512,
    )

    raw_json = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    raw_json = raw_json.strip()

    verdicts: dict[str, str] = json.loads(raw_json)

    # Ensure all four agent keys exist (default to approved if missing)
    for agent in ["title_generator", "tldr_generator", "tags_generator", "references_generator"]:
        verdicts.setdefault(agent, "approved")

    # Mark per-agent status in results
    updated_results = dict(results)
    for agent_name, verdict in verdicts.items():
        if agent_name in updated_results:
            updated_results[agent_name] = AgentResult(
                agent_name=updated_results[agent_name].agent_name,
                output=updated_results[agent_name].output,
                revision_count=updated_results[agent_name].revision_count,
                status="approved" if verdict == "approved" else "needs_revision",
                error=updated_results[agent_name].error,
            )

    all_approved = all(v == "approved" for v in verdicts.values())

    return {
        **state,
        "review_verdicts": verdicts,
        "agent_results": updated_results,
        "all_approved": all_approved,
    }
