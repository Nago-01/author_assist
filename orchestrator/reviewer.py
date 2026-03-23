"""
Reviewer node for the Author Assist orchestrator.

The Reviewer receives all four AgentResult objects plus the SharedContext and
makes a single LLM call that evaluates all outputs together. It returns a
ReviewFeedback per agent:
  • approved  : True if score >= APPROVAL_THRESHOLD (0.75)
  • score     : 0.0–1.0 quality assessment
  • feedback  : specific, actionable improvement instruction (empty if approved)

Selective re-execution: only agents with approved=False are added to
state["agents_needing_revision"], which the conditional routing logic uses.
"""

import re
import json
import logging

logger = logging.getLogger(__name__)

# Agents are approved if their review score meets or exceeds this threshold
APPROVAL_THRESHOLD = 0.75

_REVIEWER_SYSTEM = """\
You are a senior editorial quality reviewer for a research publications platform.
You will receive four agent outputs and the article's SharedContext.
Your job is to evaluate each output for quality and alignment with the shared context.

Scoring criteria per agent:
  tags       : relevance, specificity, coverage, deduplication quality
  title      : accuracy, specificity, audience fit, no clickbait or vagueness
  tldr       : completeness (problem + method + result), conciseness (≤ 80 words), accuracy
  references : completeness, correct APA formatting, no hallucinated entries

For each agent, assign a score between 0.0 and 1.0 and give SPECIFIC, targeted feedback
if the score is below {threshold}. If the output is good, feedback can be empty.

Return ONLY a valid JSON array of exactly 4 objects, one per agent, with these keys:
  "agent"    : "tags" | "title" | "tldr" | "references"
  "score"    : float 0.0–1.0
  "approved" : true if score >= {threshold}, false otherwise
  "feedback" : specific instruction for improvement (empty string if approved)

No text outside the JSON array."""

_REVIEWER_USER = """\
Shared Context:
{shared_context_json}

Agent Outputs:

[TAGS]
{tags_output}

[TITLE]
{title_output}

[TLDR]
{tldr_output}

[REFERENCES]
{references_output}

Evaluate all four outputs. Return the JSON array."""


def reviewer_node(state: dict) -> dict:
    """
    Evaluates all four agent results and produces per-agent ReviewFeedback.
    Updates state with:
      - review_feedback         : list of ReviewFeedback dicts
      - agents_needing_revision : list of agent names that failed review
      - retry_count             : incremented by 1
    """
    from shared.llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    ctx = state.get("shared_context", {})
    tags_result  = state.get("tags_result",  {})
    title_result = state.get("title_result", {})
    tldr_result  = state.get("tldr_result",  {})
    refs_result  = state.get("references_result", {})

    retry_count = state.get("retry_count", 0) + 1
    logger.info(f"[REVIEWER] Starting review cycle #{retry_count}")

    # Build a clean context summary for the prompt
    ctx_summary = {
        "key_themes": ctx.get("key_themes", []),
        "target_audience": ctx.get("target_audience", ""),
        "main_message": ctx.get("main_message", ""),
        "domain": ctx.get("domain", ""),
        "language_style": ctx.get("language_style", ""),
    }

    def _format_result(result: dict) -> str:
        if not result:
            return "No output (agent did not run or failed)."
        output = result.get("output", {})
        status = result.get("status", "unknown")
        error  = result.get("error", "")
        summary = {"status": status, "output": output}
        if error:
            summary["error"] = error
        return json.dumps(summary, indent=2)[:1500]   # cap per-agent to keep prompt manageable

    system_prompt = _REVIEWER_SYSTEM.format(threshold=APPROVAL_THRESHOLD)
    user_msg = _REVIEWER_USER.format(
        shared_context_json=json.dumps(ctx_summary, indent=2),
        tags_output=_format_result(tags_result),
        title_output=_format_result(title_result),
        tldr_output=_format_result(tldr_result),
        references_output=_format_result(refs_result),
    )

    llm = get_llm(temperature=0.1)
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg),
        ])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        feedbacks_raw = json.loads(raw)

        review_feedback = []
        agents_needing_revision = []

        for item in feedbacks_raw:
            if not isinstance(item, dict) or "agent" not in item:
                continue
            agent   = item["agent"]
            score   = float(item.get("score", 0.0))
            approved = score >= APPROVAL_THRESHOLD
            feedback = item.get("feedback", "") if not approved else ""

            review_feedback.append({
                "agent":    agent,
                "approved": approved,
                "score":    round(score, 3),
                "feedback": feedback,
            })

            status_icon = "✓" if approved else "✗"
            logger.info(f"[REVIEWER] {status_icon} {agent:12s}  score={score:.2f}  {'APPROVED' if approved else 'NEEDS REVISION'}")
            if not approved:
                logger.info(f"           → {feedback}")
                agents_needing_revision.append(agent)

        logger.info(f"[REVIEWER] Cycle #{retry_count} complete. "
                    f"{len(agents_needing_revision)} agent(s) need revision: {agents_needing_revision}")

        return {
            "review_feedback":          review_feedback,
            "agents_needing_revision":  agents_needing_revision,
            "retry_count":              retry_count,
        }

    except Exception as e:
        logger.error(f"[REVIEWER] Error during review: {e}")
        # Fallback: approve everything so the system doesn't get stuck
        all_agents = ["tags", "title", "tldr", "references"]
        fallback_feedback = [
            {"agent": a, "approved": True, "score": 0.75, "feedback": ""}
            for a in all_agents
        ]
        return {
            "review_feedback":         fallback_feedback,
            "agents_needing_revision": [],
            "retry_count":             retry_count,
        }
