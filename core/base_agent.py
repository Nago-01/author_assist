"""
Shared contract that every agent must honour.
Agents are completely ignorant of each other — they only know SharedContext
(produced by the Manager) and optional ReviewerFeedback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# Data contracts

@dataclass
class SharedContext:
    """
    Produced once by the Manager node from the raw input text.
    All agents receive this before they start — everyone is on the same page.
    """
    raw_text: str
    key_themes: list[str]
    target_audience: str
    main_message: str
    domain: str                          # e.g. "AI/ML", "Healthcare", "General"
    article_type: str = "research"       # e.g. "research", "tutorial", "opinion"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """
    Returned by every agent after a run.
    status is set by the Reviewer after evaluation (not by the agent itself).
    """
    agent_name: str
    output: dict[str, Any]
    revision_count: int = 0
    status: str = "pending"              # "pending" | "approved" | "needs_revision"
    error: str | None = None


# Base class

class BaseAgent(ABC):
    """
    Abstract base every agent subclass must implement.
    The `run` method is the only public surface the orchestrator calls.
    Internal LangGraph graphs, LLM clients, etc. are implementation details.
    """

    name: str = "base"

    def run(
        self,
        context: SharedContext,
        feedback: str | None = None,
    ) -> AgentResult:
        """
        Execute the agent.

        Args:
            context:  SharedContext produced by the Manager.
            feedback: Optional targeted feedback string from the Reviewer
                      (only present on re-runs).

        Returns:
            AgentResult with status="pending" (Reviewer sets final status).
        """
        try:
            output = self._execute(context, feedback)
            return AgentResult(agent_name=self.name, output=output)
        except Exception as exc:  # noqa: BLE001
            return AgentResult(
                agent_name=self.name,
                output={},
                status="error",
                error=str(exc),
            )

    @abstractmethod
    def _execute(
        self,
        context: SharedContext,
        feedback: str | None,
    ) -> dict[str, Any]:
        """
        Concrete agents implement their logic here.
        Must return a plain dict that is JSON-serialisable.
        """
