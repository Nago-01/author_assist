"""author_assist.core — shared contracts and orchestration."""
from core.base_agent import BaseAgent, SharedContext, AgentResult
from core.pipeline import run_pipeline

__all__ = ["BaseAgent", "SharedContext", "AgentResult", "run_pipeline"]
