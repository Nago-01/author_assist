"""
shared/llm.py — Centralised LLM factory for the Author Assist system.

Every agent and orchestrator component imports from here so that:
  • Model name and default settings are changed in one place.
  • The API key is loaded once from the .env at the project root.

Usage:
    from shared.llm import get_llm

    llm = get_llm()                    # default model
    llm = get_llm(temperature=0.0)    # override temperature
    llm = get_llm(model="llama-3.1-8b-instant", max_tokens=512)
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from the project root (two levels up from shared/)
_env_path = Path(__file__).resolve().parent.parent / "code" / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    # Fallback: try project root .env
    load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2048


def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    """
    Returns a configured ChatGroq instance.

    Args:
        model:       Groq model name. Defaults to llama-3.3-70b-versatile.
        temperature: Sampling temperature. Lower = more deterministic.
        max_tokens:  Maximum tokens in the response.

    Raises:
        EnvironmentError: If GROQ_API_KEY is not set.
    """
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Add it to code/.env or export it as an environment variable."
        )

    logger.debug(f"[LLM] Instantiating ChatGroq — model={model}, temp={temperature}")
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )
