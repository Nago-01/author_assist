"""
agents/tags/state.py — State schema for the Tags extraction agent.

Self-contained: this TypedDict is used only inside the tags graph.
The orchestrator maps results into AgentResult when collecting outputs.
"""

from typing import TypedDict, Annotated
import operator


class ExtractedTag(TypedDict):
    """A single candidate tag from any extractor."""
    term: str
    category: str
    source: str       # "gazetteer" | "spacy" | "llm"
    confidence: float


class FinalTag(TypedDict):
    """A final selected and ranked tag."""
    tag: str
    category: str
    rationale: str


class TagAgentState(TypedDict):
    """
    Internal state for the Tags extraction LangGraph workflow.

    Inputs (required before invoke):
      - article_text   : full article text
      - shared_context : SharedContext dict from the Manager
      - revision_note  : optional Reviewer feedback (empty string on first run)
      - top_n          : how many final tags to produce (default 10)

    Outputs (populated by nodes):
      - gazetteer_tags, spacy_tags, llm_tags  : parallel extractor outputs
      - all_candidate_tags                    : merged, deduplicated
      - final_tags                            : top-N selected by aggregator
    """
    article_text: str
    shared_context: dict        # SharedContext passed through from Orchestrator
    revision_note: str          # Reviewer feedback — empty on first run
    top_n: int

    # Parallel extractor outputs — LangGraph merges concurrent branches with operator.add
    gazetteer_tags: Annotated[list[ExtractedTag], operator.add]
    spacy_tags:     Annotated[list[ExtractedTag], operator.add]
    llm_tags:       Annotated[list[ExtractedTag], operator.add]

    all_candidate_tags: list[ExtractedTag]
    final_tags: list[FinalTag]
    error: str
