"""
State schema for the Tag Extractor LangGraph workflow.

The TagState TypedDict is the single shared state object that flows
through every node in the graph.
"""

from typing import TypedDict, Annotated
import operator


class ExtractedTag(TypedDict):
    """Represents a single extracted tag with its source and category."""
    term: str         
    category: str       
    source: str         # gazetteer, spacy, llm
    confidence: float   


class FinalTag(TypedDict):
    """Represents a final selected tag after aggregation."""
    tag: str
    category: str
    rationale: str      # Why the aggregator selected this tag


class TagState(TypedDict):
    """
    Shared state flowing through the entire LangGraph workflow.

    Fields populated progressively as nodes execute:
      - article_text      : set by the caller / start node
      - top_n             : how many final tags to select (fixed at 10)
      - gazetteer_tags    : filled by the gazetteer extraction node
      - spacy_tags        : filled by the spaCy extraction node
      - llm_tags          : filled by the LLM extraction node
      - all_candidate_tags: union of above three, assembled before aggregation
      - final_tags        : top-N tags selected by the aggregation node
      - error             : any error message (non-fatal, nodes degrade gracefully)
    """
    article_text: str
    top_n: int

    # Parallel extractor outputs so LangGraph can merge results from concurrent branches into a single list
    gazetteer_tags: Annotated[list[ExtractedTag], operator.add]
    spacy_tags:     Annotated[list[ExtractedTag], operator.add]
    llm_tags:       Annotated[list[ExtractedTag], operator.add]

  
    all_candidate_tags: list[ExtractedTag]
    final_tags: list[FinalTag]
    error: str