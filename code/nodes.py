"""
Nodes for the Tag Extractor LangGraph workflow.

Node inventory:
  1. start_node          
  2. gazetteer_node      
  3. spacy_node         
  4. llm_extractor_node  
  5. aggregator_node     
  6. end_node           
"""

import re
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# START NODE
def start_node(state: dict) -> dict:
    """
    Validates the incoming article text and sets default state values.
    Acts as the entry gate before fan-out to parallel extractors.
    """
    article_text = state.get("article_text", "").strip()

    if not article_text:
        return {
            "error": "No article text provided.",
            "gazetteer_tags": [],
            "spacy_tags": [],
            "llm_tags": [],
            "all_candidate_tags": [],
            "final_tags": [],
            "top_n": 10,
        }

    word_count = len(article_text.split())
    logger.info(f"[START] Article received — {word_count} words.")

    return {
        "article_text": article_text,
        "top_n": state.get("top_n", 10),
        "gazetteer_tags": [],
        "spacy_tags": [],
        "llm_tags": [],
        "all_candidate_tags": [],
        "final_tags": [],
        "error": "",
    }


# GAZETTEER NODE
def gazetteer_node(state: dict) -> dict:
    """
    Scans the article text against the curated GAZETTEER_INDEX.
    Matches longer phrases first to avoid partial overlaps.
    Returns a list of ExtractedTag dicts.
    """
    from gazetteer import GAZETTEER_INDEX

    text = state.get("article_text", "")
    text_lower = text.lower()

    matched: list[dict] = []
    seen_terms: set[str] = set()

    for lower_term, canonical_term, category in GAZETTEER_INDEX:
        if canonical_term in seen_terms:
            continue

        # Word-boundary aware search using regex
        pattern = r'(?<!\w)' + re.escape(lower_term) + r'(?!\w)'
        if re.search(pattern, text_lower):
            matched.append({
                "term": canonical_term,
                "category": category,
                "source": "gazetteer",
                "confidence": 1.0,
            })
            seen_terms.add(canonical_term)

    logger.info(f"[GAZETTEER] {len(matched)} matches found.")
    return {"gazetteer_tags": matched}


# SPACY NODE
SPACY_LABEL_MAP = {
    "ORG":      "Organization",
    "PERSON":   "Person",
    "GPE":      "Geopolitical Entity",
    "LOC":      "Location",
    "PRODUCT":  "Product",
    "EVENT":    "Event",
    "WORK_OF_ART": "Work of Art",
    "LAW":      "Law/Policy",
    "LANGUAGE": "Language",
    "NORP":     "Nationality/Group",
    "FAC":      "Facility",
    "DATE":     "Date",
    "MONEY":    "Financial",
    "PERCENT":  "Statistic",
    "QUANTITY": "Quantity",
    "ORDINAL":  "Ordinal",
    "CARDINAL": "Cardinal",
}

# Labels needed
USEFUL_LABELS = {"ORG", "PERSON", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "NORP"}


def spacy_node(state: dict) -> dict:
    """
    Runs spaCy en_core_web_md NER over the article.
    Filters to entity types meaningful for publication tagging.
    """
    import spacy

    text = state.get("article_text", "")

    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        logger.warning("[SPACY] Model not found. Run: python -m spacy download en_core_web_md")
        return {"spacy_tags": []}

    doc = nlp(text)

    seen: set[str] = set()
    tags: list[dict] = []

    for ent in doc.ents:
        if ent.label_ not in USEFUL_LABELS:
            continue
        term = ent.text.strip()
        if not term or term in seen:
            continue
        # Skip very short tokens or purely numeric strings
        if len(term) < 2 or term.isdigit():
            continue

        seen.add(term)
        tags.append({
            "term": term,
            "category": SPACY_LABEL_MAP.get(ent.label_, ent.label_),
            "source": "spacy",
            "confidence": 0.85,
        })

    logger.info(f"[SPACY] {len(tags)} entities extracted.")
    return {"spacy_tags": tags}



# LLM EXTRACTOR NODE
LLM_EXTRACTION_SYSTEM_PROMPT = """You are an expert scientific editor specialising in AI and healthcare publications.
Your task is to extract meaningful tags from the given article text.

Tags should capture:
- Key AI/ML concepts, methods, and architectures
- Named AI models, datasets, and benchmarks
- Organisations, labs, and research groups
- Prominent researchers or authors mentioned
- Application domains (e.g. healthcare, robotics, finance)
- Conferences or journals referenced

Return ONLY a valid JSON array. Each element must be an object with:
  "term"      : the tag text (canonical, concise)
  "category"  : one of [AI/ML Concept, AI Model, Dataset, Benchmark, Organization, Person, Application Domain, Conference, Journal, Method, Other]
  "confidence": a float between 0.0 and 1.0 reflecting your certainty

Example:
[
  {"term": "RAG", "category": "AI/ML Concept", "confidence": 0.97},
  {"term": "DeepMind", "category": "Organization", "confidence": 0.99}
]

Do not include any text outside the JSON array. Do not truncate the array."""

LLM_EXTRACTION_USER_TEMPLATE = """Extract all meaningful publication tags from the following article text:

---
{article_text}
---

Return a JSON array only."""


def llm_extractor_node(state: dict) -> dict:
    """
    Uses Groq's llama-3.3-70b-versatile to perform semantic tag extraction.
    The LLM reasons about the article holistically and surfaces tags that
    rule-based methods may miss (implicit concepts, model names, domains).
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    text = state.get("article_text", "")

    # Truncate very long articles to stay within context limits
    max_chars = 12_000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Article truncated for extraction]"

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=2048,
    )

    messages = [
        SystemMessage(content=LLM_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=LLM_EXTRACTION_USER_TEMPLATE.format(article_text=text)),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        extracted = json.loads(raw)

        tags = []
        for item in extracted:
            if not isinstance(item, dict) or "term" not in item:
                continue
            tags.append({
                "term": item.get("term", "").strip(),
                "category": item.get("category", "Other"),
                "source": "llm",
                "confidence": float(item.get("confidence", 0.8)),
            })

        logger.info(f"[LLM EXTRACTOR] {len(tags)} tags extracted.")
        return {"llm_tags": tags}

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"[LLM EXTRACTOR] Error: {e}")
        return {"llm_tags": []}



# AGGREGATOR NODE
AGGREGATION_SYSTEM_PROMPT = """You are a senior editorial AI assistant for an AI and healthcare publications platform.

You will receive a list of candidate tags extracted by three different methods:
  1. Gazetteer (exact dictionary match — high precision)
  2. spaCy NER (statistical NLP model)
  3. LLM semantic extraction (language model reasoning)

Your job is to select the BEST {top_n} final tags that would be most useful to a reader or search engine discovering this article.

Selection criteria:
- Prefer tags that are specific, meaningful, and publication-relevant
- Deduplicate synonyms (keep the more canonical or precise form)
- Favour terms that appear across multiple extraction sources
- Prefer concepts over generic words
- Balance coverage: include both technical concepts AND entities (orgs, people, events)

Return ONLY a valid JSON array of exactly {top_n} objects. Each object must have:
  "tag"       : the final tag text
  "category"  : the tag category
  "rationale" : one sentence explaining why this tag was selected

Do not include any text outside the JSON array."""

AGGREGATION_USER_TEMPLATE = """Article (first 2000 chars for context):
---
{article_snippet}
---

Candidate tags from all three extractors:
{candidates_json}

Select the best {top_n} tags. Return a JSON array only."""


def aggregator_node(state: dict) -> dict:
    """
    Collects all candidate tags, deduplicates, and uses Groq LLM to
    intelligently select the top-N most publication-relevant final tags.
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    gazetteer_tags = state.get("gazetteer_tags", [])
    spacy_tags     = state.get("spacy_tags", [])
    llm_tags       = state.get("llm_tags", [])
    top_n          = state.get("top_n", 10)
    article_text   = state.get("article_text", "")

    all_candidates = gazetteer_tags + spacy_tags + llm_tags
    logger.info(f"[AGGREGATOR] Total candidates: {len(all_candidates)} "
                f"(gazetteer={len(gazetteer_tags)}, spacy={len(spacy_tags)}, llm={len(llm_tags)})")

    if not all_candidates:
        logger.warning("[AGGREGATOR] No candidates received from any extractor.")
        return {"all_candidate_tags": [], "final_tags": []}

    # Build a deduplicated candidate list for the LLM prompt
    seen: set[str] = set()
    deduped: list[dict] = []
    for tag in all_candidates:
        key = tag["term"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(tag)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=2048,
    )

    candidates_json = json.dumps(deduped, indent=2)
    article_snippet = article_text[:2000]

    messages = [
        SystemMessage(content=AGGREGATION_SYSTEM_PROMPT.format(top_n=top_n)),
        HumanMessage(content=AGGREGATION_USER_TEMPLATE.format(
            article_snippet=article_snippet,
            candidates_json=candidates_json,
            top_n=top_n,
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        final = json.loads(raw)
        final_tags = []
        for item in final:
            if not isinstance(item, dict) or "tag" not in item:
                continue
            final_tags.append({
                "tag": item.get("tag", "").strip(),
                "category": item.get("category", "Other"),
                "rationale": item.get("rationale", ""),
            })

        logger.info(f"[AGGREGATOR] {len(final_tags)} final tags selected.")
        return {
            "all_candidate_tags": deduped,
            "final_tags": final_tags,
        }

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"[AGGREGATOR] Error: {e}")
        # Fallback: return top-N by confidence from candidates
        sorted_candidates = sorted(deduped, key=lambda x: x.get("confidence", 0), reverse=True)
        fallback_tags = [
            {"tag": t["term"], "category": t["category"], "rationale": "Fallback (aggregator error)"}
            for t in sorted_candidates[:top_n]
        ]
        return {"all_candidate_tags": deduped, "final_tags": fallback_tags}



# 6. END NODE
def end_node(state: dict) -> dict:
    """
    Finalises the workflow. Logs a summary and returns the state unchanged.
    The caller can read state['final_tags'] for the result.
    """
    final_tags = state.get("final_tags", [])
    error      = state.get("error", "")

    if error:
        logger.error(f"[END] Workflow completed with error: {error}")
    else:
        logger.info(f"[END] Workflow complete. {len(final_tags)} final tags produced.")
        for i, tag in enumerate(final_tags, 1):
            logger.info(f"  {i:>2}. [{tag['category']}] {tag['tag']}")

    return state