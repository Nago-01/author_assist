"""
main.py — Tag Extractor entry point.

"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from file_reader import read_document, get_document_metadata, SUPPORTED_EXTENSIONS

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Sample article (to be used when no input is provided)
SAMPLE_ARTICLE = """
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

Abstract

Large language models (LLMs) have demonstrated remarkable capabilities across a wide range
of natural language processing tasks. However, they continue to struggle with knowledge-intensive
tasks that require access to specific, up-to-date, or domain-specific information. In this paper,
we propose a retrieval-augmented generation (RAG) framework that combines the generative
strengths of transformer-based LLMs with a dense vector retrieval mechanism over an external
knowledge corpus.

Our architecture builds on the foundational work of Vaswani et al. on the transformer and
integrates a bi-encoder retriever inspired by DPR (Dense Passage Retrieval). We evaluate our
approach on the Natural Questions, TriviaQA, and FEVER benchmarks, demonstrating state-of-the-art
results on open-domain question answering.

We also show that RAG substantially reduces hallucination rates compared to vanilla GPT-4 and
LLaMA-3 baselines, particularly in the medical domain. Experiments on clinical notes sourced
from electronic health records (EHR) highlight the promise of RAG for clinical decision support
and health informatics applications. Our findings are consistent with those recently reported
in npj Digital Medicine and The Lancet Digital Health.

This work was conducted in collaboration with researchers at DeepMind, Hugging Face, and the
Allen Institute for AI (AI2). We thank Yoshua Bengio and Fei-Fei Li for their insightful
feedback on early drafts.

The model weights and evaluation code are publicly released on the Hugging Face Hub.
A preliminary version of this work was presented at NeurIPS 2024 and a full version
appears in the Journal of Machine Learning Research (JMLR).

Introduction

The advent of foundation models such as GPT, BERT, and the vision transformer (ViT) has
transformed the NLP landscape. Yet a persistent limitation of these models is their reliance
on parametric knowledge frozen at training time. When queried about recent events, niche
domains, or highly specific factual claims, even the most capable LLMs — including Claude,
Gemini, and Mistral — tend to hallucinate plausible-sounding but incorrect responses.

Retrieval-augmented generation (RAG) addresses this by equipping the model with a retrieval
module that fetches relevant documents from a knowledge base at inference time. The retrieved
passages are concatenated with the user query and passed to the generator, allowing the model
to ground its responses in verifiable evidence.

This paper makes the following contributions:
  1. We propose a modular RAG pipeline with a PEFT-tuned LLaMA backbone and a contrastive
     retriever trained with LoRA adapters.
  2. We introduce a new healthcare benchmark — MedRAG-QA — comprising 5,000 questions derived
     from JAMA, NEJM, and BMJ case reports.
  3. We demonstrate that federated learning can be used to train the retriever across multiple
     hospital systems without sharing raw patient data, addressing privacy concerns raised in
     recent HIPAA-related policy discussions.

Methods

Our retriever is a bi-encoder based on BERT-large. Documents are encoded offline and stored in
a FAISS vector index. At query time, the top-k documents are retrieved via maximum inner product
search (MIPS) and passed to a LLaMA-3.3-70B generator, hosted on Groq's inference platform
for low-latency serving.

We fine-tune the generator using instruction tuning on a curated mixture of:
  - 120K open-domain QA pairs from Natural Questions
  - 45K clinical QA pairs from EHR-derived datasets, validated by medical professionals
  - 20K multi-hop reasoning chains generated using chain-of-thought prompting

Evaluation

We report results on four benchmarks: Natural Questions (NQ), TriviaQA, FEVER, and MedRAG-QA.
Our RAG model outperforms GPT-4, Mixtral-8x7B, and Gemma-2 on all four benchmarks.
On MedRAG-QA, we observe a 14.3% absolute improvement over the strongest baseline.

The model is also evaluated on Named Entity Recognition (NER) using spaCy's en_core_web_lg
pipeline, which was used to pre-annotate our training corpus.

Conclusion

RAG represents a compelling path toward grounding generative AI in verifiable knowledge.
Our results suggest that combining retrieval with fine-tuned LLMs — particularly in high-stakes
domains such as healthcare — can meaningfully reduce hallucination and improve factual accuracy.
Future work will explore integration with knowledge graphs and real-time retrieval from
medical databases such as PubMed and ClinicalTrials.gov.
"""


def _make_output_path(source_name: str) -> Path:
    """
    Builds a unique output filename using the source document name + timestamp.

    Examples:
        "rag_paper.pdf"    -> outputs/rag_paper_20250220_143502.json
        "inline_text"      -> outputs/inline_text_20250220_143502.json
        "sample_article"   -> outputs/sample_article_20250220_143502.json
    """
    # Strip extension and sanitise the stem for use in a filename
    stem = Path(source_name).stem           
    stem = stem.replace(" ", "_")    
    stem = "".join(c for c in stem if c.isalnum() or c in "_-")  # strip special chars

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    return output_dir / f"{stem}_{timestamp}.json"


def print_results(state: dict, source_name: str = "article") -> None:
    """Pretty-prints the tag extraction results to stdout and saves a timestamped JSON file."""
    print("\n" + "═" * 70)
    print("  TAG EXTRACTOR — RESULTS")
    print("═" * 70)

    error = state.get("error", "")
    if error:
        print(f"\nError: {error}\n")
        return

    # Candidate summary
    gz  = state.get("gazetteer_tags", [])
    sp  = state.get("spacy_tags", [])
    llm = state.get("llm_tags", [])
    all_cands = state.get("all_candidate_tags", [])

    print(f"\nCandidates extracted:")
    print(f"   • Gazetteer  : {len(gz):>3} tags")
    print(f"   • spaCy NER  : {len(sp):>3} tags")
    print(f"   • LLM        : {len(llm):>3} tags")
    print(f"   • Total (deduped): {len(all_cands)}")

    # Final tags
    final = state.get("final_tags", [])
    print(f"\nFinal top-{state.get('top_n', 10)} tags:\n")
    print(f"  {'#':<4} {'Tag':<35} {'Category':<25} Rationale")
    print(f"  {'─'*4} {'─'*35} {'─'*25} {'─'*40}")
    for i, tag in enumerate(final, 1):
        rationale = tag.get("rationale", "")[:60]
        print(f"  {i:<4} {tag['tag']:<35} {tag['category']:<25} {rationale}")

    print("\n" + "═" * 70)

    # Save timestamped JSON — never overwrites a previous run
    output_path = _make_output_path(source_name)
    with open(output_path, "w") as f:
        json.dump({
            "source": source_name,
            "processed_at": datetime.now().isoformat(),
            "candidate_counts": {
                "gazetteer": len(gz),
                "spacy": len(sp),
                "llm": len(llm),
                "total_deduped": len(all_cands),
            },
            "final_tags": final,
        }, f, indent=2)
    print(f"\nOutput saved to: {output_path.resolve()}\n")


def main():
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    parser = argparse.ArgumentParser(
        description=f"Tag Extractor — AI Publication Tagging System\nSupported file types: {supported}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--file", type=str,
        help=f"Path to a document file ({supported})"
    )
    group.add_argument("--text", type=str, help="Article text passed directly as a string")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY environment variable not set. Create a .env file or export it.")
        sys.exit(1)

    # Resolve article text
    if args.file:
        path = Path(args.file)
        if not path.exists():
            logger.error(f"File not found: {path}")
            sys.exit(1)

        try:
            meta = get_document_metadata(args.file)
            logger.info(
                f"Document: {meta['filename']} | "
                f"Type: {meta['extension'].upper()} | "
                f"Size: {meta['size_kb']} KB"
                + (f" | Pages: {meta['pages']}" if meta.get("pages") else "")
            )
        except Exception:
            pass

        try:
            article_text = read_document(args.file)
        except ValueError as e:
            logger.error(str(e))
            logger.error(f"Supported formats: {supported}")
            sys.exit(1)
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            sys.exit(1)

        if not article_text.strip():
            logger.error(
                "Extracted text is empty. "
                "The file may be image-only, password-protected, or corrupted."
            )
            sys.exit(1)

        logger.info(f"Successfully extracted {len(article_text.split())} words from {path.name}")
        source_name = path.name

    elif args.text:
        article_text = args.text
        logger.info(f"Using inline article text ({len(article_text.split())} words)")
        source_name = "inline_text"
    else:
        article_text = SAMPLE_ARTICLE
        logger.info("No input specified — using built-in sample article.")
        source_name = "sample_article"

    # Run the LangGraph pipeline
    sys.path.insert(0, str(Path(__file__).parent))
    from graph import tag_extractor_graph

    logger.info("Running Tag Extractor graph...")
    initial_state = {
        "article_text": article_text,
        "top_n": 10,
    }

    final_state = tag_extractor_graph.invoke(initial_state)
    print_results(final_state, source_name=source_name)


if __name__ == "__main__":
    main()