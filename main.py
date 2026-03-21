"""
main.py — Entry point for the Author Assist Multi-Agent System.

Usage:
    python main.py                           # Run built-in sample article
    python main.py --file paper.pdf          # Run on a document (PDF, DOCX, TXT)
    python main.py --text "Your text..."     # Run on inline text
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Logging setup (must happen before importing other modules)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Ensure the current directory is in PYTHONPATH so we can import orchestrator/, shared/
sys.path.insert(0, str(Path(__file__).parent))

from shared.file_reader import read_document, get_document_metadata, SUPPORTED_EXTENSIONS
from orchestrator.graph import build_orchestrator_graph

# Built-in sample article
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
    stem = Path(source_name).stem           
    stem = stem.replace(" ", "_")    
    stem = "".join(c for c in stem if c.isalnum() or c in "_-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir / f"{stem}_{timestamp}.json"


def print_results(final_output: dict) -> None:
    """Pretty-prints the final orchestrator results to the console."""
    print("\n" + "=" * 70)
    print("  AUTHOR ASSIST — MULTI-AGENT RESULTS")
    print("=" * 70)

    # 1. Context
    ctx = final_output.get("shared_context", {})
    print(f"\n[SHARED CONTEXT]")
    print(f"Domain    : {ctx.get('domain', '')}")
    print(f"Audience  : {ctx.get('target_audience', '')}")
    print(f"Message   : {ctx.get('main_message', '')}")
    print(f"Themes    : {', '.join(ctx.get('key_themes', []))}")

    # 2. Title
    title = final_output.get("title", {})
    print(f"\n[TITLE GENERATOR]")
    print(f"Primary   : {title.get('primary', '')}")
    print(f"Alternates: {', '.join(title.get('alternates', []))}")

    # 3. TLDR
    tldr = final_output.get("tldr", {})
    print(f"\n[TLDR GENERATOR]")
    print(f"({tldr.get('word_count', 0)} words) {tldr.get('tldr', '')}")

    # 4. Tags
    tags_out = final_output.get("tags", {})
    final_tags = tags_out.get("final_tags", [])
    print(f"\n[TAG EXTRACTOR]")
    print(f"Top tags  : {', '.join(t.get('tag', '') for t in final_tags[:7])}...")

    # 5. References
    refs = final_output.get("references", {})
    ref_list = refs.get("references", [])
    print(f"\n[REFERENCES FORMATTER]")
    print(f"Found {refs.get('count', 0)} references.")
    for i, ref in enumerate(ref_list[:3], 1):
        print(f"  {i}. {ref[:80]}...")
    if len(ref_list) > 3:
        print(f"  ...and {len(ref_list) - 3} more.")

    # 6. Review Summary
    rev = final_output.get("review", {})
    print(f"\n[REVIEW SUMMARY]")
    print(f"Cycles run: {rev.get('cycles', 0)}")
    
    unresolved = rev.get("unresolved_agents", [])
    if unresolved:
        print(f"\n⚠️ WARNING: The following agents could not be resolved after max retries: {unresolved}")
        for fb in rev.get("feedback", []):
            if fb.get("agent") in unresolved:
                print(f"   [{fb['agent'].upper()}] {fb.get('feedback', '')}")

    print("\n" + "=" * 70)


def main():
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    parser = argparse.ArgumentParser(
        description=f"Author Assist Multi-Agent CLI\nSupported files: {supported}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help=f"Path to a document file ({supported})")
    group.add_argument("--text", type=str, help="Article text passed directly as a string")
    args = parser.parse_args()

    # Determine input
    if args.file:
        path = Path(args.file)
        try:
            meta = get_document_metadata(args.file)
            logger.info(f"Document: {meta['filename']} | Type: {meta['extension'].upper()} | Size: {meta['size_kb']} KB")
            article_text = read_document(args.file)
            source_name = path.name
        except Exception as e:
            logger.error(str(e))
            sys.exit(1)
        if not article_text.strip():
            logger.error("Extracted text is empty. File may be image-only.")
            sys.exit(1)

    elif args.text:
        article_text = args.text
        logger.info(f"Using inline article text ({len(article_text.split())} words)")
        source_name = "inline_text"

    else:
        article_text = SAMPLE_ARTICLE
        logger.info("No input specified — using built-in sample article.")
        source_name = "sample_article"

    # Build and run graph
    logger.info("Building multi-agent orchestrator graph...")
    graph = build_orchestrator_graph()

    initial_state = {
        "article_text": article_text,
        "source_name": source_name,
        "max_retries": 3,
    }

    logger.info("Invoking orchestrator...")
    final_state = graph.invoke(initial_state)

    # Save output
    final_output = final_state.get("final_output", {})
    output_path = _make_output_path(source_name)
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    logger.info(f"Workflow complete. Saved to: {output_path.resolve()}")
    print_results(final_output)


if __name__ == "__main__":
    main()
