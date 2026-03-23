"""
CLI entry point for the Author Assist multi-agent pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env (GROQ_API_KEY)
load_dotenv(override=True)


def _check_env() -> None:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        print(
            "[ERROR] GROQ_API_KEY is not set.\n"
            "Create a .env file in the project root with:\n"
            "  GROQ_API_KEY=your_key_here\n"
            "Get your key at: https://console.groq.com",
            file=sys.stderr,
        )
        sys.exit(1)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def _print_results(output: dict, verbose: bool) -> None:
    """Pretty-print the pipeline results to stdout."""
    sep = "─" * 60

    print(f"\n{sep}")
    print("  AUTHOR ASSIST — RESULTS")
    print(sep)

    # Title
    title_data = output.get("title", {})
    print(f"\n📌  TITLE\n    {title_data.get('final_title', 'N/A')}")
    if verbose and title_data.get("alternative_titles"):
        print("    Alternatives:")
        for alt in title_data["alternative_titles"]:
            print(f"      • {alt}")

    # TLDR
    tldr_data = output.get("tldr", {})
    print(f"\n📝  ONE-LINER\n    {tldr_data.get('one_liner', 'N/A')}")
    print(f"\n📄  TLDR\n    {tldr_data.get('final_tldr', 'N/A')}")
    if verbose and tldr_data.get("key_points"):
        print("\n    Key points extracted:")
        for pt in tldr_data["key_points"]:
            print(f"      • {pt}")

    # Tags
    tags_data = output.get("tags", {})
    tags = tags_data.get("final_tags", [])
    print(f"\n🏷️   TAGS  ({len(tags)} total)")
    for tag in tags:
        if isinstance(tag, dict):
            print(f"      [{tag.get('category', '?')}]  {tag.get('tag', tag)}")
        else:
            print(f"      • {tag}")
    if verbose and tags_data.get("candidate_counts"):
        cc = tags_data["candidate_counts"]
        print(
            f"\n    Extraction sources — "
            f"Gazetteer: {cc.get('gazetteer', 0)}  "
            f"spaCy: {cc.get('spacy', 0)}  "
            f"LLM: {cc.get('llm', 0)}  "
            f"Deduped total: {cc.get('total_deduped', 0)}"
        )

    # References
    refs_data = output.get("references", {})
    refs = refs_data.get("final_references", [])
    print(
        f"\n📚  REFERENCES  "
        f"({len(refs)} found · style: {refs_data.get('citation_style', 'Unknown')})"
    )
    for i, ref in enumerate(refs[:5], 1):
        if isinstance(ref, dict):
            print(f"      {i}. {ref.get('formatted', ref.get('raw', str(ref)))}")
        else:
            print(f"      {i}. {ref}")
    if len(refs) > 5:
        print(f"      … and {len(refs) - 5} more (see output file)")

    # Meta
    meta = output.get("meta", {})
    print(f"\n{sep}")
    print(
        f"  Revision rounds: {meta.get('revision_rounds', 0)}  |  "
        f"Timestamp: {meta.get('timestamp', 'N/A')}"
    )
    if verbose and meta.get("review_verdicts"):
        print("\n  Final review verdicts:")
        for agent, verdict in meta["review_verdicts"].items():
            status = "✓" if verdict == "approved" else "↺"
            print(f"    {status}  {agent}: {verdict}")
    print(sep + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Author Assist — AI-powered publication metadata generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", metavar="PATH", help="Path to article file (PDF/DOCX/DOC/TXT)")
    source.add_argument("--text", metavar="TEXT", help="Article text as inline string")

    parser.add_argument(
        "--output",
        metavar="PATH",
        default="author_assist_output.json",
        help="Output JSON file path (default: author_assist_output.json)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    _check_env()
    _setup_logging(args.verbose)

    # Read input text#
    if args.file:
        from core.file_reader import read_file  # noqa: PLC0415

        print(f"Reading file: {args.file}")
        text = read_file(args.file)
        if not text.strip():
            print("[ERROR] Could not extract text from the file.", file=sys.stderr)
            sys.exit(1)
        print(f"Extracted {len(text):,} characters.\n")
    else:
        text = args.text

    # Run pipeline
    print("Starting Author Assist pipeline…")
    print("  Step 1/3 — Manager: analysing article…")

    from core.pipeline import run_pipeline  # noqa: PLC0415

    try:
        output = run_pipeline(text)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Save output
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Step 3/3 — Output saved to: {out_path.resolve()}")

    # Print summary
    _print_results(output, args.verbose)


if __name__ == "__main__":
    main()
