#!/usr/bin/env python3
"""Build the memory-search corpus for the arena baseline.

Reads the same contact selection IMPersona used for training, filters to
contacts whose `dataset.json` actually exists on disk (= truly included
in training, not just selected at some point), parses each contact's
raw iMessage export into conversation chunks, and persists them as
`.memory_corpus/conversations.json`.

The output is consumed by `IMPersona2.memory_search` and the arena
search tool — one conversation per chunk, tagged with date + contact
name only (no LLM extraction).

Usage:
    python scripts/build_memory_corpus.py
    python scripts/build_memory_corpus.py --output /tmp/corpus.json
    python scripts/build_memory_corpus.py --min-messages 4
"""
import argparse
import sys
from pathlib import Path

# Make IMPersona2 + pipeline importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from IMPersona2.experiment_config import ParsingConfig
from IMPersona2.memory_search import (
    DEFAULT_MAX_CHARS_PER_CHUNK,
    DEFAULT_MIN_CHARS_PER_CHUNK,
    build_conversation_corpus,
    corpus_summary,
    write_corpus,
)


# Input resolution lives in pipeline.py (shared with run.py --step memory).


def main():
    parser = argparse.ArgumentParser(
        description="Build the memory-search corpus from training-included contacts.",
    )
    parser.add_argument(
        "--output", type=str,
        default=".memory_corpus/conversations.json",
        help="Where to write the corpus JSON (default: .memory_corpus/conversations.json)",
    )
    parser.add_argument(
        "--min-messages", type=int, default=2,
        help="Skip conversations shorter than this many messages (default: 2)",
    )
    parser.add_argument(
        "--max-gap-hours", type=float, default=None,
        help="Override ParsingConfig.max_gap_hours (default: pipeline standard, 48h)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK,
        help=(
            f"Upper bound on individual chunk text length (default: "
            f"{DEFAULT_MAX_CHARS_PER_CHUNK}, ≈ {DEFAULT_MAX_CHARS_PER_CHUNK // 4} "
            f"tokens). Conversations exceeding this are recursively split at "
            f"their largest internal time gap. Set 0 to disable subchunking. "
            f"Any chunk still over this after subchunking is dropped."
        ),
    )
    parser.add_argument(
        "--min-chars", type=int, default=DEFAULT_MIN_CHARS_PER_CHUNK,
        help=(
            f"Lower bound on chunk text length (default: "
            f"{DEFAULT_MIN_CHARS_PER_CHUNK}). Short bursts below this are "
            f"dropped — they have little retrieval signal. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-file progress",
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════")
    print("║ Building memory-search corpus")
    print("╠══════════════════════════════════════════")

    from pipeline import resolve_memory_corpus_inputs
    included, excluded, contacts_vcf = resolve_memory_corpus_inputs()

    print(f"║ Contacts included (trained): {len(included)}")
    if excluded:
        print(f"║ Contacts excluded (selected but no dataset.json): {len(excluded)}")
        for f in excluded[:5]:
            print(f"║   - {f}")
        if len(excluded) > 5:
            print(f"║   ... and {len(excluded) - 5} more")
    print(f"║ Contacts vCard: {contacts_vcf or '(none — using export labels as-is)'}")
    print(f"║ Output: {args.output}")
    print("╚══════════════════════════════════════════\n")

    if not included:
        print("ERROR: No included contacts found. Did you run the train step?")
        sys.exit(1)

    pc = ParsingConfig()
    if args.max_gap_hours is not None:
        pc.max_gap_hours = args.max_gap_hours

    chunks = build_conversation_corpus(
        input_files=included,
        contacts_vcf=contacts_vcf,
        parsing_config=pc,
        min_messages=args.min_messages,
        max_chars_per_chunk=args.max_chars,
        min_chars_per_chunk=args.min_chars,
        verbose=not args.quiet,
    )

    output_path = Path(args.output)
    write_corpus(chunks, output_path)

    summary = corpus_summary(chunks)
    print(f"\n╔══ Corpus summary ══")
    print(f"║ Total chunks:     {summary['total_chunks']:,}")
    print(f"║ Total messages:   {summary['total_messages']:,}")
    print(f"║ Unique contacts:  {summary['contacts']}")
    print(f"║ Avg msgs/chunk:   {summary.get('messages_per_chunk_avg')}")
    print(f"║ Date range:       {summary['date_range'][0]} → {summary['date_range'][1]}")
    print(f"║ File:             {output_path.resolve()}")
    print(f"║ File size:        {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("╚════════════════════")


if __name__ == "__main__":
    main()
