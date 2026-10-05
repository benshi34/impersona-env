#!/usr/bin/env python3
"""One-time embedding-index build for the memory-search corpus.

Reads `.memory_corpus/conversations.json` (the per-conversation chunks
produced by build_memory_corpus.py) and writes:
    .memory_corpus/embeddings.npy   — L2-normalized float32 array
    .memory_corpus/manifest.json    — model name, corpus hash, etc.

Uses the `embeddings` model from configs/models.yaml (OpenAI; text-embedding-3-large by default). ~$0.30 for a 3k-chunk
corpus and runs in ~30s. The index is keyed to a hash of the corpus
file, so MemoryStore refuses to use a stale index if the corpus
changes underneath it.

Usage:
    python scripts/build_memory_index.py
    python scripts/build_memory_index.py --corpus .memory_corpus/conversations.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from IMPersona2.memory_search import (
    DEFAULT_EMBEDDING_MODEL,
    build_dense_index,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build the dense-retrieval index for the memory corpus.",
    )
    parser.add_argument(
        "--corpus", type=str,
        default=".memory_corpus/conversations.json",
        help="Path to the corpus JSON (default: .memory_corpus/conversations.json)",
    )
    parser.add_argument(
        "--index-dir", type=str,
        default=".memory_corpus",
        help="Directory for the index files (default: .memory_corpus)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Embedding batch size (default: 100 chunks per OpenAI request)",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    index_dir = Path(args.index_dir)
    if not corpus_path.exists():
        print(f"ERROR: corpus not found at {corpus_path}")
        print(f"  Run `python scripts/build_memory_corpus.py` first.")
        sys.exit(1)

    # Lazy import dotenv so we pick up OPENAI_API_KEY from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    print("╔══════════════════════════════════════════")
    print("║ Building dense embedding index")
    print(f"║   Corpus:    {corpus_path}")
    print(f"║   Index dir: {index_dir}")
    print(f"║   Model:     {args.model}")
    print("╚══════════════════════════════════════════")

    summary = build_dense_index(
        corpus_path=corpus_path,
        index_dir=index_dir,
        model=args.model,
        batch_size=args.batch_size,
    )

    print()
    print(f"╔══ Index summary ══")
    print(f"║ Chunks:       {summary['num_chunks']:,}")
    print(f"║ Dim:          {summary['embedding_dim']}")
    print(f"║ Corpus hash:  {summary['corpus_hash']}")
    print(f"║ Built at:     {summary['built_at']}")
    print(f"║ Total chars:  {summary['total_chars']:,}")
    print(f"║ Files:")
    print(f"║   {index_dir / 'embeddings.npy'}")
    print(f"║   {index_dir / 'manifest.json'}")
    print(f"╚════════════════")


if __name__ == "__main__":
    main()
