#!/usr/bin/env python3
"""Memory-search corpus builder for the arena baseline.

Produces a per-conversation chunk list from the same raw iMessage files
the IMPersona was trained on. This is the data layer for the agentic
search-memory baseline in the arena.

Why this exists in a separate module from `generate_dataset.py`: the
training pipeline explodes each conversation into one row per persona
response, which is great for SFT but produces heavily-overlapping
chunks for retrieval (the same conversation prefix appears N times,
once per persona reply). For retrieval we want **one chunk per
conversation session**, which is the natural unit `parse_imessage_file`
already returns BEFORE the per-turn explosion.

This module just re-uses the existing parsing functions
(parse_imessage_file + _combine_consecutive) directly, applies no LLM
calls, and emits a flat list of conversation chunks tagged with date
and contact name only. The intent: minimal-bias retrieval over the
exact same source data the IMPersona was trained on.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from IMPersona2.experiment_config import ParsingConfig
from IMPersona2.model_config import model_for
from IMPersona2.generate_dataset import (
    _build_contacts_lookup,
    _combine_consecutive,
    parse_imessage_file,
)


# ────────────────────────────────────────────────────────────
# Chunk shape
# ────────────────────────────────────────────────────────────
# Each chunk is one full back-and-forth between the persona and one
# contact within a single contiguous time window (≤ ParsingConfig.max_gap_hours
# between adjacent messages).
#
#     {
#         "contact":      str,    # e.g. "Sarah Chen"
#         "date_start":   str,    # "YYYY-MM-DD"
#         "date_end":     str,    # "YYYY-MM-DD"
#         "num_messages": int,
#         "text":         str,    # canonical transcript used for indexing
#                                 # AND returned to the model
#     }
#
# `text` is what gets embedded and what is shown to the model when
# retrieved. Keeping a single canonical representation avoids
# inconsistencies between what's searched and what the model sees.

PERSONA_LABEL = "Me"  # parse_imessage_file uses "Me" for the persona's own messages

# Default subchunking threshold, ~1,250 tokens: short enough to read at a
# glance, and K=5 retrievals fit easily in the model's context window
# (~25k chars / ~6k tokens for the retrieved memory, leaving plenty of room
# for the actual conversation).
DEFAULT_MAX_CHARS_PER_CHUNK = 5000

# Drop chunks below this size — single short bursts ("ok", "lol",
# "running late") carry little retrieval signal. The cutoff is generous
# enough that any chunk with even one substantive message survives.
DEFAULT_MIN_CHARS_PER_CHUNK = 500


def _estimate_chars(messages: List[Dict]) -> int:
    """Approximate the eventual chunk-text length without formatting it
    every time. Matches the format produced by ``conversation_to_chunk``:
    ``"<speaker>: <content>\\n"`` per message, plus a one-line header.
    """
    # +2 for ": ", +1 for newline; ignore empty-content messages
    return sum(
        len(m.get("speaker", "")) + len((m.get("content") or "").strip()) + 3
        for m in messages
        if (m.get("content") or "").strip()
    )


def _position_of_largest_internal_gap(messages: List[Dict]) -> Optional[int]:
    """Return the message-list index at which the largest internal time
    gap occurs (i.e. ``messages[idx]`` starts a new sub-segment).

    Returns ``None`` if the list is too short to have any internal gaps.
    """
    if len(messages) <= 1:
        return None
    fmt = "%Y-%m-%d %H:%M:%S"
    best_idx, best_gap = -1, -1.0
    for i in range(1, len(messages)):
        try:
            t0 = datetime.strptime(messages[i - 1]["timestamp"], fmt)
            t1 = datetime.strptime(messages[i]["timestamp"], fmt)
        except (KeyError, ValueError):
            continue
        gap = (t1 - t0).total_seconds()
        if gap > best_gap:
            best_gap, best_idx = gap, i
    return best_idx if best_idx > 0 else None


def _subchunk_by_largest_gap(
    messages: List[Dict],
    max_chars: int,
) -> List[List[Dict]]:
    """Recursively split a conversation at the largest internal time gap
    until every resulting subchunk fits under ``max_chars``.

    Design notes:
      - No gap-magnitude threshold. The original chunking already split
        on 48h+ gaps; anything left inside a chunk is by definition
        within the same "session". The largest *remaining* internal gap
        is still the most natural sub-break (someone went to sleep, took
        a meeting, etc.).
      - Chunks with ≤2 messages are returned as-is even if oversize: a
        two-message chunk can only split into two single-message chunks,
        which fail the downstream `min_messages` filter and would just
        be dropped — better to keep the pair intact.
      - If no internal gaps exist at all (e.g. all messages share the
        same timestamp — very rare), fall back to halving by message
        count.
    """
    if _estimate_chars(messages) <= max_chars or len(messages) <= 2:
        return [messages]

    gap_idx = _position_of_largest_internal_gap(messages)
    if gap_idx is None:
        gap_idx = max(1, len(messages) // 2)

    left = _subchunk_by_largest_gap(messages[:gap_idx], max_chars)
    right = _subchunk_by_largest_gap(messages[gap_idx:], max_chars)
    return left + right


def _conversation_date_range(messages: List[Dict]) -> Tuple[str, str]:
    """Return (first_date, last_date) as YYYY-MM-DD strings."""
    if not messages:
        return ("", "")
    fmt = "%Y-%m-%d %H:%M:%S"
    first = datetime.strptime(messages[0]["timestamp"], fmt)
    last = datetime.strptime(messages[-1]["timestamp"], fmt)
    return (first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"))


def _detect_contact_name(
    conversations: List[List[Dict]],
    fallback: str,
) -> str:
    """Pick the most-frequent non-persona speaker across all conversations in
    a single contact's file.

    The IMPersona pipeline resolves phone-number speakers to vCard names
    inside the message dicts (via `_build_contacts_lookup`), but the file
    name itself is often still the unresolved phone number
    (e.g. ``imessage_export_mac/+15555550123.txt``). We prefer the
    resolved name inside the messages whenever one shows up; otherwise we
    fall back to the filename stem.
    """
    counts: Dict[str, int] = {}
    for conv in conversations:
        for msg in conv:
            spk = msg.get("speaker", "")
            if spk and spk != PERSONA_LABEL:
                counts[spk] = counts.get(spk, 0) + 1
    if not counts:
        return fallback
    # The most-frequent non-persona speaker wins
    return max(counts, key=counts.get)


def conversation_to_chunk(
    messages: List[Dict],
    contact_name: str,
) -> Dict:
    """Convert a list of message dicts to a retrievable chunk dict."""
    date_start, date_end = _conversation_date_range(messages)
    date_label = date_start if date_start == date_end else f"{date_start} → {date_end}"
    header = f"[{date_label} · {contact_name}]"

    lines = [header]
    for m in messages:
        spk = m.get("speaker", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{spk}: {content}")

    return {
        "contact": contact_name,
        "date_start": date_start,
        "date_end": date_end,
        "num_messages": len(messages),
        "text": "\n".join(lines),
    }


def build_conversation_corpus(
    input_files: List[str],
    contacts_vcf: Optional[str] = None,
    parsing_config: Optional[ParsingConfig] = None,
    min_messages: int = 2,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    min_chars_per_chunk: int = DEFAULT_MIN_CHARS_PER_CHUNK,
    verbose: bool = True,
) -> List[Dict]:
    """Build a flat list of conversation chunks for retrieval.

    Args:
        input_files: Paths to raw iMessage *.txt exports. Same kind of
            files listed in PipelineConfig.input_files. Only these will
            be parsed — callers should filter to "contacts that were
            actually included in training" upstream (see
            scripts/build_memory_corpus.py).
        contacts_vcf: Optional path to a vCard file for phone→name
            resolution. Falls back to whatever labels the export already
            contains.
        parsing_config: Defaults to IMPersona's standard ParsingConfig
            (48h max gap, combine_consecutive=True, 4h max combine gap)
            so the chunking matches what was used during training.
        min_messages: Skip conversations shorter than this. Two-message
            exchanges like "I'll call you back" / "ok" carry little
            retrieval signal.
        max_chars_per_chunk: Upper bound on individual chunk text length.
            Conversations longer than this are recursively split at
            their largest remaining internal time gap (see
            ``_subchunk_by_largest_gap``). The default (5,000 chars ≈
            1,250 tokens) leaves room for K=5 retrievals plus the
            actual conversation in a 32k context. Set to 0 to
            disable subchunking entirely. After subchunking, any chunk
            still above this ceiling (typically 2-message sessions with
            one huge message body that can't be split without
            single-message fragments) is dropped as low quality.
        min_chars_per_chunk: Lower bound on chunk text length. Chunks
            below this are dropped — short bursts like "ok" / "running
            late" / "lol" don't carry useful retrieval signal. Default 500
            chars (~125 tokens); set to 0 to disable.
        verbose: Print per-file progress.

    Returns:
        Flat list of chunk dicts.
    """
    parsing_config = parsing_config or ParsingConfig()
    contacts = _build_contacts_lookup(contacts_vcf) if contacts_vcf else {}

    all_chunks: List[Dict] = []
    n_files_processed = 0
    n_files_skipped = 0
    n_raw_total = 0       # pre-subchunk session count, for reporting
    n_dropped_short = 0   # chunks below min_chars_per_chunk
    n_dropped_long = 0    # chunks still above max_chars_per_chunk after subchunk

    for path in input_files:
        if not os.path.exists(path):
            if verbose:
                print(f"  ⚠ skip {path} (file not found)")
            n_files_skipped += 1
            continue

        try:
            conversations = parse_imessage_file(path, parsing_config, contacts)
        except Exception as e:
            if verbose:
                print(f"  ⚠ skip {path} ({type(e).__name__}: {e})")
            n_files_skipped += 1
            continue

        if parsing_config.combine_consecutive:
            conversations = [
                _combine_consecutive(
                    c,
                    parsing_config.combine_delimiter,
                    max_gap_hours=parsing_config.max_combine_gap_hours,
                )
                for c in conversations
            ]

        # Resolve the contact name from the (already-resolved) speaker
        # labels inside the parsed messages, with the filename stem as a
        # fallback. This avoids exposing raw phone numbers when the
        # iMessage export file is phone-named but the vCard has resolved
        # the speakers internally.
        filename_stem = Path(path).stem
        contact_name = _detect_contact_name(conversations, fallback=filename_stem)

        n_kept = 0
        n_subsplits = 0
        for conv in conversations:
            if len(conv) < min_messages:
                continue
            n_raw_total += 1

            # Apply gap-based subchunking if the parent session is too
            # long. Sub-chunks inherit the contact name; each derives
            # its own date range from the messages it actually contains.
            if max_chars_per_chunk > 0:
                sub_msg_lists = _subchunk_by_largest_gap(conv, max_chars_per_chunk)
            else:
                sub_msg_lists = [conv]
            if len(sub_msg_lists) > 1:
                n_subsplits += 1

            for sub in sub_msg_lists:
                if len(sub) < min_messages:
                    # A sub-fragment of <min_messages can appear (e.g. a
                    # 3-msg conv splits into [1, 2]); drop the tiny one.
                    continue
                chunk = conversation_to_chunk(sub, contact_name)
                chunk_len = len(chunk["text"])
                if chunk_len < min_chars_per_chunk:
                    n_dropped_short += 1
                    continue
                if max_chars_per_chunk > 0 and chunk_len > max_chars_per_chunk:
                    # Sub-chunking couldn't get below the ceiling (e.g. a
                    # 2-msg session with one giant body). Drop as low quality.
                    n_dropped_long += 1
                    continue
                all_chunks.append(chunk)
                n_kept += 1

        n_files_processed += 1
        if verbose:
            display = contact_name
            if contact_name != filename_stem:
                display = f"{contact_name} ({filename_stem})"
            split_note = f", {n_subsplits} sessions split" if n_subsplits else ""
            print(f"  ✓ {display}: {n_kept} chunks{split_note}")

    if verbose:
        print(f"\nCorpus build complete: {len(all_chunks)} chunks from "
              f"{n_files_processed} files ({n_files_skipped} skipped)")
        print(f"  Pre-subchunk sessions: {n_raw_total}")
        if n_dropped_short:
            print(f"  Dropped (<{min_chars_per_chunk} chars):  {n_dropped_short}")
        if n_dropped_long:
            print(f"  Dropped (>{max_chars_per_chunk} chars after subchunk): {n_dropped_long}")
    return all_chunks


def write_corpus(chunks: List[Dict], output_path: Path) -> None:
    """Persist a corpus to JSON at ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_corpus(corpus_path: Path) -> List[Dict]:
    """Load a previously-built corpus from JSON."""
    return json.loads(corpus_path.read_text(encoding="utf-8"))


# ────────────────────────────────────────────────────────────
# Dense retrieval — MemoryStore + index build
# ────────────────────────────────────────────────────────────
# Single-shot embedding of every chunk (model: `embeddings` in configs/models.yaml)
# into a numpy array on disk. Search = cosine over that array.
#
# Why no FAISS / Chroma / etc.: the corpus is ≤10k chunks. Numpy dot
# product over a (10k, 3072) array runs in <1ms locally. Anything more
# is overkill.

DEFAULT_EMBEDDING_MODEL = model_for("embeddings")
DEFAULT_EMBEDDING_DIM = 3072


def _embed_texts(texts: List[str], model: str, batch_size: int = 100):
    """Embed a list of texts via the OpenAI API in batches.

    Returns a numpy array of shape (len(texts), embedding_dim), L2-normalized
    row-wise so cosine similarity = dot product at query time.
    """
    import numpy as np
    from openai import OpenAI

    client = OpenAI()
    vectors: List = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        for d in resp.data:
            v = np.asarray(d.embedding, dtype=np.float32)
            # L2 normalize so we can use a plain dot product for cosine
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vectors.append(v)
        print(f"  embedded {min(i + batch_size, n)}/{n}", flush=True)
    return np.stack(vectors).astype(np.float32)


def build_dense_index(
    corpus_path: Path,
    index_dir: Path,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 100,
) -> Dict:
    """Embed every chunk in ``corpus_path`` and persist:
        index_dir/embeddings.npy  — (N, D) L2-normalized float32 array
        index_dir/manifest.json   — model name, dim, num_chunks, corpus_hash

    The corpus_hash lets MemoryStore detect staleness — if the corpus
    changes (e.g. the user re-runs build_memory_corpus), the index is
    no longer valid for it and the user is warned.

    Returns a small summary dict (n_chunks, total_tokens_estimate, etc.).
    """
    import hashlib
    import json
    from datetime import datetime
    import numpy as np

    corpus_bytes = corpus_path.read_bytes()
    corpus_hash = hashlib.sha256(corpus_bytes).hexdigest()[:16]
    chunks = json.loads(corpus_bytes.decode("utf-8"))
    if not chunks:
        raise RuntimeError(f"Empty corpus at {corpus_path}")

    print(f"Embedding {len(chunks)} chunks via {model}...", flush=True)
    texts = [c["text"] for c in chunks]
    embeddings = _embed_texts(texts, model=model, batch_size=batch_size)

    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "manifest.json"

    np.save(embeddings_path, embeddings)
    manifest = {
        "embedding_model": model,
        "embedding_dim": int(embeddings.shape[1]),
        "num_chunks": int(embeddings.shape[0]),
        "corpus_hash": corpus_hash,
        "corpus_path": str(corpus_path),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_chars = sum(len(t) for t in texts)
    print(
        f"Index built: {embeddings.shape[0]} chunks × {embeddings.shape[1]}-dim "
        f"({embeddings.nbytes / 1024 / 1024:.1f} MB)",
        flush=True,
    )
    return {**manifest, "total_chars": total_chars}


class MemoryStore:
    """Dense-retrieval interface over the conversation corpus.

    Initialization loads the corpus + pre-computed embedding index into
    memory. ``.search(query, k)`` embeds the query at call time and
    returns the top-K chunks by cosine similarity. Each result includes
    the original chunk dict plus a ``score`` field.

    Detects index staleness via a corpus_hash check on load and raises
    early if the on-disk embeddings were built against a different
    corpus.
    """

    def __init__(
        self,
        corpus_path: Path,
        index_dir: Path,
        embedding_model: Optional[str] = None,
    ):
        import hashlib
        import json
        import numpy as np

        corpus_path = Path(corpus_path)
        index_dir = Path(index_dir)
        manifest_path = index_dir / "manifest.json"
        embeddings_path = index_dir / "embeddings.npy"

        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")
        if not manifest_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(
                f"Index not built. Run: python scripts/build_memory_index.py"
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()[:16]
        if corpus_hash != manifest.get("corpus_hash"):
            raise RuntimeError(
                f"Index is stale: corpus changed since last index build. "
                f"Re-run: python scripts/build_memory_index.py"
            )

        self.chunks: List[Dict] = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.embeddings = np.load(embeddings_path)
        if len(self.chunks) != len(self.embeddings):
            raise RuntimeError(
                f"Index/corpus length mismatch: "
                f"{len(self.chunks)} chunks vs {len(self.embeddings)} embeddings"
            )

        self.embedding_model = embedding_model or manifest.get(
            "embedding_model", DEFAULT_EMBEDDING_MODEL
        )
        self.manifest = manifest

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Return the top-K chunks most similar to ``query``.

        Result dicts are the original chunk dicts with an added
        ``score`` field (cosine similarity, range roughly [-1, 1] but
        in practice [0, 1] for natural-language queries).
        """
        import numpy as np
        from openai import OpenAI

        if not query.strip():
            return []

        client = OpenAI()
        resp = client.embeddings.create(model=self.embedding_model, input=query)
        q = np.asarray(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        scores = self.embeddings @ q  # (N,)
        if k >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, k)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

        out: List[Dict] = []
        for i in top_idx[:k]:
            chunk = dict(self.chunks[int(i)])
            chunk["score"] = float(scores[int(i)])
            out.append(chunk)
        return out


def corpus_summary(chunks: List[Dict]) -> Dict:
    """Quick stats useful for verifying a freshly-built corpus."""
    if not chunks:
        return {
            "total_chunks": 0, "total_messages": 0,
            "contacts": 0, "date_range": (None, None),
        }
    contacts = {c["contact"] for c in chunks}
    total_msgs = sum(c["num_messages"] for c in chunks)
    dates = [c["date_start"] for c in chunks if c["date_start"]]
    return {
        "total_chunks": len(chunks),
        "total_messages": total_msgs,
        "contacts": len(contacts),
        "messages_per_chunk_avg": round(total_msgs / len(chunks), 1),
        "date_range": (min(dates), max(dates)) if dates else (None, None),
    }
