#!/usr/bin/env python3
"""
Post-process a LlamaFactory training dataset to expose prior-turn
chain-of-thought in each example's conversation history.

Motivation
----------
IMPersona training examples currently show the conversation history with
bare "Me" messages (no <think> blocks). At inference, the model generates
``<think>cot</think>response`` but the next turn's history only contains
the CoT-stripped response — so every turn's CoT is produced stateless,
with no access to prior internal reasoning, which can make the reasoning
inconsistent from one turn to the next.

This script rewrites each training example's ``input`` field so that prior
"Me" lines look like:

    [N] [ts] Me: <think>cot</think>content

That matches the model's own output format, so when eval.py is run with
``--include_prior_cot`` the input distribution at inference matches what
was seen during training.

The script is a pure post-processor: it reads an existing training JSON
file and writes a new one. It never modifies the input in place.

Usage
-----
    # Transform the combined training_data.json
    python -m IMPersona2.add_prior_cot \\
        --input training_data.json \\
        --output training_data_with_prior_cot.json

    # Preview stats without writing
    python -m IMPersona2.add_prior_cot --input training_data.json --dry-run

    # Allow overwriting an existing output
    python -m IMPersona2.add_prior_cot \\
        --input training_data.json \\
        --output training_data_with_prior_cot.json \\
        --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Training history lines look like:
#   [3] [2026-01-01 12:00:05] Me: actual content here
# Timestamps from generate_dataset.py use strftime("%Y-%m-%d %H:%M:%S").
_ME_LINE_RE = re.compile(
    r"^(?P<prefix>\[(?P<num>\d+)\] \[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] Me: )(?P<content>.*)$"
)


# Matches the leading <think>...</think> in an example's output field,
# capturing both the CoT and the response that follows.
_OUTPUT_COT_AND_RESPONSE_RE = re.compile(r"^<think>(.*?)</think>(.*)$", re.DOTALL)

# <reply:N> numbers are remapped based on each example's history truncation
# offset — so the SAME underlying Me message can appear as
# "<reply:5>hello" in one example's output and "<reply:3>hello" in another
# example's history. We normalize the number away for content-keyed lookup.
_REPLY_TAG_RE = re.compile(r"<reply:\d+>")


def _normalize_content_for_key(s: str) -> str:
    """Normalize a Me message body for use as part of the compound map key.

    Strips the number from ``<reply:N>`` tags (see module-level note above)
    and trims whitespace. Everything else — <msg> separators, punctuation,
    case — is preserved, so the key remains tight enough to distinguish
    two different Me bursts that happen to share a second-precision
    timestamp.
    """
    return _REPLY_TAG_RE.sub("<reply:>", s).strip()


def _extract_example_ts_cot_response(ex: Dict) -> Tuple[str, str, str]:
    """Extract (timestamp, cot, response) for a single example.

    The combined training_data.json (LlamaFactory alpaca format) only
    preserves ``instruction``, ``input``, ``output``. Per-example metadata
    like ``timestamp`` and ``metadata.chain_of_thought`` are dropped by
    the combine step. But the information we need is still recoverable:

    - The example's timestamp is on the trailing ``[N] [ts] Me: ``
      generation-prompt line in ``input``.
    - The example's CoT and response are the ``<think>...</think>`` prefix
      and the body that follows in ``output`` (the <think> prefix is
      always present after generate_dataset.py's step 5).

    Any missing field is returned as an empty string.
    """
    # Timestamp: scan backwards for the last Me line.
    ts = ""
    for line in reversed(ex.get("input", "").split("\n")):
        m = _ME_LINE_RE.match(line)
        if m:
            ts = m.group("ts")
            break

    # CoT + response from output.
    cot = ""
    response = ""
    m = _OUTPUT_COT_AND_RESPONSE_RE.match(ex.get("output", "") or "")
    if m:
        cot = m.group(1).strip()
        response = m.group(2)

    return ts, cot, response


def build_cot_map(
    examples: List[Dict],
) -> Tuple[Dict[Tuple[str, str], str], set, Dict[str, int]]:
    """Build a (timestamp, normalized_content) -> CoT lookup from all examples.

    Prefers metadata fields when present (per-contact dataset.json files
    keep them), but falls back to extracting timestamp from the
    ``input`` field's trailing prompt line and the CoT + response from the
    ``<think>...</think>`` block at the start of ``output`` (combined
    LlamaFactory-format training_data.json).

    The map is keyed on **(timestamp, normalized_content)** rather than
    timestamp alone so we disambiguate the rare case where two different
    Me bursts share a second-precision timestamp. Without the content
    component, later bursts would silently overwrite earlier ones in the
    map, causing their history references to be injected with a different
    burst's CoT.

    Even the compound key leaves a small number of genuine duplicates —
    cases where the same text was sent twice in the same second and the
    annotator produced different CoTs for each occurrence. Those can't be
    disambiguated from a history reference (identical ts + content), so
    we:

    1. Drop the colliding keys entirely (no injection into history
       references — safer than choosing one arbitrarily).
    2. Return the set of source-example indices that participated in
       any collision, so ``process_examples`` can exclude them from the
       output dataset rather than keep slightly ambiguous examples.

    Returns (cot_map, collision_source_indices, diagnostic_stats).
    """
    # Phase 1: collect all (key -> list of (cot, source_idx)) candidates.
    candidates: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
    timestamp_only_collisions = 0
    missing_cot = 0
    missing_ts = 0
    missing_response = 0
    skip_cot = 0
    seen_timestamps: Dict[str, str] = {}

    for i, ex in enumerate(examples):
        ts = ex.get("timestamp", "") or ""
        meta = ex.get("metadata") or {}
        cot = meta.get("chain_of_thought", "") or ""
        _ts_parsed, _cot_parsed, response = _extract_example_ts_cot_response(ex)
        ts = ts or _ts_parsed
        cot = cot or _cot_parsed

        if not ts:
            missing_ts += 1
            continue
        if not cot:
            missing_cot += 1
            continue
        if not response:
            missing_response += 1
            continue
        if cot.startswith("[SKIP]") or cot.startswith("[COT GENERATION FAILED"):
            skip_cot += 1
            continue

        norm = _normalize_content_for_key(response)
        key = (ts, norm)

        # Diagnostics: count ts-only collisions so we can show that
        # compound keys are doing their disambiguation work.
        if ts in seen_timestamps and seen_timestamps[ts] != norm:
            timestamp_only_collisions += 1
        seen_timestamps[ts] = norm

        candidates.setdefault(key, []).append((cot, i))

    # Phase 2: resolve. Distinct-CoT collisions → drop and mark for exclusion.
    cot_map: Dict[Tuple[str, str], str] = {}
    collision_source_indices: set = set()
    compound_key_collisions = 0
    collision_source_count = 0

    for key, entries in candidates.items():
        distinct_cots = set(c for c, _ in entries)
        if len(distinct_cots) > 1:
            compound_key_collisions += 1
            for _, idx in entries:
                collision_source_indices.add(idx)
                collision_source_count += 1
        else:
            cot_map[key] = entries[0][0]

    stats = {
        "cot_map_size": len(cot_map),
        "compound_key_collisions": compound_key_collisions,
        "collision_source_examples_excluded": collision_source_count,
        "timestamp_only_collisions": timestamp_only_collisions,
        "missing_timestamp": missing_ts,
        "missing_cot": missing_cot,
        "missing_response": missing_response,
        "skip_or_failed_cot": skip_cot,
    }
    return cot_map, collision_source_indices, stats


def rewrite_input_with_prior_cots(
    input_text: str,
    cot_map: Dict[Tuple[str, str], str],
) -> Tuple[str, int, int]:
    """Rewrite a single example's ``input`` to include prior Me CoTs.

    Leaves the trailing empty "Me: " generation prompt alone.

    Returns (new_input_text, num_injections, num_missing_lookups).
    ``num_missing_lookups`` counts non-empty "Me" history lines where the
    CoT was not found in the map (either because it was a [SKIP] example,
    the dataset was partially assembled, or — rarely — the compound key
    landed on a timestamp+content pair with no corresponding source).
    """
    lines = input_text.split("\n")
    num_injected = 0
    num_missing = 0

    for i, line in enumerate(lines):
        m = _ME_LINE_RE.match(line)
        if not m:
            continue
        content = m.group("content")
        # Empty content = the trailing "[N] [ts] Me: " generation prompt.
        # It's also possible (though unusual) for an intermediate Me line
        # to be empty after stripping; either way, nothing to decorate.
        if not content.strip():
            continue

        ts = m.group("ts")
        key = (ts, _normalize_content_for_key(content))
        cot = cot_map.get(key)
        if not cot:
            num_missing += 1
            continue

        new_line = f"{m.group('prefix')}<think>{cot}</think>{content}"
        lines[i] = new_line
        num_injected += 1

    return "\n".join(lines), num_injected, num_missing


def process_examples(examples: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Transform all examples. Returns (new_examples, stats).

    Source examples involved in a compound-key collision are excluded from
    the output by design: dropping a handful of ambiguous examples costs
    little and keeps the training data clean.
    """
    cot_map, collision_indices, map_stats = build_cot_map(examples)

    new_examples: List[Dict] = []
    examples_changed = 0
    total_injections = 0
    total_missing_lookups = 0
    examples_excluded = 0

    for i, ex in enumerate(examples):
        if i in collision_indices:
            examples_excluded += 1
            continue

        new_ex = dict(ex)
        new_input, n_injected, n_missing = rewrite_input_with_prior_cots(
            ex.get("input", ""), cot_map
        )
        new_ex["input"] = new_input
        if n_injected > 0:
            examples_changed += 1
            total_injections += n_injected
        total_missing_lookups += n_missing
        new_examples.append(new_ex)

    stats = {
        "total_input_examples": len(examples),
        **map_stats,
        "total_output_examples": len(new_examples),
        "examples_excluded_due_to_collisions": examples_excluded,
        "examples_changed": examples_changed,
        "total_cot_injections": total_injections,
        "total_missing_lookups": total_missing_lookups,
    }
    return new_examples, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inject prior-turn chain-of-thought into IMPersona training "
            "examples' conversation history fields."
        ),
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to existing training dataset JSON (e.g. training_data.json).",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output JSON path (required unless --dry-run). "
            "Must NOT collide with --input; use --force to overwrite an existing "
            "output file."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print stats only; do not write a new file.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Allow overwriting --output if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        if not args.output:
            print("ERROR: --output is required unless --dry-run is set.", file=sys.stderr)
            sys.exit(1)
        output_path = Path(args.output).resolve()
        if output_path == input_path:
            print(
                "ERROR: --output must be a different path from --input "
                "(this script never modifies the input in place).",
                file=sys.stderr,
            )
            sys.exit(1)
        if output_path.exists() and not args.force:
            print(
                f"ERROR: output path already exists: {output_path}\n"
                "       Pass --force to overwrite, or choose a different path.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        output_path = None

    print(f"Loading {input_path} ...", flush=True)
    with open(input_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    if not isinstance(examples, list):
        print("ERROR: input must be a JSON list of examples.", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(examples):,} examples")

    print("Processing...", flush=True)
    new_examples, stats = process_examples(examples)

    print()
    print("Stats:")
    width = max(len(k) for k in stats)
    for k, v in stats.items():
        print(f"  {k:<{width}}  {v:,}" if isinstance(v, int) else f"  {k:<{width}}  {v}")

    if args.dry_run:
        print("\nDry run — not writing output.")
        return

    assert output_path is not None
    print(f"\nWriting {output_path} ...", flush=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_examples, f, ensure_ascii=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {len(new_examples):,} examples ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
