#!/usr/bin/env python3
"""
Combine DPO datasets from multiple eval runs into a single file
ready for LlamaFactory DPO training.

Auto-discovers all dpo_dataset*.json files under eval_output/ and combines
them.

Usage:
    # Combine all DPO datasets from all runs (auto-discover):
    python -m IMPersona2.combine_dpo_datasets

    # Filter by rewrite model (substring of its name):
    python -m IMPersona2.combine_dpo_datasets --rewrite-model <name-substring>

    # Filter by run name pattern:
    python -m IMPersona2.combine_dpo_datasets --run-pattern "round2"

    # Filter by IMPersona version tag in the run name (e.g. "_v2_" runs only):
    python -m IMPersona2.combine_dpo_datasets --version v2

    # Custom output and register in LlamaFactory:
    python -m IMPersona2.combine_dpo_datasets \
        --name impersona_dpo_combined \
        --register

    # Keep single-turn alpaca format (not recommended):
    python -m IMPersona2.combine_dpo_datasets --single-turn

    # Dry run — show what would be combined:
    python -m IMPersona2.combine_dpo_datasets --dry-run
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from IMPersona2.prepare_dpo_dataset import (
    convert_to_sharegpt_dpo,
    deduplicate,
    register_dataset,
)


# ── Default paths (relative to this file's location) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMPERSONA_DIR = os.path.dirname(SCRIPT_DIR)  # repo root
WORKSPACE_DIR = os.path.dirname(IMPERSONA_DIR)  # parent of the repo (local LlamaFactory checkout)
DEFAULT_EVAL_OUTPUT_DIR = os.path.join(IMPERSONA_DIR, "eval_output")
DEFAULT_LLAMAFACTORY_DIR = os.path.join(WORKSPACE_DIR, "LlamaFactory")


def find_all_dpo_datasets(
    eval_output_dir: str,
    *,
    version: Optional[str] = None,
    run_pattern: Optional[str] = None,
    rewrite_model: Optional[str] = None,
    filename_pattern: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Discover all dpo_dataset*.json files under eval_output_dir.

    Returns a list of (filepath, metadata) tuples, where metadata is
    extracted from the first example in each file.

    Filters:
        version:        Only include runs whose folder name contains this
                        version string (e.g. "v2").
        run_pattern:    Only include runs whose folder name contains this
                        substring (e.g. "round2").
        rewrite_model:  Only include datasets where the _rewrite_model field
                        (or filename) contains this substring.
        filename_pattern: Only include files whose name matches this pattern
                        (e.g. "dpo_dataset_local.json"). Supports glob patterns.
    """
    pattern = os.path.join(eval_output_dir, "run_*", "dpo_dataset*.json")
    all_paths = sorted(glob.glob(pattern))

    results: List[Tuple[str, Dict[str, Any]]] = []

    for path in all_paths:
        run_dir = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)

        # Filter by version in run directory name
        if version:
            # Match a version tag like "v2" in "eval_v2_2026-01-01_12-00-00"
            if f"_{version}_" not in run_dir and not run_dir.endswith(f"_{version}"):
                continue

        # Filter by run name pattern
        if run_pattern:
            if run_pattern.lower() not in run_dir.lower():
                continue

        # Filter by filename pattern
        if filename_pattern:
            import fnmatch
            if not fnmatch.fnmatch(filename, filename_pattern):
                continue

        # Load metadata from first example
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ⚠️  Skipping {path}: {e}")
            continue

        if not data or not isinstance(data, list):
            print(f"  ⚠️  Skipping {path}: empty or not a list")
            continue

        first_example = data[0]
        metadata = {
            "count": len(data),
            "run_label": first_example.get("_run_label", "unknown"),
            "run_id": first_example.get("_run_id", "unknown"),
            "rewrite_model": first_example.get("_rewrite_model", "unknown"),
            "persona_name": first_example.get("_persona_name", "unknown"),
            "run_dir": run_dir,
            "filename": filename,
        }

        # Filter by rewrite model
        if rewrite_model:
            model_match = rewrite_model.lower()
            file_model = metadata["rewrite_model"].lower()
            file_name = filename.lower()
            if model_match not in file_model and model_match not in file_name:
                continue

        results.append((path, metadata))

    return results


def load_dpo_examples(path: str) -> List[Dict]:
    """Load a DPO dataset JSON file and return the list of examples."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_discovery_table(
    discovered: List[Tuple[str, Dict[str, Any]]],
) -> None:
    """Pretty-print the discovered datasets."""
    if not discovered:
        print("  No DPO datasets found.")
        return

    # Header
    print(f"  {'#':>3}  {'Run':40}  {'File':30}  {'Rewrite Model':25}  {'Count':>5}")
    print(f"  {'─'*3}  {'─'*40}  {'─'*30}  {'─'*25}  {'─'*5}")

    for i, (path, meta) in enumerate(discovered, 1):
        run = meta["run_dir"]
        if len(run) > 40:
            run = run[:37] + "..."
        fname = meta["filename"]
        if len(fname) > 30:
            fname = fname[:27] + "..."
        model = meta["rewrite_model"]
        if len(model) > 25:
            model = model[:22] + "..."
        count = meta["count"]
        print(f"  {i:>3}  {run:40}  {fname:30}  {model:25}  {count:>5}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine DPO datasets from multiple eval runs into a single "
            "file ready for LlamaFactory DPO training."
        )
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default=DEFAULT_EVAL_OUTPUT_DIR,
        help=f"Path to eval_output/ directory (default: {DEFAULT_EVAL_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="impersona_dpo_combined",
        help="Dataset name for LlamaFactory registration (default: impersona_dpo_combined)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: LlamaFactory/data/<name>.json)",
    )

    # ── Filtering ──
    filter_group = parser.add_argument_group("filtering")
    filter_group.add_argument(
        "--version",
        type=str,
        default=None,
        help=(
            "Only include runs matching this IMPersona version "
            "(e.g. 'v2'). Matches against the run folder name."
        ),
    )
    filter_group.add_argument(
        "--run-pattern",
        type=str,
        default=None,
        help=(
            "Only include runs whose folder name contains this substring "
            "(e.g. 'round2')."
        ),
    )
    filter_group.add_argument(
        "--rewrite-model",
        type=str,
        default=None,
        help=(
            "Only include datasets where the rewrite model or filename "
            "contains this substring."
        ),
    )
    filter_group.add_argument(
        "--filename-pattern",
        type=str,
        default=None,
        help=(
            "Only include files matching this glob pattern "
            "(e.g. 'dpo_dataset_local.json')."
        ),
    )
    filter_group.add_argument(
        "--exclude-runs",
        type=str,
        nargs="+",
        default=None,
        help="Run directory names (or substrings) to exclude.",
    )

    # ── Format ──
    format_group = parser.add_argument_group("format")
    format_group.add_argument(
        "--single-turn",
        action="store_true",
        help=(
            "Keep the single-turn alpaca format instead of converting to "
            "multi-turn sharegpt. Not recommended: the model won't see "
            "conversation context during DPO training."
        ),
    )
    format_group.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Keep _* metadata fields in the output (for debugging)",
    )

    # ── LlamaFactory ──
    lf_group = parser.add_argument_group("llamafactory")
    lf_group.add_argument(
        "--llamafactory-dir",
        type=str,
        default=DEFAULT_LLAMAFACTORY_DIR,
        help=f"Path to LlamaFactory directory (default: {DEFAULT_LLAMAFACTORY_DIR})",
    )
    lf_group.add_argument(
        "--register",
        action="store_true",
        help="Register the dataset in LlamaFactory's dataset_info.json",
    )
    lf_group.add_argument(
        "--no-register",
        action="store_true",
        help="Skip registering even when writing to LlamaFactory/data/",
    )

    # ── Misc ──
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be combined without writing any files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve output path
    llamafactory_data_dir = os.path.join(args.llamafactory_dir, "data")
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        output_path = os.path.join(llamafactory_data_dir, f"{args.name}.json")

    dataset_info_path = os.path.join(llamafactory_data_dir, "dataset_info.json")

    print(f"\n{'═' * 66}")
    print(f"  IMPersona DPO Dataset Combiner")
    print(f"{'═' * 66}")
    print(f"  Eval output dir: {args.eval_output_dir}")
    if args.version:
        print(f"  Version filter:  {args.version}")
    if args.run_pattern:
        print(f"  Run pattern:     {args.run_pattern}")
    if args.rewrite_model:
        print(f"  Rewrite model:   {args.rewrite_model}")
    if args.filename_pattern:
        print(f"  Filename pattern: {args.filename_pattern}")

    # ── Step 1: Discover DPO datasets ──
    print(f"\n🔍 Discovering DPO datasets...")
    discovered = find_all_dpo_datasets(
        args.eval_output_dir,
        version=args.version,
        run_pattern=args.run_pattern,
        rewrite_model=args.rewrite_model,
        filename_pattern=args.filename_pattern,
    )

    # Apply exclusion filter
    if args.exclude_runs and discovered:
        before = len(discovered)
        discovered = [
            (path, meta)
            for path, meta in discovered
            if not any(
                excl.lower() in meta["run_dir"].lower()
                for excl in args.exclude_runs
            )
        ]
        excluded = before - len(discovered)
        if excluded:
            print(f"  Excluded {excluded} dataset(s) by --exclude-runs")

    if not discovered:
        print("❌ No DPO datasets found matching the filters.")
        sys.exit(1)

    print(f"\n📋 Found {len(discovered)} DPO dataset file(s):\n")
    print_discovery_table(discovered)

    # ── Step 2: Load and combine ──
    print(f"\n📥 Loading and combining...")
    combined: List[Dict] = []
    stats: Dict[str, int] = {}

    for path, meta in discovered:
        examples = load_dpo_examples(path)
        combined.extend(examples)

        key = f"{meta['run_dir']}/{meta['filename']} ({meta['rewrite_model']})"
        stats[key] = len(examples)
        print(f"  ✅ {meta['run_dir']}/{meta['filename']}: {len(examples)} examples")

    if not combined:
        print("\n❌ No examples after loading!")
        sys.exit(1)

    # ── Step 3: Deduplicate ──
    before_dedup = len(combined)
    combined = deduplicate(combined)
    dupes = before_dedup - len(combined)

    print(f"\n📊 Combined: {len(combined)} examples", end="")
    if dupes > 0:
        print(f" ({dupes} duplicates removed)")
    else:
        print()

    if args.dry_run:
        print(f"\n🏃 Dry run — would write {len(combined)} examples to:")
        print(f"  {output_path}")
        print(f"\n  Sources:")
        for source, count in sorted(stats.items()):
            print(f"    {source}: {count}")
        print()
        return

    # ── Step 4: Format conversion ──
    has_chat_history = any(
        len(ex.get("_chat_messages_before", [])) > 0 for ex in combined
    )
    examples_with_history = sum(
        1 for ex in combined if len(ex.get("_chat_messages_before", [])) > 0
    )

    is_sharegpt = not args.single_turn
    if is_sharegpt:
        if has_chat_history:
            print(f"\n🔄 Converting to multi-turn sharegpt format...")
            print(f"  {examples_with_history}/{len(combined)} examples have conversation context")
            combined = convert_to_sharegpt_dpo(combined)
            print(f"  ✅ Converted {len(combined)} examples to sharegpt format")
        else:
            print(f"\n⚠️  No conversation context found in _chat_messages_before.")
            print(f"  Falling back to single-turn alpaca format.")
            is_sharegpt = False

    if not is_sharegpt:
        print(f"\n📋 Using single-turn alpaca format")
        clean_examples = []
        for ex in combined:
            clean: Dict[str, Any] = {
                "instruction": ex.get("instruction", ""),
                "input": ex.get("input", ""),
                "chosen": ex.get("chosen", ""),
                "rejected": ex.get("rejected", ""),
            }
            if args.keep_metadata:
                for key, value in ex.items():
                    if key.startswith("_"):
                        clean[key] = value
            clean_examples.append(clean)
        combined = clean_examples

    # ── Step 5: Write output ──
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"\n💾 Writing dataset...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  ✅ Saved {len(combined)} examples to {output_path}")

    # ── Step 6: Register in dataset_info.json ──
    should_register = args.register or (
        not args.no_register
        and os.path.isdir(llamafactory_data_dir)
        and output_path.startswith(os.path.abspath(llamafactory_data_dir))
    )

    if should_register:
        if not os.path.isfile(dataset_info_path):
            print(f"  ⚠️  dataset_info.json not found: {dataset_info_path}")
            print(f"  Skipping registration.")
        else:
            print(f"\n📝 Registering dataset...")
            output_filename = os.path.basename(output_path)
            register_dataset(
                dataset_info_path=dataset_info_path,
                name=args.name,
                filename=output_filename,
                is_sharegpt=is_sharegpt,
            )

    # ── Summary ──
    print(f"\n{'─' * 66}")
    print(f"  ✅ Done! Combined DPO dataset '{args.name}' is ready.")
    print(f"  Format:   {'sharegpt (multi-turn)' if is_sharegpt else 'alpaca (single-turn)'}")
    print(f"  Examples: {len(combined)}")
    print(f"  Sources:  {len(discovered)} dataset files from {len(set(m['run_dir'] for _, m in discovered))} run(s)")
    print(f"  File:     {output_path}")
    print(f"\n  Sources breakdown:")
    for source, count in sorted(stats.items()):
        print(f"    {source}: {count}")
    print(f"\n  To train on Modal:")
    print(f"    python run_dpo_worker.py --dataset {output_path}")
    print(f"  Or with your own LlamaFactory checkout (dataset name '{args.name}'):")
    print(f"    llamafactory-cli train <your DPO config>.yaml dataset={args.name}")
    print(f"{'─' * 66}\n")


if __name__ == "__main__":
    main()
