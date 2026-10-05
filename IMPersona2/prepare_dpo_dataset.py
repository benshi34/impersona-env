#!/usr/bin/env python3
"""
Prepare a DPO dataset for LlamaFactory training.

Takes a dpo_dataset*.json produced by generate_dpo_dataset.py and:
1. Converts it from single-turn alpaca format to multi-turn sharegpt format
   (so the model sees full conversation context during DPO training).
2. Copies the converted file into LlamaFactory/data/.
3. Registers it in LlamaFactory/data/dataset_info.json.

Usage:
    # Prepare a specific DPO dataset
    python -m IMPersona2.prepare_dpo_dataset \
        --input eval_output/run_.../dpo_dataset.json \
        --name impersona_dpo

    # With custom LlamaFactory path
    python -m IMPersona2.prepare_dpo_dataset \
        --input eval_output/run_.../dpo_dataset.json \
        --name impersona_dpo_v2 \
        --llamafactory-dir /path/to/LlamaFactory

    # Combine multiple DPO datasets from different eval runs
    python -m IMPersona2.prepare_dpo_dataset \
        --input eval_output/run_A/dpo_dataset.json \
        --input eval_output/run_B/dpo_dataset.json \
        --name impersona_dpo_combined

    # Skip multi-turn conversion (keep single-turn alpaca format)
    python -m IMPersona2.prepare_dpo_dataset \
        --input eval_output/run_.../dpo_dataset.json \
        --name impersona_dpo \
        --single-turn
"""

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List


# ── Default paths (relative to this file's location) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMPERSONA_DIR = os.path.dirname(SCRIPT_DIR)  # repo root
WORKSPACE_DIR = os.path.dirname(IMPERSONA_DIR)  # parent of the repo (local LlamaFactory checkout)
DEFAULT_LLAMAFACTORY_DIR = os.path.join(WORKSPACE_DIR, "LlamaFactory")


def convert_to_sharegpt_dpo(examples: List[Dict]) -> List[Dict]:
    """
    Convert alpaca-style DPO examples to LlamaFactory's sharegpt DPO format.

    Input (alpaca ranking — single-turn):
        {
            "instruction": "You are an AI assistant.",
            "input": "last user message",
            "chosen": "better response text",
            "rejected": "worse response text",
            "_chat_messages_before": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        }

    Output (sharegpt ranking — multi-turn):
        {
            "conversations": [
                {"from": "system", "value": "You are an AI assistant."},
                {"from": "human", "value": "first user message"},
                {"from": "gpt", "value": "first assistant response"},
                ...
                {"from": "human", "value": "last user message"}
            ],
            "chosen": {"from": "gpt", "value": "better response text"},
            "rejected": {"from": "gpt", "value": "worse response text"}
        }
    """
    converted = []

    for ex in examples:
        conversations = []

        # Add system prompt if present and non-empty
        system_prompt = ex.get("instruction", "").strip()
        if system_prompt:
            conversations.append({"from": "system", "value": system_prompt})

        # Add multi-turn conversation history.
        # _chat_messages_before already includes the final user message
        # that triggered the assistant response, so we do NOT need to
        # append `input` separately (that would duplicate the last turn).
        chat_before = ex.get("_chat_messages_before", [])
        for msg in chat_before:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                conversations.append({"from": "human", "value": content})
            elif role == "assistant":
                conversations.append({"from": "gpt", "value": content})

        # If no conversation history was available, fall back to the
        # `input` field as the sole user message.
        if not chat_before:
            final_input = ex.get("input", "").strip()
            if final_input:
                conversations.append({"from": "human", "value": final_input})

        converted_example: Dict[str, Any] = {
            "conversations": conversations,
            "chosen": {"from": "gpt", "value": ex["chosen"]},
            "rejected": {"from": "gpt", "value": ex["rejected"]},
        }

        # Preserve metadata fields for debugging/filtering
        for key, value in ex.items():
            if key.startswith("_") and key != "_chat_messages_before":
                converted_example[key] = value

        converted.append(converted_example)

    return converted


def register_dataset(
    dataset_info_path: str,
    name: str,
    filename: str,
    is_sharegpt: bool,
) -> None:
    """Add or update the dataset entry in dataset_info.json."""
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    if is_sharegpt:
        entry: Dict[str, Any] = {
            "file_name": filename,
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        }
    else:
        # Alpaca-style (single-turn)
        entry = {
            "file_name": filename,
            "ranking": True,
            "columns": {
                "prompt": "input",
                "chosen": "chosen",
                "rejected": "rejected",
                "system": "instruction",
            },
        }

    already_existed = name in dataset_info
    dataset_info[name] = entry

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    action = "Updated" if already_existed else "Registered"
    print(f"  ✅ {action} '{name}' in {dataset_info_path}")


def load_and_combine(input_paths: List[str]) -> List[Dict]:
    """Load and combine one or more DPO dataset JSON files."""
    all_examples = []
    for path in input_paths:
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            print(f"  ⚠️  File not found: {abs_path}")
            continue

        with open(abs_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        print(f"  📄 Loaded {len(examples)} examples from {os.path.basename(abs_path)}")
        all_examples.extend(examples)

    return all_examples


def deduplicate(examples: List[Dict]) -> List[Dict]:
    """Remove duplicate DPO pairs based on scenario_id + turn_index + run_id + transcript position.

    The key includes ``_transcript_ai_msg_index`` so that distinct conversation
    iterations (e.g. several runs of the same scenario) are preserved, while
    true duplicates (filtered subsets, multi-rewrite-model variants of the same
    source turn) are still collapsed.
    """
    seen = set()
    deduped = []
    for ex in examples:
        key = (
            ex.get("_scenario_id", ""),
            ex.get("_turn_index", ""),
            ex.get("_run_id", ""),
            ex.get("_transcript_ai_msg_index", ""),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(ex)
    dupes = len(examples) - len(deduped)
    if dupes > 0:
        print(f"  🔄 Removed {dupes} duplicate examples")
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a DPO dataset for LlamaFactory training."
    )
    parser.add_argument(
        "--input",
        type=str,
        action="append",
        required=True,
        help=(
            "Path to a dpo_dataset*.json file from generate_dpo_dataset.py. "
            "Can be specified multiple times to combine datasets."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default="impersona_dpo",
        help="Dataset name for LlamaFactory registration (default: impersona_dpo)",
    )
    parser.add_argument(
        "--llamafactory-dir",
        type=str,
        default=DEFAULT_LLAMAFACTORY_DIR,
        help=f"Path to LlamaFactory directory (default: {DEFAULT_LLAMAFACTORY_DIR})",
    )
    parser.add_argument(
        "--single-turn",
        action="store_true",
        help=(
            "Keep the single-turn alpaca format instead of converting to "
            "multi-turn sharegpt. Not recommended: the model won't see "
            "conversation context during DPO training."
        ),
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip registering the dataset in dataset_info.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing any files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    llamafactory_data_dir = os.path.join(args.llamafactory_dir, "data")
    dataset_info_path = os.path.join(llamafactory_data_dir, "dataset_info.json")

    if not os.path.isdir(llamafactory_data_dir):
        print(f"❌ LlamaFactory data dir not found: {llamafactory_data_dir}")
        sys.exit(1)
    if not os.path.isfile(dataset_info_path):
        print(f"❌ dataset_info.json not found: {dataset_info_path}")
        sys.exit(1)

    print(f"\n{'═' * 60}")
    print(f"  IMPersona DPO Dataset Preparation")
    print(f"{'═' * 60}")

    # ── Step 1: Load input(s) ──
    print(f"\n📥 Loading input dataset(s)...")
    examples = load_and_combine(args.input)
    if not examples:
        print("❌ No examples loaded. Check your input paths.")
        sys.exit(1)

    # Deduplicate if combining multiple files
    if len(args.input) > 1:
        examples = deduplicate(examples)

    print(f"  Total examples: {len(examples)}")

    # Check if multi-turn context is available
    has_chat_history = any(
        len(ex.get("_chat_messages_before", [])) > 0 for ex in examples
    )
    examples_with_history = sum(
        1 for ex in examples if len(ex.get("_chat_messages_before", [])) > 0
    )

    # ── Step 2: Format conversion ──
    is_sharegpt = not args.single_turn
    if is_sharegpt:
        if has_chat_history:
            print(f"\n🔄 Converting to multi-turn sharegpt format...")
            print(f"  {examples_with_history}/{len(examples)} examples have conversation context")
            examples = convert_to_sharegpt_dpo(examples)
            print(f"  ✅ Converted {len(examples)} examples to sharegpt format")
        else:
            print(f"\n⚠️  No conversation context found in _chat_messages_before.")
            print(f"  Falling back to single-turn alpaca format.")
            is_sharegpt = False

    if not is_sharegpt:
        print(f"\n📋 Using single-turn alpaca format")
        # Strip metadata fields to keep the file clean for LlamaFactory
        clean_examples = []
        for ex in examples:
            clean = {
                "instruction": ex.get("instruction", ""),
                "input": ex.get("input", ""),
                "chosen": ex.get("chosen", ""),
                "rejected": ex.get("rejected", ""),
            }
            clean_examples.append(clean)
        examples = clean_examples

    # ── Step 3: Write to LlamaFactory/data/ ──
    output_filename = f"{args.name}.json"
    output_path = os.path.join(llamafactory_data_dir, output_filename)

    if args.dry_run:
        print(f"\n🏃 Dry run — would write:")
        print(f"  Dataset: {output_path} ({len(examples)} examples)")
        if not args.no_register:
            print(f"  Register: '{args.name}' in {dataset_info_path}")
        print(f"\n  Format: {'sharegpt (multi-turn)' if is_sharegpt else 'alpaca (single-turn)'}")
        return

    print(f"\n💾 Writing dataset...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  ✅ Saved {len(examples)} examples to {output_path}")

    # ── Step 4: Register in dataset_info.json ──
    if not args.no_register:
        print(f"\n📝 Registering dataset...")
        register_dataset(
            dataset_info_path=dataset_info_path,
            name=args.name,
            filename=output_filename,
            is_sharegpt=is_sharegpt,
        )

    # ── Summary ──
    print(f"\n{'─' * 60}")
    print(f"  ✅ Done! Dataset '{args.name}' is ready for DPO training.")
    print(f"  Format: {'sharegpt (multi-turn)' if is_sharegpt else 'alpaca (single-turn)'}")
    print(f"  Examples: {len(examples)}")
    print(f"  File: {output_path}")
    print(f"\n  To train on Modal:")
    print(f"    python run_dpo_worker.py --dataset {output_path}")
    print(f"  Or with your own LlamaFactory checkout (dataset name '{args.name}'):")
    print(f"    llamafactory-cli train <your DPO config>.yaml dataset={args.name}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
