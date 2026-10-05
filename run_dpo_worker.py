#!/usr/bin/env python3
"""
DPO training worker — submits a preference-optimization job to Modal.

Takes a combined DPO dataset (sharegpt format from prepare_dpo_dataset.py)
and launches LlamaFactory DPO training on the assistant model
(assistant.base in configs/models.yaml) via the deployed Modal app.

Usage:
    # Standard DPO run on the combined dataset:
    python run_dpo_worker.py \
        --dataset eval_output/dpo_combined.json \
        --adapter adapter-dpo

    # Custom hyperparameters:
    python run_dpo_worker.py \
        --dataset eval_output/dpo_combined.json \
        --adapter adapter-dpo-v2 \
        --epochs 3 --lr 5e-6 --lora-rank 64 --pref-beta 0.1

    # A different model than assistant.base:
    python run_dpo_worker.py \
        --dataset eval_output/dpo_combined.json \
        --model <hf-model-id> \
        --adapter adapter-dpo-other
"""

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Submit DPO training to Modal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to combined DPO dataset (sharegpt JSON from prepare_dpo_dataset.py)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to DPO-tune (default: assistant.base in configs/models.yaml)",
    )
    parser.add_argument("--adapter", type=str, default="adapter-dpo",
                        help="Adapter name on the Modal volume (default: adapter-dpo)")
    parser.add_argument("--init-adapter", type=str, default="",
                        help="Existing adapter on Modal volume to initialize from (for continued DPO). "
                             "Defaults to empty = train a fresh adapter from scratch.")
    parser.add_argument("--template", type=str, default="",
                        help="LlamaFactory template (default: auto-detect)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--cutoff-len", type=int, default=4096,
                        help="Max sequence length (default: 4096; 8192 ran out of memory for the default assistant on 2× H100)")
    parser.add_argument("--pref-beta", type=float, default=0.1,
                        help="DPO beta parameter (default: 0.1)")
    parser.add_argument("--pref-loss", type=str, default="sigmoid",
                        choices=["sigmoid", "orpo", "simpo"],
                        help="Preference loss function (default: sigmoid = standard DPO)")

    args = parser.parse_args()
    if not args.model:
        from IMPersona2.model_config import model_for
        args.model = model_for("assistant", "base")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    dataset_json = dataset_path.read_text()
    examples = json.loads(dataset_json)
    print(f"DPO Training Worker")
    print(f"  Dataset:  {dataset_path} ({len(examples)} preference pairs)")
    print(f"  Model:    {args.model}")
    print(f"  Adapter:  {args.adapter}")
    if args.init_adapter:
        print(f"  Init from: {args.init_adapter} (continued DPO)")
    print(f"  Epochs:   {args.epochs}")
    print(f"  LR:       {args.lr}")
    print(f"  LoRA:     rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  Batch:    {args.batch_size} x {args.grad_accum} grad_accum")
    print(f"  Beta:     {args.pref_beta}")
    print(f"  Loss:     {args.pref_loss}")

    try:
        import modal
        from pipeline import _modal_app_name

        app_name = _modal_app_name()
        print(f"\n  Modal app: {app_name}")
        print(f"  Calling train_dpo.remote()...", flush=True)

        t0 = time.time()
        train_fn = modal.Function.from_name(app_name, "train_dpo")
        result = train_fn.remote(
            dataset_json,
            model_name=args.model,
            adapter_name=args.adapter,
            init_adapter_name=args.init_adapter,
            template=args.template,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            cutoff_len=args.cutoff_len,
            pref_beta=args.pref_beta,
            pref_loss=args.pref_loss,
        )
        elapsed = time.time() - t0

        print(f"\nDPO training complete! ({elapsed/60:.1f} min)")
        print(f"  Result: {result}")

    except Exception as e:
        print(f"\nDPO training failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
