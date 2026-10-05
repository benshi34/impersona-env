#!/usr/bin/env python3
"""
Standalone training worker — runs as a detached subprocess so it survives
CLI shutdown. All files (config, state, logs) live in a single
run directory passed via --run-dir.

Can also be called directly to restart a failed run or start a new one.

Usage:
    # Restart a failed run (re-reads its config.json):
    python run_sft_worker.py --rerun user-model_2026-01-01_12-00-00

    # Start from a run directory (used by submit_training_job):
    python run_sft_worker.py --run-dir train_output/user-model_2026-01-01_12-00-00

    # Start a new run directly:
    python run_sft_worker.py --model <hf-model-id> --adapter adapter-user-model

    # With custom hyperparameters:
    python run_sft_worker.py --model <hf-model-id> \\
        --adapter adapter-user-model --epochs 5 --lr 1e-4 --lora-rank 128 --gpus 2
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
TRAIN_RUNS_DIR = WORKSPACE / "train_output"
TRAINING_DATA_FILE = WORKSPACE / "training_data.json"


def param_billions(model_name: str):
    """Parameter count in billions read from a model name ("8B", "70b", "8x7B"), or None."""
    name = model_name.rstrip("/").split("/")[-1]
    sizes = [float(n) * float(b) for n, b in re.findall(r"(\d+)x(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", name)]
    sizes += [float(b) for b in re.findall(r"(?<![\dx.])(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", name)]
    return max(sizes) if sizes else None


def gpus_for_model(model_name: str) -> int:
    """H100s for LoRA SFT, by model size: 1 below 20B, 2 below 60B, else 4."""
    size = param_billions(model_name)
    if size is None:
        return 1
    return 4 if size >= 60 else 2 if size >= 20 else 1


def _load_state(run_dir: Path) -> dict:
    state_file = run_dir / "state.json"
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_state(run_dir: Path, update: dict) -> None:
    existing = _load_state(run_dir)
    existing.update(update)
    (run_dir / "state.json").write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _resolve_run_dir(name: str) -> Path:
    """Resolve a run name to a directory path."""
    candidate = TRAIN_RUNS_DIR / name
    if candidate.is_dir():
        return candidate
    matches = sorted(TRAIN_RUNS_DIR.glob(f"{name}*"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Ambiguous name '{name}', matches:")
        for m in matches:
            print(f"  {m.name}")
        sys.exit(1)
    print(f"No run directory matching '{name}' found in {TRAIN_RUNS_DIR}")
    sys.exit(1)


def _build_config_from_args(args) -> tuple[Path, dict]:
    """Create a new run directory and config from CLI args."""
    from IMPersona2.model_config import model_for, template_for

    model = args.model or model_for("user_model", "base")
    adapter = args.adapter or f"adapter-{model.split('/')[-1].lower()}"
    label = args.label or model.split("/")[-1].lower()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = TRAIN_RUNS_DIR / f"{label}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data) if args.data else TRAINING_DATA_FILE
    if not data_path.exists():
        print(f"Training data not found: {data_path}")
        sys.exit(1)

    try:
        num_examples = len(json.loads(data_path.read_text()))
    except Exception:
        num_examples = None

    template = template_for("user_model", model)

    config = {
        "model_name": model,
        "adapter_name": adapter,
        "template": template,
        "training_data": str(data_path),
        "num_examples": num_examples,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target": "all",
        "cutoff_len": args.cutoff_len,
        "bf16": True,
        "gpu": "H100",
        "timestamp": timestamp,
        "label": label,
        "engine": "llamafactory",
    }
    if args.quantization_bit is not None:
        config["quantization_bit"] = args.quantization_bit
    if args.gpus:
        config["gpus"] = args.gpus

    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return run_dir, config


def run_training(run_dir: Path) -> None:
    """Execute the Modal training call. Updates state.json throughout."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        _save_state(run_dir, {"status": "error", "error": f"Config not found: {config_path}"})
        sys.exit(1)

    config = json.loads(config_path.read_text())

    data_path = Path(config.get("training_data", str(TRAINING_DATA_FILE)))
    if not data_path.exists():
        _save_state(run_dir, {"status": "error", "error": f"Training data not found: {data_path}"})
        sys.exit(1)

    model_name = config["model_name"]
    adapter_name = config["adapter_name"]

    # Read prior state before we overwrite it below: if an earlier worker left
    # a still-running detached job, reattach instead of launching a duplicate.
    # Non-running prior states (failed/cancelled/completed) start fresh.
    prior_state = _load_state(run_dir)
    resume_fc_id = (
        prior_state.get("function_call_id")
        if prior_state.get("status") == "running"
        else None
    )

    _save_state(run_dir, {
        "status": "running",
        "uploaded": True,
        "detail": f"Loading training data ({model_name})...",
        "pid": os.getpid(),
        "started_at": time.time(),
    })

    print(f"Training run: {run_dir.name}")
    print(f"  Model:   {model_name}")
    print(f"  Adapter: {adapter_name}")
    print(f"  Data:    {data_path}")

    dataset_json = data_path.read_text()
    num_examples = len(json.loads(dataset_json))
    print(f"  Examples: {num_examples}")

    has_quantization = config.get("quantization_bit") is not None
    gpus = 1 if has_quantization else (config.get("gpus") or gpus_for_model(model_name))
    fn_name = {1: "train_llamafactory", 2: "train_llamafactory_multi_gpu", 4: "train_llamafactory_4gpu"}[gpus]
    print(f"  GPUs:    {gpus} × H100" + ("" if config.get("gpus") or has_quantization else
                                         " (from the model size; override with --gpus)"))
    if has_quantization:
        print(f"  Quantization: {config['quantization_bit']}-bit → using single GPU (QLoRA fits on 1 GPU)")

    try:
        import modal
        from pipeline import _modal_app_name

        app_name = _modal_app_name()
        print(f"  Modal app: {app_name}")
        print(f"  Function:  {fn_name}")

        train_fn = modal.Function.from_name(app_name, fn_name)

        if resume_fc_id:
            print(f"  Reattaching to in-flight Modal call: {resume_fc_id}", flush=True)
            function_call = modal.FunctionCall.from_id(resume_fc_id)
            _save_state(run_dir, {
                "detail": f"Reattached to Modal call {resume_fc_id} ({fn_name})...",
            })
        else:
            # spawn() detaches the job server-side so it survives this worker's
            # connection dropping (laptop sleep, terminal close); remote() would
            # be cancelled. Persist the call id so a later worker can reattach.
            print(f"  Spawning detached Modal call...", flush=True)
            function_call = train_fn.spawn(
                dataset_json,
                model_name=model_name,
                adapter_name=adapter_name,
                template=config.get("template", ""),
                num_epochs=config.get("num_epochs", 3),
                batch_size=config.get("batch_size", 4),
                gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
                learning_rate=config.get("learning_rate", 2e-4),
                lora_rank=config.get("lora_rank", 64),
                lora_alpha=config.get("lora_alpha", 128),
                lora_dropout=config.get("lora_dropout", 0.05),
                cutoff_len=config.get("cutoff_len", 4096),
                quantization_bit=config.get("quantization_bit"),
            )
            _save_state(run_dir, {
                "function_call_id": function_call.object_id,
                "detail": f"GPU provisioning & loading {model_name} ({fn_name})...",
            })
            print(f"  Spawned detached call: {function_call.object_id}", flush=True)

        print(f"  Waiting for training to finish...", flush=True)
        result = function_call.get()

        if isinstance(result, dict) and "llamafactory_config" in result:
            lf_config_path = run_dir / "llamafactory_config.json"
            lf_config_path.write_text(
                json.dumps(result["llamafactory_config"], indent=2),
                encoding="utf-8",
            )

        _save_state(run_dir, {
            "status": "completed",
            "detail": f"Training complete: {result}",
            "finished_at": time.time(),
        })
        print(f"\nTraining complete!")
        print(f"  Result: {result}")

    except Exception as e:
        error_str = str(e)
        short_error = error_str[:500]
        if "gated repo" in error_str.lower() or "401" in error_str:
            short_error = (
                "HuggingFace authentication failed. "
                "Accept the model license on HuggingFace and ensure your "
                "Modal secret has a valid HF_TOKEN."
            )
        elif "cancel" in error_str.lower():
            fc_hint = _load_state(run_dir).get("function_call_id", "?")
            short_error = (
                "Training call was cancelled. The detached Modal job was "
                "stopped (manual `modal app stop`, dashboard Stop, or the job "
                f"itself failed). Inspect with: modal app logs {fc_hint}"
            )
        _save_state(run_dir, {
            "status": "error",
            "error": short_error,
            "detail": short_error,
            "finished_at": time.time(),
        })
        print(f"\nTraining failed: {short_error}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="IMPersona training worker — submit LoRA fine-tuning to Modal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_sft_worker.py --rerun user-model_2026-01-01_12-00-00\n"
            "  python run_sft_worker.py --model <hf-model-id> --adapter adapter-user-model\n"
            "  python run_sft_worker.py --run-dir train_output/user-model_2026-01-01_12-00-00\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rerun", metavar="RUN_NAME",
                       help="Restart a failed run (re-uses its config.json)")
    mode.add_argument("--run-dir", metavar="DIR",
                       help="Run directory with config.json (used by pipeline)")
    mode.add_argument("--model", metavar="MODEL",
                       help="Start a new run with this HuggingFace model")

    parser.add_argument("--adapter", help="Adapter name (default: derived from model)")
    parser.add_argument("--label", help="Run label (default: derived from model)")
    parser.add_argument("--data", help=f"Training data path (default: {TRAINING_DATA_FILE})")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (regularization). Higher = stronger regularization (e.g. 0.1).")
    parser.add_argument("--cutoff-len", type=int, default=4096,
                        help="Max token length per training example (longer examples get truncated)")
    parser.add_argument("--gpus", type=int, default=None, choices=[1, 2, 4],
                        help="H100s to train on (default: chosen from the parameter count in the model name)")
    parser.add_argument("--quantization-bit", type=int, default=None, choices=[4, 8],
                        help="Quantize model (4-bit QLoRA or 8-bit LLM.int8)")

    args = parser.parse_args()

    if args.rerun:
        run_dir = _resolve_run_dir(args.rerun)
        print(f"Re-running: {run_dir.name}")
        run_training(run_dir)

    elif args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_dir():
            print(f"Run directory not found: {run_dir}")
            sys.exit(1)
        run_training(run_dir)

    elif args.model:
        run_dir, config = _build_config_from_args(args)
        print(f"Created run: {run_dir.name}")
        run_training(run_dir)


if __name__ == "__main__":
    main()
