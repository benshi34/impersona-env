#!/usr/bin/env python3
"""
Standalone eval worker — runs as a detached subprocess so it survives
CLI shutdown. All files (config, state, logs, results) live in a single
run directory passed via --run-dir.

Usage:
    python run_eval_worker.py --run-dir eval_output/eval_2026-01-01_12-00-00
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
EVAL_SCENARIOS_FILE = WORKSPACE / "IMPersona2" / "eval_scenarios.json"


def _save_state(run_dir: Path, update: dict) -> None:
    state_file = run_dir / "state.json"
    existing = {}
    if state_file.exists():
        try:
            existing = json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    existing.update(update)
    state_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _warmup_model(label: str, generate_fn, kwargs: dict) -> None:
    """Fire a single dummy request to trigger @modal.enter() model loading."""
    print(f"  Warming up {label}...", flush=True)
    t0 = time.time()
    try:
        generate_fn(**kwargs)
    except Exception as e:
        print(f"  Warmup {label} got error (may be ok): {e}", flush=True)
    print(f"  {label} warm in {time.time() - t0:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Directory for this eval run")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    from dotenv import load_dotenv
    load_dotenv(WORKSPACE / ".env")

    config_path = run_dir / "config.json"
    if not config_path.exists():
        _save_state(run_dir, {"status": "error", "error": f"Config not found: {config_path}"})
        sys.exit(1)

    config = json.loads(config_path.read_text())

    try:
        from IMPersona2.eval import (
            load_scenarios,
            run_conversation,
            evaluate_coherence,
            evaluate_disagreements,
            generate_viewer_html,
            ModalIMPersonaClient,
            ModalPartnerClient,
            AIPartnerClient,
        )
        from IMPersona2.experiment_config import default_persona_name
        from IMPersona2.model_config import model_for, template_for

        app_name = config.get("modal_app_name", "impersona")
        persona_name = config.get("persona_name") or default_persona_name()
        ai_partner_model = config.get("ai_partner_model")
        ai_partner_modal_model = config.get("ai_partner_modal_model")
        if not ai_partner_model and not ai_partner_modal_model:
            # No partner configured: use the assistant from configs/models.yaml on Modal.
            ai_partner_modal_model = model_for("assistant", "base")
        ai_partner_model = ai_partner_model or ai_partner_modal_model
        ai_partner_modal_adapter = config.get("ai_partner_modal_adapter", "")
        impersona_model_name = config.get("impersona_model_name") or model_for("user_model", "base")
        impersona_adapter_name = config.get("impersona_adapter_name", "adapter-user-model")
        impersona_template = config.get("impersona_template") or template_for("user_model", impersona_model_name)
        category = config.get("category")
        quick = config.get("quick", False)
        num_runs = config.get("num_runs", 1)
        max_messages = config.get("max_messages", 20)
        run_label = config.get("run_label", "eval")
        parallel_conversations = config.get("parallel_conversations", 50)
        inference_class = config.get("inference_class", "VLLMInference")
        ai_partner_inference_class = config.get("ai_partner_inference_class") or "VLLMPartner"
        # None = assistant.thinking in configs/models.yaml
        ai_partner_enable_thinking = config.get("ai_partner_enable_thinking")
        critic_mode = config.get("critic_mode", False)
        include_prior_cot = config.get("include_prior_cot", False)

        scenarios = load_scenarios(
            str(EVAL_SCENARIOS_FILE),
            scenario_ids=None,
            category=category,
            quick=quick,
        )

        if not scenarios:
            _save_state(run_dir, {"status": "error", "error": "No scenarios found"})
            sys.exit(1)

        total_runs = len(scenarios) * num_runs
        _save_state(run_dir, {
            "status": "running",
            "detail": "Warming up models...",
            "progress": 0,
            "total": total_runs,
            "pid": os.getpid(),
            "started_at": time.time(),
        })

        # ── Create model clients ──
        print(f"\nIMPersona model: {impersona_model_name}", flush=True)
        print(f"  Adapter: {impersona_adapter_name}", flush=True)
        print(f"  Template: {impersona_template}", flush=True)
        print(f"  Inference class: {inference_class}", flush=True)

        chat_model = ModalIMPersonaClient(
            app_name=app_name,
            model_name=impersona_model_name,
            adapter_name=impersona_adapter_name,
            template=impersona_template,
            cls_name=inference_class,
        )

        if ai_partner_modal_model:
            print(f"AI partner (Modal): {ai_partner_modal_model}", flush=True)
            print(f"  Inference class: {ai_partner_inference_class}", flush=True)
            ai_partner = ModalPartnerClient(
                model_name=ai_partner_modal_model,
                app_name=app_name,
                adapter_name=ai_partner_modal_adapter,
                cls_name=ai_partner_inference_class,
                enable_thinking=ai_partner_enable_thinking,
            )
            print(f"  enable_thinking: {ai_partner.enable_thinking}", flush=True)
        else:
            print(f"AI partner (API): {ai_partner_model}", flush=True)
            ai_partner = AIPartnerClient(model=ai_partner_model)

        # ── Warm up Modal containers in parallel ──
        print("\nWarming up containers (triggering model loads)...", flush=True)
        warmup_start = time.time()
        warmup_threads = []
        impersona_warmup_kwargs = dict(
            system_prompt="You are a person. Respond briefly.",
            conversation_input="[1] [2026-01-01 12:00:00] Friend: Hi\n[2] [2026-01-01 12:00:01] Me: ",
            max_new_tokens=8,
            temperature=0.7,
        )
        t = threading.Thread(
            target=_warmup_model,
            args=("IMPersona", chat_model.generate, impersona_warmup_kwargs),
        )
        t.start()
        warmup_threads.append(t)

        if ai_partner_modal_model:
            partner_warmup_kwargs = dict(
                system_prompt="You are an AI assistant.",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=8,
            )
            t = threading.Thread(
                target=_warmup_model,
                args=("AI Partner", ai_partner.generate, partner_warmup_kwargs),
            )
            t.start()
            warmup_threads.append(t)

        for t in warmup_threads:
            t.join()
        print(f"Warmup done in {time.time() - warmup_start:.1f}s\n", flush=True)

        # ── Resume: reload any conversations already saved in a prior run ──
        # The worker checkpoints to results_partial.json after every finished
        # conversation, so a re-launch pointed at the same run dir can skip
        # work that's already done instead of starting over from scratch.
        partial_path = run_dir / "results_partial.json"
        conversations = []
        completed_keys = set()
        if partial_path.exists():
            try:
                partial = json.loads(partial_path.read_text())
                conversations = partial.get("conversations", [])
                for c in conversations:
                    sid = c.get("scenario", {}).get("id")
                    if sid is not None:
                        completed_keys.add((sid, c.get("run_index", 0)))
                print(
                    f"Resuming: {len(completed_keys)} conversations already "
                    f"completed (skipping those)",
                    flush=True,
                )
            except Exception as e:
                print(f"Could not read partial results ({e}); starting fresh", flush=True)
                conversations = []
                completed_keys = set()

        # ── Build work items (skip anything already completed) ──
        work_items = []
        for scenario in scenarios:
            for run_idx in range(num_runs):
                if (scenario["id"], run_idx) in completed_keys:
                    continue
                work_items.append((scenario, run_idx))

        remaining = len(work_items)
        _save_state(run_dir, {
            "detail": (
                f"Running {remaining} conversations ({parallel_conversations} parallel)"
                + (f" — resumed, {len(completed_keys)} already done" if completed_keys else "")
            ),
        })

        # ── Run conversations in parallel ──
        lock = threading.Lock()
        progress_counter = [len(conversations)]
        failed_count = [0]

        def _run_one(scenario, run_idx):
            result = run_conversation(
                scenario=scenario,
                chat_model=chat_model,
                ai_partner=ai_partner,
                persona_name=persona_name,
                max_messages=max_messages,
                temperature=0.7,
                show_live=True,
                critic_mode=critic_mode,
                include_prior_cot=include_prior_cot,
            )
            if num_runs > 1:
                result["run_index"] = run_idx
            return result

        print(f"Starting {total_runs} conversations with {parallel_conversations} workers...", flush=True)
        conv_start = time.time()

        with ThreadPoolExecutor(max_workers=parallel_conversations) as executor:
            future_to_info = {}
            for scenario, run_idx in work_items:
                future = executor.submit(_run_one, scenario, run_idx)
                future_to_info[future] = (scenario, run_idx)

            for future in as_completed(future_to_info):
                scenario, run_idx = future_to_info[future]
                with lock:
                    progress_counter[0] += 1
                    count = progress_counter[0]

                try:
                    result = future.result()
                    with lock:
                        conversations.append(result)
                        partial_path.write_text(
                            json.dumps({"conversations": conversations}, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    print(f"  [{count}/{total_runs}] {scenario['name']} ✓", flush=True)
                except Exception as e:
                    with lock:
                        failed_count[0] += 1
                    print(f"  [{count}/{total_runs}] {scenario['name']} FAILED: {e}", flush=True)

                _save_state(run_dir, {
                    "detail": f"Completed {count}/{total_runs} conversations",
                    "progress": count,
                })

        elapsed = time.time() - conv_start
        print(f"\nConversations done: {len(conversations)} ok, {failed_count[0]} failed in {elapsed:.1f}s", flush=True)

        # ── Run LLM judges ──
        _save_state(run_dir, {"detail": "Running LLM judges (coherence + disagreement)..."})

        coherence_summary = None
        disagreement_summary = None
        if conversations:
            coherence_summary = evaluate_coherence(
                conversations, persona_name=persona_name,
            )
            disagreement_summary = evaluate_disagreements(
                conversations, persona_name=persona_name,
            )

        partner_usage = ai_partner.get_usage() if hasattr(ai_partner, "get_usage") else {}

        results_bundle = {
            "metadata": {
                "run_label": run_label,
                "persona_name": persona_name,
                "modal_app_name": app_name,
                "impersona_model": impersona_model_name,
                "impersona_adapter": impersona_adapter_name,
                "ai_partner_model": ai_partner_modal_model or ai_partner_model,
                "max_messages": max_messages,
                "num_scenarios": len(scenarios),
                "num_runs": num_runs,
                "num_completed": len(conversations),
                "num_failed": failed_count[0],
                "parallel_conversations": parallel_conversations,
                "conversation_time_s": round(elapsed, 1),
                "critic_mode": critic_mode,
                "include_prior_cot": include_prior_cot,
                "ai_partner_usage": partner_usage,
                "timestamp": datetime.now().isoformat(),
            },
            "evaluations": {
                "coherence": coherence_summary,
                "disagreements": disagreement_summary,
            },
            "conversations": conversations,
        }

        results_path = run_dir / "results.json"
        results_path.write_text(
            json.dumps(results_bundle, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        viewer_html = generate_viewer_html(results_bundle)
        (run_dir / "viewer.html").write_text(viewer_html, encoding="utf-8")

        if partial_path.exists():
            partial_path.unlink()

        _save_state(run_dir, {
            "status": "completed",
            "detail": f"Done — {len(conversations)} conversations in {elapsed:.0f}s",
            "finished_at": time.time(),
        })

    except Exception as e:
        _save_state(run_dir, {
            "status": "error",
            "error": str(e)[:500],
            "detail": str(e)[:500],
            "finished_at": time.time(),
        })
        raise


if __name__ == "__main__":
    main()
