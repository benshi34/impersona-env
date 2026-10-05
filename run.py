#!/usr/bin/env python3
"""
Impersona-Env — End-to-End Pipeline Runner (CLI)
=================================================

Interactive CLI for the full loop. At each step you choose whether to
run it, skip it, or re-run with --force.

  Step 1: Export iMessages from this Mac
  Step 2: Select conversations for training
  Step 3: Generate SFT dataset (via the configured annotation model)
  Step 4: Fine-tune on Modal cloud GPUs
  Step 5: Evaluate IMPersona vs AI partner
  Step 5b: (optional) Style + fact-recall judges for the user model
  Step 6: Generate DPO dataset from disagreements
  Step 7: Prepare combined DPO dataset for training
  Step 8: DPO-train the assistant on Modal
  Step 9: Build the memory-search corpus + index (arena baseline)
  Step 10: Blind arena — compare assistants in your normal chat use

Usage:
    python run.py                     # interactive walkthrough
    python run.py --step export       # jump to a specific step
    python run.py --step train        # jump to training
    python run.py --step eval         # jump to evaluation
    python run.py --step fidelity     # style + fact-recall judges
    python run.py --step dpo          # jump to DPO generation
    python run.py --step prepare-dpo  # jump to DPO dataset preparation
    python run.py --step train-dpo    # jump to DPO training
    python run.py --step memory       # build memory corpus + index
    python run.py --step arena        # blind arena
"""

import argparse
import re
import sys
import time

from pipeline import (
    CONTACTS_VCF,
    MAC_EXPORT_DIR,
    IOS_EXPORT_DIR,
    MERGED_EXPORT_DIR,
    PERSONA_CONFIG_FILE,
    SELECTED_CONVOS_FILE,
    build_contacts_lookup,
    combine_datasets_to_jsonl,
    export_contacts_via_applescript,
    find_existing_vcf,
    find_iphone_backups,
    get_eval_scenarios,
    get_eval_status,
    list_eval_runs,
    get_generation_status,
    get_training_status,
    has_imessage_exporter,
    install_imessage_exporter,
    is_macos,
    load_persona,
    load_selection,
    merge_exports,
    pick_best_export_dir,
    run_generation,
    run_imessage_export,
    save_persona,
    save_selection,
    scan_all_conversations,
    submit_eval_job,
    submit_training_job,
    upload_training_file,
)


def _ask_proceed(step_name: str, description: str, done: bool = False) -> bool:
    """Ask user whether to run this step. Returns True to proceed."""
    print()
    print("=" * 60)
    status = " [DONE]" if done else ""
    print(f"  {step_name}{status}")
    print(f"  {description}")
    print("=" * 60)

    if done:
        resp = input("\n  Already completed. Re-run? [y/N] ").strip().lower()
        return resp == "y"
    else:
        resp = input("\n  Run this step? [Y/n/q] ").strip().lower()
        if resp == "q":
            print("\n  Exiting.")
            sys.exit(0)
        return resp != "n"


# ───────────────────────────────────────────────
# CLI: Contacts export (interactive)
# ───────────────────────────────────────────────

def _cli_export_contacts(force: bool = False) -> None:
    existing = find_existing_vcf()
    if existing and not force:
        print(f"  Contacts VCF already exists: {existing}")
        return

    print("  " + "-" * 56)
    print("  CONTACTS EXPORT")
    print("  " + "-" * 56)
    print()
    print("  A contacts file (.vcf) is needed to map phone numbers to")
    print("  real names. Without it, conversations will be labeled by")
    print("  phone number (e.g. '+15555550123') instead of by name.")
    print()
    print("  IMPORTANT: Your Mac's Contacts app often has far fewer")
    print("  contacts than your iPhone. For best results, export from")
    print("  your phone.")
    print()
    print("  Choose an option:")
    print()
    print("    [1] Export from iPhone (RECOMMENDED)")
    print("        On your iPhone:")
    print("          Contacts -> Lists -> long-press 'All Contacts'")
    print("          -> Export -> AirDrop to this Mac")
    print(f"        Save the file as: {CONTACTS_VCF}")
    print()
    print("    [2] Export from Mac Contacts app (may be incomplete)")
    print("        Contacts app -> Select All (Cmd+A)")
    print("        -> File -> Export vCard...")
    print(f"        Save the file as: {CONTACTS_VCF}")
    print()
    print("    [3] Auto-export from Mac Contacts app now")
    print("        (Quick, but may have fewer contacts than your phone)")
    print()
    print("    [4] Skip — continue without contacts")
    print("        (Phone numbers will be used instead of names)")
    print()

    while True:
        choice = input("  Your choice [1/2/3/4]: ").strip()

        if choice in ("1", "2"):
            print()
            print(f"  Please save the VCF file as:")
            print(f"    {CONTACTS_VCF}")
            print()
            input("  Press Enter when the file is ready...")
            if CONTACTS_VCF.exists() and CONTACTS_VCF.stat().st_size > 0:
                n = CONTACTS_VCF.read_text().count("BEGIN:VCARD")
                print(f"  Found {n} contacts in {CONTACTS_VCF}")
                return
            found = find_existing_vcf()
            if found:
                print(f"  Found VCF at: {found}")
                return
            print(f"  File not found at {CONTACTS_VCF}.")
            continue

        elif choice == "3":
            print()
            print("  Exporting from Mac Contacts app via AppleScript...")
            n, msg = export_contacts_via_applescript(CONTACTS_VCF)
            if n > 0:
                print(f"  {msg}")
                return
            print(f"  {msg}")
            continue

        elif choice == "4":
            print()
            print("  Skipping contacts export.")
            return

        else:
            print("  Invalid choice. Please enter 1, 2, 3, or 4.")


# ───────────────────────────────────────────────
# CLI: Step 1 — Export
# ───────────────────────────────────────────────

def step_export(force: bool = False) -> None:
    if not is_macos():
        print("  ERROR: iMessage export is only supported on macOS.")
        print(f"  Place exported .txt files in: {MAC_EXPORT_DIR}")
        sys.exit(1)

    has_messages = MAC_EXPORT_DIR.exists() and any(MAC_EXPORT_DIR.glob("*.txt"))
    has_contacts = find_existing_vcf() is not None

    if has_messages and has_contacts and not force:
        n = len(list(MAC_EXPORT_DIR.glob("*.txt")))
        print(f"  Found existing export with {n} conversations")
        print(f"  Contacts VCF: {find_existing_vcf()}")
        return

    if has_messages and not has_contacts and not force:
        n = len(list(MAC_EXPORT_DIR.glob("*.txt")))
        print(f"  Found existing export with {n} conversations")
        print("  But no contacts VCF found — exporting contacts now...\n")
        _cli_export_contacts(force=False)
        return

    if not has_imessage_exporter():
        print("  imessage-exporter not found. Attempting to install...")
        ok, msg = install_imessage_exporter()
        if not ok:
            print(f"  ERROR: {msg}")
            sys.exit(1)
        print(f"  {msg}")

    print("\n  Exporting from Mac iMessage database...")
    print("  (You may be prompted to grant Full Disk Access in System Settings.)\n")
    ok, msg = run_imessage_export(MAC_EXPORT_DIR)
    if not ok:
        print(f"  ERROR: {msg}")
        sys.exit(1)
    print(f"  {msg}")

    _cli_export_contacts(force)

    backups = find_iphone_backups()
    if backups:
        latest = backups[-1]
        print(f"\n  Found iPhone backup: {latest.name}")
        resp = input("  Export from this iPhone backup too? [y/N] ").strip().lower()
        if resp == "y":
            ok, msg = run_imessage_export(IOS_EXPORT_DIR, db_path=latest)
            print(f"  {msg}")
            if ok:
                print("\n  Merging Mac and iOS exports...")
                n = merge_exports(MAC_EXPORT_DIR, IOS_EXPORT_DIR, MERGED_EXPORT_DIR)
                print(f"  Merged: {n} conversation files")
    else:
        print("\n  No iPhone backups found — using Mac export only.")

    print(f"\n  Export directory: {pick_best_export_dir()}")


# ───────────────────────────────────────────────
# CLI: Step 2 — Select Conversations
# ───────────────────────────────────────────────

def step_select(force: bool = False) -> list[dict]:
    if not force:
        saved = load_selection()
        if saved:
            total = sum(c.get("training_examples", 0) for c in saved)
            print(f"  Found existing selection: {len(saved)} conversations, {total:,} examples")
            return saved

    export_dir = pick_best_export_dir()
    vcf = find_existing_vcf()
    contacts = build_contacts_lookup(vcf)
    if contacts:
        print(f"  Loaded {len(contacts)} contact entries for name resolution")

    files = list(export_dir.glob("*.txt"))
    print(f"  Scanning {len(files)} files (parsing, filtering, counting)...", flush=True)
    results, skipped_group, skipped_empty = scan_all_conversations(export_dir, contacts)
    print(f"  Done. {len(results)} 1-on-1 conversations with usable data.")
    print(f"  (Excluded {skipped_group} group chats, {skipped_empty} with no usable examples)")

    if not results:
        print("  ERROR: No conversations with usable training examples found.")
        sys.exit(1)

    show_n = min(30, len(results))
    print(f"\n  Top {show_n} conversations by training examples (after filtering):\n")
    print(f"  {'#':>4}  {'Name':<40} {'Examples':>8}  {'Convos':>6}  {'Raw Msgs':>8}")
    print(f"  {'─'*4}  {'─'*40} {'─'*8}  {'─'*6}  {'─'*8}")
    for i, c in enumerate(results[:show_n]):
        print(
            f"  {i+1:4}  {c['display_name'][:40]:<40} "
            f"{c['training_examples']:>8}  "
            f"{c['usable_conversations']:>6}  "
            f"{c['raw_messages']:>8}"
        )

    total_all = len(results)
    if total_all > show_n:
        print(f"\n  Showing {show_n} of {total_all} conversations.")
        print("  Enter 'more' to see the next batch, or 'all' to see all.")

    print("\n  " + "-" * 56)
    print("  Select conversations to include in training.\n")
    print("  Enter numbers separated by commas/spaces, or ranges:")
    print("    e.g.  1,2,3,5-10")
    print("    'top N'  — select the top N conversations")
    print("    'all'    — select all shown conversations")
    print("    'more'   — show more conversations")
    print()

    while True:
        raw = input("  Selection: ").strip().lower()

        if raw in ("more", "all"):
            limit = total_all if raw == "all" else min(show_n + 30, total_all)
            if limit > show_n:
                print()
                for i in range(show_n, limit):
                    c = results[i]
                    print(
                        f"  {i+1:4}  {c['display_name'][:40]:<40} "
                        f"{c['training_examples']:>8}  "
                        f"{c['usable_conversations']:>6}  "
                        f"{c['raw_messages']:>8}"
                    )
                show_n = limit
                print()
            continue

        if raw.startswith("top "):
            try:
                n = int(raw.split()[1])
                selected = results[:n]
                break
            except (ValueError, IndexError):
                print("  Invalid. Use 'top 10', 'top 20', etc.")
                continue

        selected = []
        try:
            parts = re.split(r"[,\s]+", raw)
            for part in parts:
                if not part:
                    continue
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    for idx in range(int(lo) - 1, int(hi)):
                        if 0 <= idx < total_all:
                            selected.append(results[idx])
                else:
                    idx = int(part) - 1
                    if 0 <= idx < total_all:
                        selected.append(results[idx])
                    else:
                        print(f"  Invalid index: {part}")
                        selected = []
                        break
        except ValueError:
            print("  Could not parse selection. Examples: '1,2,3', '1-5', 'top 10'")
            continue

        if selected:
            break

    seen = set()
    deduped = []
    for c in selected:
        if c["file"] not in seen:
            seen.add(c["file"])
            deduped.append(c)
    selected = deduped

    for c in selected:
        if "files" not in c:
            c["files"] = [c["file"]]

    total_examples = sum(c["training_examples"] for c in selected)
    print(f"\n  Selected {len(selected)} conversations:\n")
    print(f"  {'#':>4}  {'Name':<40} {'Examples':>8}")
    print(f"  {'─'*4}  {'─'*40} {'─'*8}")
    for i, c in enumerate(selected):
        print(f"  {i+1:4}  {c['display_name'][:40]:<40} {c['training_examples']:>8}")
    print(f"\n  Total training examples: {total_examples:,}")

    confirm = input("\n  Proceed with this selection? [Y/n] ").strip().lower()
    if confirm == "n":
        print("  Cancelled.")
        return []

    save_selection(selected)
    print(f"\n  Selection saved to: {SELECTED_CONVOS_FILE}")
    return selected


# ───────────────────────────────────────────────
# CLI: Step 3 — Generate Dataset
# ───────────────────────────────────────────────

def _cli_ensure_persona() -> dict:
    """Prompt for persona info if not already saved."""
    persona = load_persona()
    if persona:
        print(f"  Persona: {persona['full_name']} ({', '.join(persona.get('nicknames', []))})")
        resp = input("  Use this persona? [Y/n] ").strip().lower()
        if resp != "n":
            return persona

    print("\n  Set up your persona (used in training prompts):\n")
    name = input("  First name (e.g. Jane): ").strip()
    if not name:
        print("  Name is required.")
        sys.exit(1)
    full_name = input(f"  Full name (e.g. Jane Doe) [{name}]: ").strip() or name
    nicks_raw = input("  Nicknames (comma-separated, optional): ").strip()
    nicknames = [n.strip() for n in nicks_raw.split(",") if n.strip()] if nicks_raw else []

    persona = {"name": name, "full_name": full_name, "nicknames": nicknames}
    save_persona(persona)
    print(f"\n  Saved persona: {full_name}")
    return persona


def step_generate(force: bool = False) -> None:
    selection = load_selection()
    if not selection:
        print("  ERROR: No conversations selected. Run step 2 first.")
        return

    print(f"  {len(selection)} contacts selected\n")
    _cli_ensure_persona()

    def on_progress(name, status, detail):
        symbols = {"running": "...", "done": "OK", "error": "ERR"}
        print(f"  [{symbols.get(status, '?')}] {name[:40]:<40} {detail}")

    print()
    result = run_generation(force=force, on_progress=on_progress)

    if "error" in result:
        print(f"\n  ERROR: {result['error']}")
        return

    print(f"\n  Summary:")
    print(f"    Processed: {result.get('processed', 0)}")
    print(f"    Skipped:   {result.get('skipped', 0)}")
    print(f"    Errors:    {result.get('errors', 0)}")
    print(f"    Cost:      ${result.get('total_cost', 0):.2f}")


# ───────────────────────────────────────────────
# CLI: Step 4 — Train on Modal
# ───────────────────────────────────────────────

def step_train(force: bool = False) -> None:
    status = get_training_status()
    if status["stage"] == "completed" and not force:
        print("  Training already complete!")
        return

    if status["stage"] == "training":
        print("  Training is already running on Modal.")
        print(f"  Status: {status.get('status', '?')}")
        resp = input("  Monitor progress? [Y/n] ").strip().lower()
        if resp == "n":
            return
        _poll_training()
        return

    if force and status["stage"] == "completed":
        print("  Re-running training (a previous run already completed)...")

    print("\n  [1/3] Combining datasets...")
    result = combine_datasets_to_jsonl()
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    print(f"  {result['examples']} examples from {result['contacts']} contacts")

    print("\n  [2/3] Validating training data...")
    result = upload_training_file()
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    print(f"  {result.get('examples', '?')} examples ready ({result.get('size_mb', '?')} MB)")

    print("\n  [3/3] Submitting to Modal (GPU provisioning)...")
    result = submit_training_job()
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    print(f"  {result.get('message', 'Training submitted')}")

    _poll_training()


def _poll_training() -> None:
    print("\n  Monitoring training progress (Ctrl+C to detach)...")
    print("  Training runs as a detached process — safe to disconnect.\n")
    try:
        while True:
            time.sleep(10)
            status = get_training_status()
            if status["stage"] == "completed":
                print(f"\n  Training complete!")
                break
            elif status["stage"] == "error":
                print(f"\n  Training failed: {status.get('error', 'unknown')}")
                break
            else:
                detail = status.get("status") or "running..."
                elapsed = status.get("elapsed")
                if elapsed:
                    mins, secs = divmod(elapsed, 60)
                    detail = f"[{mins}m {secs}s] {detail}"
                print(f"  {detail}        ", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\n\n  Detached. Training continues in background.")
        print("  Re-check with: python run.py --step train")
        print("  Or view logs:  modal app logs impersona")


# ───────────────────────────────────────────────
# CLI: Step 5 — Evaluate
# ───────────────────────────────────────────────

def _monitor_eval(run_id: str | None = None) -> None:
    """Poll eval status until it finishes; Ctrl+C detaches (job keeps running)."""
    print("  Evaluation running. Monitoring progress (Ctrl+C to detach)...")
    try:
        while True:
            time.sleep(10)
            status = get_eval_status(run_id)
            if status["stage"] == "completed":
                print(f"\n  Evaluation complete!")
                print(f"  Results: {status.get('results_dir', '?')}")
                break
            elif status["stage"] == "error":
                print(f"\n  Evaluation failed: {status.get('error', 'unknown')}")
                break
            else:
                detail = status.get("status") or "running..."
                elapsed = status.get("elapsed", 0)
                mins = elapsed // 60 if elapsed else 0
                secs = elapsed % 60 if elapsed else 0
                time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
                print(f"  [{time_str}] {detail}        ", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\n\n  Detached. Evaluation continues in background.")


def _find_resumable_evals() -> list:
    """Return eval run dirs that have saved partial progress but never finished."""
    import json as _json
    from pathlib import Path

    eval_dir = Path("eval_output")
    resumable = []
    if not eval_dir.exists():
        return resumable
    for d in sorted(eval_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not d.is_dir():
            continue
        partial = d / "results_partial.json"
        if partial.exists() and not (d / "results.json").exists():
            try:
                n = len(_json.loads(partial.read_text()).get("conversations", []))
            except Exception:
                n = "?"
            resumable.append((d.name, n))
    return resumable


def step_eval() -> None:
    persona = load_persona()
    if not persona:
        print("  ERROR: No persona configured. Run steps 1-4 first.")
        return

    print(f"\n  Persona: {persona.get('name', '?')}")

    # ── Offer to resume an interrupted run with saved progress ──
    resumable = _find_resumable_evals()
    if resumable:
        print("\n  Found interrupted eval run(s) with saved progress:")
        for i, (name, n) in enumerate(resumable[:5], 1):
            print(f"    [{i}] {name}  ({n} conversations done)")
        print("    [n] Start a new evaluation instead")
        choice = (input("  Resume which run? [n]: ").strip().lower() or "n")
        if choice != "n":
            try:
                idx = int(choice) - 1
            except ValueError:
                idx = -1
            if 0 <= idx < len(resumable):
                run_id = resumable[idx][0]
                result = submit_eval_job(resume_run_id=run_id)
                if "error" in result:
                    print(f"  ERROR: {result['error']}")
                    return
                _monitor_eval(result.get("run_id"))
                return
            print("  Invalid choice — starting a new evaluation.\n")

    scenarios = get_eval_scenarios()
    categories = sorted({s["category"] for s in scenarios})

    print(f"  Available: {len(scenarios)} scenarios across {len(categories)} categories\n")

    from IMPersona2.model_config import model_for
    assistant_base = model_for("assistant", "base")

    print("  AI Partner (the assistant the IMPersona talks to):")
    print(f"    [1] {assistant_base} on Modal (assistant.base in configs/models.yaml)")
    print("    [2] Another HuggingFace model on Modal")
    print("    [3] API model (claude-*, gpt-*, or OpenAI-compatible)")
    print()

    partner_choice = input("  AI Partner [1]: ").strip() or "1"
    ai_partner_modal_model = None
    ai_partner_modal_adapter = None
    ai_partner_inference_class = None
    ai_partner_enable_thinking = None
    if partner_choice == "3":
        ai_partner = input("  Model name: ").strip()
        if not ai_partner:
            print("  ERROR: no model name given.")
            return
    else:
        ai_partner = assistant_base
        if partner_choice == "2":
            ai_partner = input(f"  Modal HuggingFace model [{assistant_base}]: ").strip() or assistant_base
        ai_partner_modal_model = ai_partner
        ai_partner_inference_class = "VLLMPartner"

    if ai_partner_modal_model:
        ai_partner_modal_adapter = (
            input("  Modal partner LoRA adapter name (optional): ").strip()
            or None
        )

    print(f"\n  Scenario options:")
    print(f"    [1] Quick subset (preselected mix)")
    print(f"    [2] All scenarios ({len(scenarios)})")
    for i, cat in enumerate(categories):
        cat_count = sum(1 for s in scenarios if s["category"] == cat)
        print(f"    [{i+3}] {cat} ({cat_count})")
    print()

    scenario_choice = input("  Scenarios [1]: ").strip() or "1"
    quick = False
    category = None
    if scenario_choice == "1":
        quick = True
    elif scenario_choice == "2":
        pass
    else:
        try:
            idx = int(scenario_choice) - 3
            if 0 <= idx < len(categories):
                category = categories[idx]
        except ValueError:
            pass

    num_runs_str = input("  Runs per scenario [1]: ").strip() or "1"
    num_runs = max(1, int(num_runs_str)) if num_runs_str.isdigit() else 1

    include_prior_cot = input(
        "  Show prior-turn <think> blocks in history? Only for an IMPersona trained on\n"
        "  data processed by IMPersona2.add_prior_cot [y/N]: "
    ).strip().lower() == "y"
    critic_mode = input("  Critic mode (nudge IMPersona to react authentically)? [y/N]: ").strip().lower() == "y"

    label = input("  Run label (optional): ").strip() or None

    print(f"\n  Starting evaluation:")
    if ai_partner_modal_model:
        print(f"    AI Partner: Modal VLLMPartner ({ai_partner_modal_model})")
        if ai_partner_modal_adapter:
            print(f"    Partner LoRA: {ai_partner_modal_adapter}")
    else:
        print(f"    AI Partner: {ai_partner}")
    if quick:
        print(f"    Scenarios:  quick subset")
    elif category:
        print(f"    Scenarios:  {category}")
    else:
        print(f"    Scenarios:  all ({len(scenarios)})")
    print(f"    Runs/each:  {num_runs}")
    print(f"    Prior CoT:  {include_prior_cot}  ·  Critic mode: {critic_mode}")
    print()

    result = submit_eval_job(
        ai_partner_model=ai_partner,
        ai_partner_modal_model=ai_partner_modal_model,
        ai_partner_modal_adapter=ai_partner_modal_adapter,
        ai_partner_inference_class=ai_partner_inference_class,
        ai_partner_enable_thinking=ai_partner_enable_thinking,
        quick=quick,
        category=category,
        num_runs=num_runs,
        run_label=label,
        critic_mode=critic_mode,
        include_prior_cot=include_prior_cot,
    )
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    _monitor_eval(result.get("run_id"))


# ───────────────────────────────────────────────
# CLI: Step 5b — Fidelity evals (style + fact recall)
# ───────────────────────────────────────────────

def step_fidelity() -> None:
    import subprocess
    from pathlib import Path
    from pipeline import MEMORY_CORPUS_DIR

    print("\n  Fidelity evals for the user model:")
    print("    [1] Build or continue the fact bank (propose → search → review)")
    print("    [2] Score an eval run with the style and fact-recall judges")
    print()
    choice = input("  Choice [1]: ").strip() or "1"

    if choice == "1":
        if not (MEMORY_CORPUS_DIR / "manifest.json").exists():
            print("\n  The fact search reads your messages through the memory index (step 9).")
            if input("  Build it now? [Y/n] ").strip().lower() in ("", "y"):
                step_memory()
            if not (MEMORY_CORPUS_DIR / "manifest.json").exists():
                print("  ERROR: memory index still missing.")
                return
        for stage in ("propose", "search"):
            if subprocess.run([sys.executable, "-m", "IMPersona2.fact_recall", stage]).returncode != 0:
                return
        subprocess.run([sys.executable, "-m", "IMPersona2.fact_recall", "review"])
        return

    if choice != "2":
        print("  Invalid choice.")
        return

    eval_dir = Path("eval_output")
    runs = sorted(
        [d for d in eval_dir.iterdir() if d.is_dir() and (d / "results.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    ) if eval_dir.exists() else []
    if not runs:
        print("  ERROR: No eval runs with results.json found. Run step 5 first.")
        return

    print(f"\n  Available eval runs:\n")
    shown = runs[:10]
    for i, run in enumerate(shown):
        print(f"    [{i+1}] {run.name}")
    print()
    pick = input("  Select run [1]: ").strip() or "1"
    if not pick.isdigit() or not 1 <= int(pick) <= len(shown):
        print("  Invalid choice.")
        return
    run_dir = shown[int(pick) - 1]

    judges = input("  Judges (style, facts) [style,facts]: ").strip() or "style,facts"
    subprocess.run([sys.executable, "-m", "IMPersona2.eval", "--eval-only", str(run_dir), "--judges", judges])


# ───────────────────────────────────────────────
# CLI: Step 6 — Generate DPO Dataset
# ───────────────────────────────────────────────

def step_dpo() -> None:
    from pathlib import Path

    eval_dir = Path("eval_output")
    if not eval_dir.exists():
        print("  ERROR: No eval output found. Run step 5 first.")
        return

    runs = sorted(
        [d for d in eval_dir.iterdir() if d.is_dir() and (d / "results.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        print("  ERROR: No eval runs with results.json found.")
        return

    print(f"\n  Available eval runs:\n")
    for i, run in enumerate(runs[:10]):
        print(f"    [{i+1}] {run.name}")
    print()

    choice = input(f"  Select run [{1}]: ").strip() or "1"
    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(runs)):
            print("  Invalid choice.")
            return
    except ValueError:
        print("  Invalid choice.")
        return

    run_dir = runs[idx]
    print(f"\n  Selected: {run_dir.name}")

    from IMPersona2.model_config import model_for
    rewrite_api_model = model_for("rewrite")

    print("\n  Rewrite model options:")
    print("    [1] On-policy via Modal GPU (recommended — uses the eval partner model)")
    print(f"    [2] {rewrite_api_model} (API — rewrite.model in configs/models.yaml)")
    print("    [3] Another API model")
    print()

    model_choice = input("  Rewrite model [1]: ").strip() or "1"

    if model_choice == "2":
        rewrite_label = rewrite_api_model
    elif model_choice == "3":
        rewrite_label = input("  Model name: ").strip() or rewrite_api_model
    else:
        rewrite_label = None

    if rewrite_label:
        rewrite_args = ["--rewrite-api-model", rewrite_label]
    else:
        rewrite_args = ["--rewrite-modal"]
        rewrite_label = "Modal VLLMPartner (on-policy)"

    print(f"\n  Running DPO generation...")
    print(f"    Run: {run_dir}")
    print(f"    Rewrite model: {rewrite_label}")
    print()

    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "IMPersona2.generate_dpo_dataset",
            "--run", str(run_dir),
            *rewrite_args,
            "--max-workers", "20",
        ],
        cwd=str(Path(__file__).resolve().parent),
    )

    if result.returncode != 0:
        print(f"\n  ERROR: DPO generation failed (exit code {result.returncode})")
    else:
        dpo_path = run_dir / "dpo_dataset.json"
        viewer_path = run_dir / "dpo_dataset_viewer.html"
        if dpo_path.exists():
            import json
            data = json.loads(dpo_path.read_text())
            print(f"\n  DPO dataset: {len(data)} preference pairs")
            print(f"  Saved to: {dpo_path}")
        if viewer_path.exists():
            print(f"  Viewer: {viewer_path}")


# ───────────────────────────────────────────────
# CLI: Step 7 — Prepare DPO Dataset for Training
# ───────────────────────────────────────────────

def step_prepare_dpo() -> None:
    import json
    from pathlib import Path
    from IMPersona2.prepare_dpo_dataset import convert_to_sharegpt_dpo, deduplicate

    eval_dir = Path("eval_output")
    if not eval_dir.exists():
        print("  ERROR: No eval output found. Run steps 5-6 first.")
        return

    dpo_files = sorted(eval_dir.glob("*/dpo_dataset.json"))
    dpo_files = [f for f in dpo_files if f.stat().st_size > 10]

    if not dpo_files:
        print("  ERROR: No DPO datasets found. Run step 6 first.")
        return

    print(f"\n  Found {len(dpo_files)} DPO dataset(s):\n")
    all_examples = []
    for f in dpo_files:
        data = json.loads(f.read_text())
        n = len(data) if isinstance(data, list) else 0
        print(f"    {f.parent.name}/{f.name}: {n} examples")
        if isinstance(data, list):
            all_examples.extend(data)

    if not all_examples:
        print("\n  ERROR: All DPO datasets are empty.")
        return

    print(f"\n  Total: {len(all_examples)} examples")
    if len(dpo_files) > 1:
        all_examples = deduplicate(all_examples)
        print(f"  After dedup: {len(all_examples)} examples")

    # Filter examples that exceed the training cutoff length.
    # DPO sequences include full context + chosen/rejected; examples
    # beyond the cutoff would be truncated and lose multi-turn context
    # that the guideline depends on.
    MAX_EST_TOKENS = 8192
    CHARS_PER_TOKEN = 3.3  # conservative estimate across common tokenizers
    max_chars = int(MAX_EST_TOKENS * CHARS_PER_TOKEN)
    before_len_filter = len(all_examples)
    kept = []
    for ex in all_examples:
        ctx = sum(len(m.get("content", "")) for m in ex.get("_chat_messages_before", []))
        chosen = len(ex.get("chosen", ""))
        rejected = len(ex.get("rejected", ""))
        if ctx + max(chosen, rejected) <= max_chars:
            kept.append(ex)
    dropped = before_len_filter - len(kept)
    if dropped:
        print(f"  Filtered {dropped} examples exceeding ~{MAX_EST_TOKENS} token cutoff")
    all_examples = kept

    has_chat = sum(1 for ex in all_examples if ex.get("_chat_messages_before"))
    if has_chat:
        print(f"\n  Converting to multi-turn sharegpt format ({has_chat}/{len(all_examples)} with context)...")
        combined = convert_to_sharegpt_dpo(all_examples)
    else:
        combined = all_examples

    output_path = eval_dir / "dpo_combined.json"
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n  Saved {len(combined)} examples to {output_path}")

    from collections import Counter
    labels = Counter(ex.get("_run_label") for ex in combined)
    if len(labels) > 1:
        print(f"\n  By run:")
        for label, count in labels.most_common():
            print(f"    {label}: {count}")

    print("\n  Ready for DPO training: python run.py --step train-dpo")


def step_train_dpo() -> None:
    import subprocess
    from pathlib import Path

    default_dataset = Path("eval_output") / "dpo_combined.json"
    dataset = input(f"  DPO dataset [{default_dataset}]: ").strip() or str(default_dataset)
    if not Path(dataset).exists():
        print(f"  ERROR: {dataset} not found. Run step 7 first.")
        return

    adapter = input("  Adapter name to create [adapter-dpo]: ").strip() or "adapter-dpo"
    init_adapter = input("  Continue from existing adapter (blank = base model): ").strip()

    cmd = [sys.executable, "run_dpo_worker.py", "--dataset", dataset, "--adapter", adapter]
    if init_adapter:
        cmd += ["--init-adapter", init_adapter]

    print(f"\n  Running: {' '.join(cmd)}")
    print("  (Other hyperparameters: python run_dpo_worker.py --help)\n")
    subprocess.run(cmd)


def step_memory() -> None:
    import subprocess

    print("\n  Building memory corpus from your trained contacts...")
    r = subprocess.run([sys.executable, "scripts/build_memory_corpus.py"])
    if r.returncode != 0:
        return

    from IMPersona2.model_config import model_for
    resp = input(f"\n  Embed the corpus with {model_for('embeddings')} (OpenAI, ~$0.30)? [Y/n] ").strip().lower()
    if resp in ("", "y"):
        subprocess.run([sys.executable, "scripts/build_memory_index.py"])


def step_arena() -> None:
    from arena import DEFAULT_DPO_ADAPTER, run_arena

    adapter = input(f"  DPO adapter to evaluate [{DEFAULT_DPO_ADAPTER}]: ").strip() or DEFAULT_DPO_ADAPTER
    run_arena(dpo_adapter=adapter)


# ───────────────────────────────────────────────
# Main — Interactive Step-by-Step
# ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Impersona-Env — Interactive Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step",
        choices=["export", "select", "generate", "train", "eval", "fidelity", "dpo",
                 "prepare-dpo", "train-dpo", "memory", "arena"],
        default=None,
        help="Jump directly to a specific step",
    )
    args = parser.parse_args()

    print()
    print("  Impersona-Env — Pipeline CLI")
    print("  " + "─" * 40)
    print()

    if args.step:
        steps = {"export": step_export, "select": step_select,
                 "generate": step_generate, "train": step_train,
                 "eval": step_eval, "fidelity": step_fidelity, "dpo": step_dpo,
                 "prepare-dpo": step_prepare_dpo, "train-dpo": step_train_dpo,
                 "memory": step_memory, "arena": step_arena}
        fn = steps[args.step]
        fn(force=True) if args.step in ("export", "select", "generate", "train") else fn()
        return

    # Step 1: Export
    has_export = MAC_EXPORT_DIR.exists() and any(MAC_EXPORT_DIR.glob("*.txt"))
    if _ask_proceed("Step 1: Export iMessages", "Export conversations and contacts from your Mac.", done=has_export):
        step_export(force=has_export)

    # Step 2: Select
    has_selection = load_selection() is not None
    if _ask_proceed("Step 2: Select Conversations", "Choose which contacts to train on.", done=has_selection):
        step_select(force=has_selection)

    # Step 3: Generate
    gen_statuses = get_generation_status()
    has_datasets = any(s["status"] == "done" for s in gen_statuses)
    if _ask_proceed("Step 3: Generate Dataset", "Create training data with LLM annotation.", done=has_datasets):
        step_generate(force=has_datasets)

    # Step 4: Train
    train_status = get_training_status()
    train_done = train_status.get("stage") == "completed"
    if _ask_proceed("Step 4: Train on Modal", "Fine-tune Llama 3.1 8B with LoRA on cloud GPUs.", done=train_done):
        step_train(force=train_done)

    # Step 5: Evaluate
    eval_status = get_eval_status()
    eval_done = eval_status.get("stage") == "completed"
    if _ask_proceed("Step 5: Evaluate", "Run automated conversations and judge coherence.", done=eval_done):
        step_eval()

    # Step 5b: Fidelity evals (optional)
    if _ask_proceed("Step 5b: Fidelity Evals (optional)",
                    "Score the user model's style and fact recall against your real data."):
        step_fidelity()

    # Step 6: DPO Dataset
    from pathlib import Path
    eval_dir = Path("eval_output")
    has_dpo = eval_dir.exists() and any(
        (d / "dpo_dataset.json").exists() for d in eval_dir.iterdir() if d.is_dir()
    )
    if _ask_proceed("Step 6: Generate DPO Dataset", "Extract disagreements into preference pairs for DPO training.", done=has_dpo):
        step_dpo()

    # Step 7: Prepare Combined DPO Dataset
    has_combined = eval_dir.exists() and (eval_dir / "dpo_combined.json").exists()
    if _ask_proceed("Step 7: Prepare DPO Dataset", "Combine, deduplicate, and format DPO pairs for training.", done=has_combined):
        step_prepare_dpo()

    # Step 8: DPO training
    if _ask_proceed("Step 8: Train DPO on Modal", "DPO-train the assistant on the combined pairs."):
        step_train_dpo()

    # Step 9: Memory corpus + index
    from pipeline import MEMORY_CORPUS_DIR
    has_index = (MEMORY_CORPUS_DIR / "manifest.json").exists()
    if _ask_proceed("Step 9: Build Memory Index", "Retrieval corpus for the arena's memory-search baselines.", done=has_index):
        step_memory()

    # Step 10: Arena
    if _ask_proceed("Step 10: Blind Arena", "Chat with two anonymous assistants and vote for the better one."):
        step_arena()

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
