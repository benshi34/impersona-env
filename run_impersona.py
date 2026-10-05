#!/usr/bin/env python3
"""
IMPersona CLI Chat — Talk to your trained IMPersona via the terminal.

Usage:
    python run_impersona.py                        # interactive adapter picker
    python run_impersona.py --adapter adapter-user-model  # specify adapter directly
    python run_impersona.py --list                  # list available adapters
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
TRAIN_RUNS_DIR = WORKSPACE / "train_output"


def discover_adapters() -> list[dict]:
    """Scan train_output for completed runs and return adapter info."""
    adapters = []
    if not TRAIN_RUNS_DIR.exists():
        return adapters

    for run_dir in sorted(TRAIN_RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue

        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        adapters.append({
            "run_dir": run_dir.name,
            "adapter_name": config.get("adapter_name", "unknown"),
            "model_name": config.get("model_name", "unknown"),
            "template": config.get("template") or "",
            "num_examples": config.get("num_examples", "?"),
            "label": config.get("label", run_dir.name),
        })
    return adapters


def pick_adapter(adapters: list[dict]) -> dict | None:
    """Interactive adapter picker."""
    if not adapters:
        print("No training runs found in train_output/.")
        return None

    print("\n  Available adapters:\n")
    for i, a in enumerate(adapters, 1):
        print(f"    {i}. {a['adapter_name']}")
        print(f"       model: {a['model_name']}")
        print(f"       run:   {a['run_dir']}  ({a['num_examples']} examples)")
        print()

    while True:
        try:
            choice = input(f"  Pick adapter [1-{len(adapters)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(adapters):
                return adapters[idx]
        except (ValueError, EOFError):
            pass
        print("  Invalid choice, try again.")


def connect_inference(model_name: str, adapter_name: str, template: str = ""):
    """Create a single persistent Modal inference handle."""
    import modal
    from IMPersona2.model_config import template_for
    from pipeline import _modal_app_name

    template = template or template_for("user_model", model_name)
    app_name = _modal_app_name()

    print(f"  Connecting to Modal app '{app_name}'...")
    Cls = modal.Cls.from_name(app_name, "VLLMInference")
    inference = Cls(
        model_name=model_name,
        adapter_name=adapter_name,
        template=template,
    )
    return inference


def build_system_prompt(persona_name: str, other_name: str,
                        relationship: str | None, goal: str | None) -> str:
    """Build the system prompt (same format as training)."""
    prompt = (
        f"You are role-playing as a person named {persona_name}. "
        f"Stay in character and respond as {persona_name} would. "
        f"When sending multiple messages in quick succession, separate them with <msg>. "
        f"Messages in the conversation are numbered [1], [2], etc. "
        f"To reply to a specific earlier message, start your response with "
        f"<reply:N> where N is that message's number. "
        f"Only use this when directly replying to a specific message, not for general responses."
    )
    if relationship:
        prompt += f" You are talking to {other_name}. Relationship context: {relationship}"
    if goal:
        prompt += f" Context: {goal}"
    return prompt


def build_conversation_input(message: str, other_name: str,
                             history: list[dict]) -> str:
    """Build the conversation input string (same format as training)."""
    lines = ["Current conversation:"]
    msg_num = 1
    for entry in history:
        speaker = "Me" if entry.get("speaker") == "Me" else other_name
        ts = entry.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        lines.append(f"[{msg_num}] [{ts}] {speaker}: {entry['content']}")
        msg_num += 1

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"[{msg_num}] [{now}] {other_name}: {message}")
    msg_num += 1
    lines.append(f"[{msg_num}] [{now}] Me: ")

    return "\n".join(lines)


def parse_response(raw: str) -> dict:
    """Parse raw LLM output into structured response."""
    cot = None
    text = raw
    cot_match = re.match(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if cot_match:
        cot = cot_match.group(1).strip()
        text = cot_match.group(2).strip()

    parts = [p.strip() for p in text.split("<msg>") if p.strip()]
    display = [re.sub(r"<reply:\d+>", "", p).strip() for p in parts]
    display = [d for d in display if d]

    return {
        "response": text,
        "messages": display or [text],
        "cot": cot,
    }


def chat_loop(inference, adapter_name: str, model_name: str):
    """Main interactive chat loop using a persistent Modal handle."""
    from pipeline import load_persona

    persona = load_persona()
    persona_name = persona.get("name", "User") if persona else "User"

    other_name = input(f"  Your name (who {persona_name} is talking to) [Friend]: ").strip() or "Friend"
    relationship = input("  Relationship context (optional, press Enter to skip): ").strip() or None
    goal = input("  Conversation goal/context (optional, press Enter to skip): ").strip() or None

    system_prompt = build_system_prompt(persona_name, other_name, relationship, goal)

    print(f"\n{'─' * 60}")
    print(f"  Chatting with {persona_name}'s IMPersona")
    print(f"  Adapter: {adapter_name} | Model: {model_name}")
    print(f"  Type 'quit' or 'exit' to end. 'clear' to reset history.")
    print(f"  'cot' to toggle chain-of-thought display.")
    print(f"{'─' * 60}\n")

    history: list[dict] = []
    show_cot = False

    # Warmup: first call wakes the container & loads the model
    print("  Warming up (first call loads the model, may take 30-60s)...", end="", flush=True)
    try:
        inference.generate.remote(
            system_prompt="warmup",
            conversation_input="Current conversation:\n[1] [2026-01-01 00:00:00] Friend: hi\n[2] [2026-01-01 00:00:01] Me: ",
        )
        print(" ready!\n")
    except Exception as e:
        print(f"\n  Warmup issue: {e}\n  First real message may be slow.\n")

    while True:
        try:
            user_input = input(f"  {other_name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "clear":
            history.clear()
            print("  (history cleared)\n")
            continue
        if user_input.lower() == "cot":
            show_cot = not show_cot
            print(f"  (chain-of-thought display: {'on' if show_cot else 'off'})\n")
            continue

        conversation_input = build_conversation_input(user_input, other_name, history)

        try:
            raw = inference.generate.remote(
                system_prompt=system_prompt,
                conversation_input=conversation_input,
            )
        except Exception as e:
            print(f"  [error] {str(e)[:300]}\n")
            continue

        result = parse_response(raw)

        if show_cot and result.get("cot"):
            print(f"  [thinking] {result['cot']}")

        messages = result["messages"]
        for msg in messages:
            print(f"  {persona_name}: {msg}")
        print()

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history.append({"speaker": other_name, "content": user_input, "timestamp": now})
        for msg in messages:
            history.append({"speaker": "Me", "content": msg, "timestamp": now})


def main():
    parser = argparse.ArgumentParser(
        description="Chat with your trained IMPersona model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python run_impersona.py\n"
               "  python run_impersona.py --adapter adapter-user-model\n"
               "  python run_impersona.py --adapter adapter-user-model --model <base-model-id>\n"
               "  python run_impersona.py --list\n",
    )
    parser.add_argument("--adapter", type=str, help="Adapter name (e.g. adapter-user-model)")
    parser.add_argument("--model", type=str, help="Base model name (default: from run config)")
    parser.add_argument("--list", action="store_true", help="List available adapters and exit")
    args = parser.parse_args()

    print("\n  ╔══════════════════════════════════╗")
    print("  ║      IMPersona CLI Chat          ║")
    print("  ╚══════════════════════════════════╝")

    adapters = discover_adapters()

    if args.list:
        if not adapters:
            print("\n  No training runs found.")
        else:
            print()
            for a in adapters:
                print(f"  {a['adapter_name']:30s}  {a['model_name']}")
        print()
        sys.exit(0)

    if args.adapter:
        match = next((a for a in adapters if a["adapter_name"] == args.adapter), None)
        template = ""
        if match:
            adapter_name = match["adapter_name"]
            model_name = args.model or match["model_name"]
            template = "" if args.model else match["template"]
        else:
            print(f"\n  Adapter '{args.adapter}' not found in local runs, using it anyway.")
            adapter_name = args.adapter
            from IMPersona2.model_config import model_for
            model_name = args.model or model_for("user_model", "base")
    else:
        picked = pick_adapter(adapters)
        if not picked:
            sys.exit(1)
        adapter_name = picked["adapter_name"]
        model_name = args.model or picked["model_name"]
        template = "" if args.model else picked["template"]

    print(f"\n  Using adapter: {adapter_name}")
    print(f"  Base model:    {model_name}\n")

    inference = connect_inference(model_name, adapter_name, template)
    chat_loop(inference, adapter_name, model_name)
    print("  Goodbye!\n")


if __name__ == "__main__":
    main()
