#!/usr/bin/env python3
"""
Impersona-Env — Blind Arena (terminal)
======================================

Head-to-head comparison of assistant configs in your normal chat use.
Each battle draws two configs at random; you chat with both side by side
without knowing which is which, then vote.

  untrained          the untrained assistant                — floor baseline
  untrained-memory   the untrained assistant + memory search — retrieval baseline
  dpo                the trained assistant (your DPO adapter)
  large-memory       a larger model + memory search          — scale + retrieval

Model ids come from configs/models.yaml (`assistant`, `arena`). The untrained
and large configs run on OpenRouter (OPENROUTER_API_KEY); `dpo` runs on the
deployed Modal app. The memory-search configs need the index built via
`python run.py --step memory`.

Verdicts are base64-encoded into .arena_data/<session>.bin so you don't see
win rates while collecting; decode with `python scripts/decode_arena.py`.

Usage:
    python arena.py
    python arena.py --dpo-adapter dpo-pipeline-v1-r3-adapter
    python arena.py --models untrained,untrained-memory   # skip Modal
    python arena.py --show-tools                          # print retrievals
"""

import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
ARENA_RESULTS_DIR = WORKSPACE / ".arena_data"
DEFAULT_DPO_ADAPTER = "adapter-dpo"  # run.py --step train-dpo default


# ───────────────────────────────────────────────
# Memory-search tool (retrieval baseline)
# ───────────────────────────────────────────────

SEARCH_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": (
            "Search the user's past iMessage conversations for context relevant "
            "to the current message. Returns the top-5 most semantically similar "
            "conversation snippets, each tagged with date and contact. Use this "
            "whenever the user references something specific (people, past events, "
            "relationships, preferences) that would help you ground your response."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to search for. Use natural language describing the "
                        "situation or topic — e.g. 'career change advice' or "
                        "'conversations with Sarah about her job'. Be specific."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

SEARCH_SYSTEM_PROMPT = (
    "You are an AI assistant. You have a tool, `search_memory`, that looks "
    "up context from the user's past conversations. The goal: respond like "
    "someone who already knows them well.\n"
    "\n"
    "SEARCHING\n"
    "Default to searching when the user's message touches their personal "
    "life: specific people, places, or situations; recommendations where "
    "their preferences matter; any question where knowing them would "
    "change the answer. When in doubt, search. Skip only for purely "
    "generic factual questions where context wouldn't matter.\n"
    "\n"
    "USING RESULTS — THE CARDINAL RULE\n"
    "From the user's perspective, you only know what they've said in THIS "
    "exchange. Everything else — every fact you got from retrieval — is "
    "invisible to them. They have no idea you have access to their "
    "history.\n"
    "\n"
    "Use retrieved knowledge to inform your reply. BE SPECIFIC — the whole "
    "point of searching is to give an answer that's noticeably more "
    "informed than a vanilla assistant's. If your reply collapses into a "
    "generic framework or a list of textbook factors, you've failed: the "
    "context was wasted.\n"
    "\n"
    "But: never reveal HOW you know the things you know. Don't attribute "
    "retrieved facts to the user as if they told you. The trick is to "
    "frame specific knowledge as your own observation, inference, or "
    "opinion — never as recall of something they disclosed.\n"
    "\n"
    "  ❌ \"You mentioned WLB matters to you\"\n"
    "  ❌ \"You've been thinking about NYC\"\n"
    "  ❌ \"You said the new role is at a startup\"\n"
    "  ❌ \"Based on what I know about you…\"\n"
    "  ❌ Generic fallback: \"Is the new role demanding more hours?\"\n"
    "       (too cautious — you HAVE specific knowledge here; use it)\n"
    "  ✅ \"WLB sounds like a real factor for this one — if the new place "
    "has one of those late-night startup cultures, that's a meaningful cost.\"\n"
    "  ✅ \"If this pulls you out of NYC, that's worth weighing — being "
    "near the people you actually want around adds up.\"\n"
    "\n"
    "The test for the attribution leak: if you find yourself writing \"you "
    "mentioned X\", \"you said Y\", or \"you've been Z-ing\" about "
    "anything the user hasn't told you IN THIS MESSAGE — stop and rephrase "
    "as your own observation or advice that USES the knowledge without "
    "crediting them with disclosing it.\n"
    "\n"
    "Also don't surface the retrieval mechanism itself: no mentions of "
    "the search tool, no citations of past conversations by date or "
    "contact, no quotes of past messages, no third parties from retrieved "
    "chats.\n"
    "\n"
    "If retrieved results aren't actually relevant to the current "
    "question (just keyword matches), ignore them and answer normally. "
    "Don't force-fit context.\n"
    "\n"
    "Think of yourself as a thoughtful friend who knows the user well. A "
    "friend doesn't say \"I recall last March you told me…\" — they just "
    "give better-informed answers because they know you."
)


def build_arena_models(dpo_adapter: str = DEFAULT_DPO_ADAPTER) -> list[dict]:
    """The four arena configs. Model ids come from configs/models.yaml."""
    from IMPersona2.model_config import model_for, template_for

    baseline = model_for("arena", "baseline")
    large_baseline = model_for("arena", "large_baseline")
    assistant = model_for("assistant", "base")
    return [
        # The untrained assistant, served via OpenRouter (`arena.baseline`). The
        # trained config runs on Modal, so serving precision can differ slightly
        # — acceptable for a floor baseline.
        {
            "id": "untrained",
            "provider": "openrouter",
            "display": "Assistant (untrained)",
            "openrouter_model": baseline,
            "system_prompt": "You are a helpful assistant.",
        },
        # Same base model plus search_memory over the corpus IMPersona trained on.
        {
            "id": "untrained-memory",
            "provider": "openrouter",
            "display": "Assistant + memory search",
            "openrouter_model": baseline,
            "system_prompt": SEARCH_SYSTEM_PROMPT,
            "tools": [SEARCH_MEMORY_TOOL],
        },
        # The trained assistant. Stays on Modal because OpenRouter doesn't
        # host custom LoRA adapters.
        {
            "id": "dpo",
            "provider": "modal",
            "display": "Assistant + DPO",
            "modal_model": assistant,
            "modal_template": template_for("assistant", assistant),
            "modal_adapter": dpo_adapter,
            "modal_enable_thinking": bool(model_for("assistant", "thinking")),
            "system_prompt": "You are a helpful assistant.",
        },
        # Same scaffold as untrained-memory with a much larger base model: does
        # targeted DPO beat raw scale with the same retrieval available?
        {
            "id": "large-memory",
            "provider": "openrouter",
            "display": "Large model + memory search",
            "openrouter_model": large_baseline,
            "system_prompt": SEARCH_SYSTEM_PROMPT,
            "tools": [SEARCH_MEMORY_TOOL],
        },
    ]


_memory_store = None
_memory_store_lock = threading.Lock()


def _get_memory_store():
    global _memory_store
    with _memory_store_lock:
        if _memory_store is None:
            from pipeline import MEMORY_CORPUS_DIR, MEMORY_CORPUS_FILE
            from IMPersona2.memory_search import MemoryStore
            _memory_store = MemoryStore(MEMORY_CORPUS_FILE, MEMORY_CORPUS_DIR)
    return _memory_store


def _execute_search_memory(args: dict) -> str:
    """Run a search_memory tool call. Returns the string fed back to the model."""
    query = (args or {}).get("query", "").strip()
    if not query:
        return "Error: empty search query."

    try:
        store = _get_memory_store()
    except FileNotFoundError as e:
        return f"Error: memory index not built. {e}"
    except RuntimeError as e:
        return f"Error: memory index is stale or invalid. {e}"

    results = store.search(query, k=5)
    if not results:
        return f"No past conversations found for query: {query!r}"

    blocks = [f"Top {len(results)} retrieved snippets for {query!r}:"]
    for i, r in enumerate(results, 1):
        blocks.append(
            f"\n[Result {i} · similarity={r['score']:.2f} · "
            f"{r['contact']} · {r['date_start']}..{r['date_end']}]\n"
            f"{r['text']}"
        )
    return "\n".join(blocks)


def _route_tool_call(name: str, args: dict) -> str:
    if name == "search_memory":
        return _execute_search_memory(args)
    return f"Error: unknown tool '{name}'"


# ───────────────────────────────────────────────
# Model backends
# ───────────────────────────────────────────────

def _openrouter_client():
    import openai
    return openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


def _complete_openrouter(model: str, messages: list[dict]) -> str:
    # Disable reasoning traces via both OpenRouter's normalized
    # `reasoning` param and vLLM's `chat_template_kwargs`, since we don't
    # control which upstream provider OpenRouter routes to.
    resp = _openrouter_client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=16384,
        extra_body={
            "reasoning": {"enabled": False},
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return resp.choices[0].message.content or ""


def _complete_openrouter_with_tools(
    model: str,
    messages: list[dict],
    tools: list[dict],
    max_tool_iterations: int = 5,
) -> tuple[str, list[dict]]:
    """Multi-turn function calling. Returns (final_text, tool_call_log).

    Thinking must stay ENABLED here: some reasoning models (including the
    default arena baseline) emit function calls from inside their thinking
    trace, so suppressing it (as _complete_openrouter does) stops the model
    from ever calling the tool.
    """
    client = _openrouter_client()
    msgs = list(messages)
    tool_log: list[dict] = []

    for iteration in range(max_tool_iterations + 1):
        kwargs = dict(model=model, messages=msgs, max_tokens=16384)
        # Final pass drops the tools so the model is forced to answer.
        if iteration < max_tool_iterations:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        msg = client.chat.completions.create(**kwargs).choices[0].message
        if not msg.tool_calls:
            return msg.content or "", tool_log

        calls = [
            {
                "id": tc.id or f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc.function.name or "",
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for i, tc in enumerate(msg.tool_calls)
        ]
        msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": calls})

        for tc in calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
            try:
                result = _route_tool_call(name, args)
            except Exception as e:
                result = f"Error executing {name}: {type(e).__name__}: {e}"
            tool_log.append({"name": name, "args": args, "preview": result[:300], "num_chars": len(result)})
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    return "", tool_log


def load_modal_model(info: dict):
    """Instantiate and warm a VLLMPartner container. Blocks until ready."""
    import modal
    from pipeline import _modal_app_name

    Cls = modal.Cls.from_name(_modal_app_name(), "VLLMPartner")
    kwargs = {"model_name": info["modal_model"], "template": info["modal_template"]}
    if info.get("modal_adapter"):
        kwargs["adapter_name"] = info["modal_adapter"]
    if "modal_enable_thinking" in info:
        kwargs["enable_thinking"] = bool(info["modal_enable_thinking"])
    instance = Cls(**kwargs)
    instance.generate.remote(
        system_prompt="You are a helpful assistant.",
        messages_json=json.dumps([{"role": "user", "content": "Hi"}]),
        max_tokens=4,
        temperature=0.7,
    )
    return instance


def _complete_modal(instance, messages: list[dict]) -> str:
    system = next((m["content"] for m in messages if m["role"] == "system"), "You are a helpful assistant.")
    chat = [m for m in messages if m["role"] != "system"]
    return instance.generate.remote(
        system_prompt=system,
        messages_json=json.dumps(chat),
        max_tokens=16384,
        temperature=0.7,
    )


def respond(info: dict, history: list[dict], modal_instances: dict) -> tuple[str, list[dict]]:
    """Get one assistant reply for a config. Returns (text, tool_call_log)."""
    messages = list(history)
    system_prompt = (info.get("system_prompt") or "").strip()
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    if info["provider"] == "openrouter" and info.get("tools"):
        return _complete_openrouter_with_tools(info["openrouter_model"], messages, info["tools"])
    if info["provider"] == "openrouter":
        return _complete_openrouter(info["openrouter_model"], messages), []
    if info["provider"] == "modal":
        return _complete_modal(modal_instances[info["id"]], messages), []
    raise RuntimeError(f"Unknown provider: {info['provider']}")


# ───────────────────────────────────────────────
# Battle loop
# ───────────────────────────────────────────────

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _print_reply(label: str, text: str, tool_log: list[dict], show_tools: bool):
    print(f"\n  ── Assistant {label} " + "─" * 44)
    if show_tools:
        for t in tool_log:
            print(f"  [tool] {t['name']}({json.dumps(t['args'], ensure_ascii=False)}) → {t['num_chars']} chars")
    for line in _strip_think(text).splitlines() or [""]:
        print(f"  {line}")


def save_result(session: dict, preference: str) -> Path:
    result = {
        "session_id": session["id"],
        "model_a": session["model_a"],
        "model_b": session["model_b"],
        "history_a": session["history_a"],
        "history_b": session["history_b"],
        "preference": preference,
        "winner": (
            session["model_a"]["id"] if preference == "a"
            else session["model_b"]["id"] if preference == "b"
            else preference
        ),
        "turns": session["turns"],
        "created_at": session["created_at"],
        "decided_at": datetime.now().isoformat(),
    }
    ARENA_RESULTS_DIR.mkdir(exist_ok=True)
    path = ARENA_RESULTS_DIR / f"{session['id']}.bin"
    path.write_bytes(base64.b64encode(json.dumps(result, ensure_ascii=False).encode("utf-8")))
    return path


def run_battle(models: list[dict], modal_instances: dict, show_tools: bool) -> bool:
    """Run one battle. Returns False if the user wants to quit the arena."""
    a, b = random.sample(models, 2)
    session = {
        "id": str(uuid.uuid4()),
        "model_a": a,
        "model_b": b,
        "history_a": [],
        "history_b": [],
        "created_at": datetime.now().isoformat(),
        "turns": 0,
    }

    print("\n" + "=" * 60)
    print("  New battle — two anonymous assistants. Type your message.")
    print("  Commands: /a /b /tie /bad to vote and end · /skip · /quit")
    print("=" * 60)

    while True:
        try:
            msg = input("\n  you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if not msg:
            continue

        cmd = msg.lower()
        if cmd in ("/quit", "/q"):
            return False
        if cmd == "/skip":
            print("  Skipped (not recorded).")
            return True
        vote = {"/a": "a", "/b": "b", "/tie": "tie", "/bad": "both_bad"}.get(cmd)
        if vote:
            if session["turns"] == 0:
                print("  Send at least one message before voting.")
                continue
            save_result(session, vote)
            print(f"  Recorded. Battles so far: {len(list(ARENA_RESULTS_DIR.glob('*.bin')))}")
            return True

        session["history_a"].append({"role": "user", "content": msg})
        session["history_b"].append({"role": "user", "content": msg})
        session["turns"] += 1

        print("  (waiting on both assistants...)", flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(respond, a, session["history_a"], modal_instances)
            fb = pool.submit(respond, b, session["history_b"], modal_instances)
            replies = []
            for fut in (fa, fb):
                try:
                    replies.append(fut.result())
                except Exception as e:
                    replies.append((f"[error: {type(e).__name__}: {e}]", []))

        for side, (text, tool_log) in zip(("a", "b"), replies):
            session[f"history_{side}"].append({"role": "assistant", "content": text})
            _print_reply(side.upper(), text, tool_log, show_tools)


def run_arena(model_ids: list[str] | None = None, dpo_adapter: str = DEFAULT_DPO_ADAPTER,
              show_tools: bool = False) -> None:
    from dotenv import load_dotenv
    load_dotenv(WORKSPACE / ".env")

    all_models = build_arena_models(dpo_adapter)
    if model_ids:
        known = {m["id"] for m in all_models}
        unknown = [m for m in model_ids if m not in known]
        if unknown:
            print(f"  ERROR: unknown model id(s): {', '.join(unknown)}")
            print(f"  Available: {', '.join(sorted(known))}")
            return
        models = [m for m in all_models if m["id"] in model_ids]
    else:
        models = all_models

    if len(models) < 2:
        print("  ERROR: need at least 2 models for a battle.")
        return

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if any(m["provider"] == "openrouter" for m in models) and (not key or key.startswith("your-")):
        print("  ERROR: OPENROUTER_API_KEY is not set in .env.")
        return

    modal_instances = {}
    for m in models:
        if m["provider"] == "modal":
            print(f"  Loading {m['display']} ({m['modal_adapter']}) on Modal — 2-5 min on cold start...", flush=True)
            try:
                modal_instances[m["id"]] = load_modal_model(m)
            except Exception as e:
                print(f"  ERROR loading {m['id']}: {type(e).__name__}: {e}")
                if "auth" in type(e).__name__.lower() or "token" in str(e).lower():
                    print("  Log in to Modal first: `modal setup`.")
                else:
                    print("  Deploy the app first: `modal deploy modal_app.py`.")
                print("  To leave this config out: python arena.py --models untrained,untrained-memory,large-memory")
                return
            print("  Ready.")

    print(f"\n  Arena pool: {len(models)} configs. Identities stay hidden until you decode results.")
    while run_battle(models, modal_instances, show_tools):
        pass
    print("\n  Decode results with: python scripts/decode_arena.py")


def main():
    parser = argparse.ArgumentParser(
        description="Blind head-to-head arena for assistant configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated config ids to include (default: all four)")
    parser.add_argument("--dpo-adapter", type=str, default=DEFAULT_DPO_ADAPTER,
                        help=f"DPO adapter name on the Modal volume (default: {DEFAULT_DPO_ADAPTER})")
    parser.add_argument("--show-tools", action="store_true",
                        help="Print memory-search tool calls under each reply")
    args = parser.parse_args()

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()] or None
    run_arena(model_ids, args.dpo_adapter, args.show_tools)


if __name__ == "__main__":
    sys.exit(main())
