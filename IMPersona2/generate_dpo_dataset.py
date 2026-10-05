#!/usr/bin/env python3
"""
Generate a DPO dataset from an IMPersona eval run.

The preference pairs target the **AI Assistant** — the model whose responses
the IMPersona (simulated user) reacted to.  When the IMPersona's
chain-of-thought expresses disagreement with the assistant's reply, we ask the
same assistant model to rewrite that reply incorporating the persona's feedback.

Algorithm:
1. Load a `results.json` eval run that already has per-turn disagreement
   labels saved (from eval.py's disagreement evaluator).
2. For each flagged turn, reconstruct the AI assistant's multi-turn chat
   state up to the problematic response.
3. Ask the assistant model to rewrite its response given the persona's
   internal feedback.
4. The rewritten assistant response = `chosen`; the original = `rejected`.

NOTE: The eval run must contain per-turn disagreement details (saved by the
disagreement judge). If a run lacks them, re-run the judges first:
    python -m IMPersona2.eval --eval-only <run_dir>

The output is a flat JSON list in LlamaFactory's default alpaca-style ranking
shape:
    {
      "instruction": "...",   # AI assistant system prompt
      "input": "...",         # user message(s) the assistant should respond to
      "chosen": "...",        # improved assistant response
      "rejected": "..."      # original assistant response
    }

Extra `_...` metadata fields are also included for debugging and filtering.

Usage:
    # Rewrite with the same model that was the AI partner in the eval run
    python -m IMPersona2.generate_dpo_dataset \
        --run /path/to/eval_run_dir

    # Override the rewrite model
    python -m IMPersona2.generate_dpo_dataset \
        --run /path/to/eval_run_dir \
        --rewrite-api-model <api-model-id>
"""

import argparse
import json
import os
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from IMPersona2.model_config import model_for
from IMPersona2.eval import (
    DEFAULT_PERSONA_NAME,
    AIPartnerClient,
    C,
    clean_text_for_ai_partner,
    strip_think_tokens,
)


# Translate model (CoT → observable guideline). Runs through AIPartnerClient, so
# any supported provider works. Configured under `translate` in
# configs/models.yaml (model, reasoning, base_url); TRANSLATE_MODEL /
# TRANSLATE_REASONING env vars override.
TRANSLATE_MODEL = model_for("translate")
TRANSLATE_REASONING = model_for("translate", "reasoning") or None
TRANSLATE_BASE_URL = model_for("translate", "base_url") or None


# ─────────────────────────────────────────────
# Step 1: Translate private CoT → observable guideline
# ─────────────────────────────────────────────

TRANSLATE_SYSTEM_PROMPT = """\
You are an expert at translating private user feedback into actionable \
improvement guidelines for an AI assistant.

You will receive:
1. A multi-turn conversation between the user and an AI ASSISTANT.
2. the user's private internal reaction (chain-of-thought) to the \
assistant's last response — thoughts the assistant CANNOT see.
3. A judge's brief summary of the issue.

Your job: produce a short, concrete **improvement guideline** that tells the \
assistant what to do differently — referencing ONLY signals that are \
**observable in the conversation itself** (message content, tone, length, \
repetition patterns, topic flow, explicit requests from the user, etc.).

CRITICAL RULES:
- Do NOT reveal the user's private thoughts, emotions, or internal state.
- Do NOT say "the user felt…" or "the user was thinking…".
- Reference only what a careful reader of the conversation could notice.
- Be specific and actionable: say what the assistant should do differently.
- Refer to the user as "the user" — do not invent or assume a name.
- 2-4 sentences max.
- Output ONLY the guideline — no labels, no preamble.

QUALITY GATE — output [SKIP] followed by a brief reason instead of a \
guideline if ANY of the following apply:
1. The assistant's response already adequately addresses the concern raised in \
the private reaction — there is no meaningful improvement to make. A rewrite \
would produce essentially the same response.
2. the user's private reaction is based on a factual misunderstanding \
(e.g. they think they already provided information they didn't) and the \
assistant's response is actually correct. Training on this would teach the \
assistant to validate incorrect beliefs.
3. The disagreement is purely about the user's private emotional state \
with no observable conversational signal the assistant could have acted on — \
there are no cues in the messages themselves that would let the assistant \
respond differently.
4. The conversation context is too incoherent, nonsensical, or off-topic to \
produce a useful training example.
5. The feedback is too vague or minor to produce a meaningfully different \
rewrite (e.g. mild annoyance with no specific actionable change).
6. The conversation contains sexually explicit, pornographic, or otherwise \
inappropriate content (e.g. the user requesting sexual acts, sending \
explicit messages, or engaging in sexual roleplay). These conversations are \
not suitable for training data regardless of the assistant's response quality.
7. The assistant's response contains visible chain-of-thought reasoning, \
internal planning, or self-correction monologue (e.g. "I need to…", \
"*Wait, let me reconsider*", "Step 1:") instead of a direct response to \
the user. A rewrite based on this would produce unusable training data.
8. The disagreement stems from the user treating the assistant as \
human — expecting it to have personal experiences, a physical existence, or \
firsthand knowledge it cannot have. If the assistant correctly identified \
itself as an AI and the user's dissatisfaction is rooted in that \
clarification or in a perceived contradiction between disclaiming human \
experience and still offering guidance, the example should be skipped. \
Training on these would teach the assistant to either pretend to be human \
or to withhold helpful information after a disclaimer — both are harmful.
"""

TRANSLATE_USER_TEMPLATE = """\
CONVERSATION:
{chat_history}

ASSISTANT'S RESPONSE BEING EVALUATED:
{original_response}

THE USER'S PRIVATE INTERNAL REACTION (not visible to assistant):
{chain_of_thought}

JUDGE'S SUMMARY:
{judge_reason}

Write an improvement guideline for the assistant that references only \
observable conversation signals. Do not reveal the user's private thoughts."""


# ─────────────────────────────────────────────
# Step 2: Rewrite the assistant response using the guideline
# ─────────────────────────────────────────────

REWRITE_SYSTEM_PROMPT = """\
You are an AI assistant talking to the user. You will be given:
1. The conversation so far between you and the user.
2. Your previous response to the user's last message.
3. An improvement guideline describing what to do differently, based on \
observable conversation signals.

Rewrite your response to follow the guideline while staying helpful and natural.

Requirements:
- Respond directly to the user's last message.
- Follow the improvement guideline.
- Keep a natural, conversational tone.
- Do not mention the guideline, the rewrite process, or that anything was changed.
- Output ONLY the rewritten response — no preamble, no explanation.
"""

REWRITE_USER_TEMPLATE = """\
Here is the conversation so far with the user:
{chat_history}

Your previous response:
{original_response}

Improvement guideline:
{guideline}

Rewrite your response following this guideline. Output only the improved response.
"""


@dataclass
class FlaggedAssistantTurn:
    """Represents an AI Assistant response that the persona disagreed with."""
    turn_index: int                          # index among CoT persona turns (matches judge)
    # The AI assistant's chat state up to (but not including) the flagged response
    chat_messages_before: List[Dict[str, str]]  # role/content dicts for multi-turn
    ai_partner_system_prompt: str
    original_ai_response: Optional[str]      # the assistant response being criticized (None for opening turns)
    user_message_before: str                 # the user message the assistant was responding to
    persona_chain_of_thought: str            # persona's internal reaction (the feedback)
    persona_visible_response: str            # what the persona actually replied
    transcript_ai_msg_index: int             # index in the transcript of the AI message (-1 if N/A)


def resolve_results_path(path: str) -> str:
    """Accept either a run directory or a direct path to results.json."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return os.path.join(path, "results.json")
    return path


def load_results_bundle(path: str) -> Dict:
    """Load the eval results bundle from disk."""
    results_path = resolve_results_path(path)
    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"results.json not found at {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)



def normalize_text(text: str) -> str:
    """Whitespace-insensitive comparison helper."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Format multi-turn chat messages into a readable block for the rewrite prompt."""
    lines = []
    for msg in messages:
        role = msg["role"].upper()
        lines.append(f"[{role}]: {msg['content']}")
    return "\n\n".join(lines)


def extract_flagged_assistant_turns(
    conversation: Dict,
    persona_name: str,
) -> List[FlaggedAssistantTurn]:
    """
    Walk the transcript and build one entry per CoT-bearing persona message,
    matching the exact same indexing that eval.py's disagreement judge uses.

    CRITICAL: The disagreement judge iterates over *individual* transcript
    messages (not merged groups) and assigns turn_index 0, 1, 2, ... to each
    message that has a chain_of_thought — including the very first persona
    message (which has no preceding AI response).  We MUST mirror that
    counting so that ``turns[N]`` here corresponds to the judge's
    ``turn_index: N``.

    For turns where there is no preceding AI response (e.g. the persona's
    opening message), ``original_ai_response`` is set to None and those
    turns are skipped during rewrite but still occupy their index slot.
    """
    transcript = conversation.get("messages", [])
    ai_partner_system_prompt = (
        conversation.get("scenario", {}).get("ai_partner_system_prompt", "")
        or "You are an AI assistant."
    )

    # Reconstruct the multi-turn chat the AI partner saw.
    chat_messages: List[Dict[str, str]] = []  # role/content pairs (grows as we walk)
    turns: List[FlaggedAssistantTurn] = []
    turn_index = 0

    # State trackers for the most recent AI response
    last_ai_response: Optional[str] = None
    last_ai_msg_index: Optional[int] = None
    chat_before_last_ai: List[Dict[str, str]] = []
    user_msg_that_prompted_ai: Optional[str] = None

    # Track which persona messages we've already added to chat_messages
    # (we merge consecutive persona messages into one user turn)
    pending_persona_parts: List[str] = []

    def _flush_persona_to_chat():
        """Merge buffered persona messages into one user turn in chat_messages."""
        nonlocal pending_persona_parts
        if pending_persona_parts:
            combined = "<msg>".join(p.strip() for p in pending_persona_parts if p.strip())
            cleaned = clean_text_for_ai_partner(combined) if combined else ""
            if cleaned:
                chat_messages.append({"role": "user", "content": cleaned})
            pending_persona_parts = []

    for i, msg in enumerate(transcript):
        speaker = msg.get("speaker", "")

        if speaker in (persona_name, "Me"):
            cot = msg.get("chain_of_thought")
            content = msg.get("content", "")
            pending_persona_parts.append(content)

            if cot:
                # This message has CoT — record a turn entry to keep
                # indices aligned with the judge.
                if last_ai_response is not None:
                    turns.append(
                        FlaggedAssistantTurn(
                            turn_index=turn_index,
                            chat_messages_before=[dict(m) for m in chat_before_last_ai],
                            ai_partner_system_prompt=ai_partner_system_prompt,
                            original_ai_response=last_ai_response,
                            user_message_before=user_msg_that_prompted_ai or "",
                            persona_chain_of_thought=cot,
                            persona_visible_response=content,
                            transcript_ai_msg_index=last_ai_msg_index or 0,
                        )
                    )
                else:
                    # Opening persona message — no AI response to rewrite.
                    # Still occupy the index slot so numbering stays aligned.
                    turns.append(
                        FlaggedAssistantTurn(
                            turn_index=turn_index,
                            chat_messages_before=[],
                            ai_partner_system_prompt=ai_partner_system_prompt,
                            original_ai_response=None,  # type: ignore[arg-type]
                            user_message_before="",
                            persona_chain_of_thought=cot,
                            persona_visible_response=content,
                            transcript_ai_msg_index=-1,
                        )
                    )
                turn_index += 1

        else:
            # AI Assistant message — first flush any pending persona messages
            _flush_persona_to_chat()

            ai_content = msg.get("content", "")
            # Strip <think>...</think> blocks from the chat context.
            # Thinking models never see their own previous thoughts during
            # inference, so the DPO training context must match that —
            # otherwise we create a distribution mismatch.
            ai_content_for_context = strip_think_tokens(ai_content)

            # Snapshot chat state *before* appending this AI response
            chat_before_last_ai = [dict(m) for m in chat_messages]
            user_msg_that_prompted_ai = None
            for prev in reversed(chat_messages):
                if prev["role"] == "user":
                    user_msg_that_prompted_ai = prev["content"]
                    break
            # Keep raw content for rewriting (original_ai_response);
            # use stripped content for the conversation context that
            # ends up in _chat_messages_before.
            last_ai_response = ai_content
            last_ai_msg_index = i
            chat_messages.append({"role": "assistant", "content": ai_content_for_context})

    return turns



def choose_conversations(
    conversations: List[Dict],
    scenario_ids: Optional[List[str]] = None,
    limit_conversations: Optional[int] = None,
) -> List[Dict]:
    """Apply simple scenario / count filters."""
    selected = conversations
    if scenario_ids:
        wanted = set(scenario_ids)
        selected = [conv for conv in selected if conv.get("scenario", {}).get("id") in wanted]
    if limit_conversations is not None:
        selected = selected[:limit_conversations]
    return selected


def load_saved_disagreement_details(results_bundle: Dict) -> Dict[str, List[Dict]]:
    """
    Load per-turn disagreement classifications from the saved evaluations.

    Returns a mapping of scenario_id → list of per-turn classification dicts
    (each with turn_index, is_disagreement, reason).

    Raises RuntimeError if the run has no saved per-turn details; re-run the
    judges with ``python -m IMPersona2.eval --eval-only <run_dir>``.
    """
    evals = results_bundle.get("evaluations", {})
    disagreement_eval = evals.get("disagreements", {})
    per_conversation = disagreement_eval.get("per_conversation", [])

    if not per_conversation:
        raise RuntimeError(
            "This eval run has no disagreement evaluations at all.\n"
            "Run the evaluations first:\n"
            "  python -m IMPersona2.eval --eval-only <run_dir>"
        )

    details_map: Dict[str, List[Dict]] = {}
    missing_details = []

    for row in per_conversation:
        scenario_id = row.get("scenario_id", "")
        details = row.get("details", [])
        if not details and (row.get("num_disagreements") or 0) > 0:
            missing_details.append(scenario_id)
        details_map[scenario_id] = details

    if missing_details:
        raise RuntimeError(
            f"Per-turn disagreement details are missing for {len(missing_details)} "
            f"conversation(s) that have >0 disagreements:\n"
            f"  {', '.join(missing_details)}\n\n"
            "Re-run the evaluations to save per-turn labels:\n"
            "  python -m IMPersona2.eval --eval-only <run_dir>"
        )

    return details_map


class ModalRewriteClient:
    """Wraps a Modal VLLMPartner container for on-policy rewrites.

    The VLLMPartner runs the same model used as the AI partner in the
    eval run, ensuring DPO rewrites stay on-distribution.
    """

    def __init__(self, model_name: str, adapter_name: str = "", app_name: str = "impersona"):
        import modal

        adapter_note = f" + {adapter_name}" if adapter_name else " (no adapter)"
        print(f"  Connecting to Modal VLLMPartner ({model_name}{adapter_note})...", flush=True)
        print(f"  (First call may take 5-10 min while the GPU container starts)", flush=True)
        Cls = modal.Cls.from_name(app_name, "VLLMPartner")
        # Rewrite quality benefits from chain-of-thought, so we keep
        # enable_thinking=True here even though eval-side partners use False.
        # The rewrite's <think>...</think> block is stripped downstream via
        # strip_think_tokens before it lands in the DPO dataset.
        self._partner = Cls(model_name=model_name, adapter_name=adapter_name, enable_thinking=True)
        self._model_name = model_name
        self._adapter_name = adapter_name
        self._calls = 0
        self._errors = 0
        print(f"  enable_thinking: True (rewrite model thinks before writing)", flush=True)

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
    ) -> str:
        self._calls += 1
        return self._partner.generate.remote(
            system_prompt=system_prompt,
            messages_json=json.dumps(messages),
            max_tokens=max_tokens,
        )

    def get_usage(self) -> Dict:
        return {
            "num_calls": self._calls,
            "total_tokens": 0,
            "estimated_cost_usd": 0,
        }


def resolve_rewrite_model(args, run_metadata: Dict) -> Tuple:
    """Choose the model used to rewrite flagged AI assistant responses.

    Supports API models, local LlamaFactory models, and Modal VLLMPartner.
    """
    # ── Modal mode (on-policy, recommended for DPO) ──
    if args.rewrite_modal:
        model_name = args.rewrite_modal_model or run_metadata.get("ai_partner_model")
        if not model_name:
            raise ValueError(
                "No model name for Modal rewrite. Either set --rewrite-modal-model "
                "or ensure the eval run metadata contains ai_partner_model."
            )
        # Read adapter from metadata (or CLI override) so on-policy rewrites
        # actually use the LoRA-tuned partner, not the base model.
        adapter_name = (
            getattr(args, "rewrite_modal_adapter", None)
            or run_metadata.get("ai_partner_adapter", "")
            or ""
        )
        client = ModalRewriteClient(model_name=model_name, adapter_name=adapter_name, app_name="impersona")
        label_suffix = f" + {adapter_name}" if adapter_name else ""
        return (client, f"Modal VLLMPartner — {model_name}{label_suffix}")

    # ── Local model mode ──
    if args.rewrite_local_model:
        if not args.rewrite_template:
            raise ValueError(
                "--rewrite-template is required when using --rewrite-local-model"
            )
        from llamafactory.chat.chat_model import ChatModel

        print(f"\n{C.CYAN}Loading local rewrite model via LlamaFactory...{C.RESET}")
        print(f"  Model: {args.rewrite_local_model}")
        print(f"  Template: {args.rewrite_template}")

        model_args = {
            "model_name_or_path": args.rewrite_local_model,
            "template": args.rewrite_template,
            "infer_backend": "huggingface",
        }
        if args.rewrite_quantization_bit:
            model_args["quantization_bit"] = args.rewrite_quantization_bit
            print(f"  Quantization: {args.rewrite_quantization_bit}-bit")

        chat_model = ChatModel(model_args)
        print(f"{C.GREEN}✅ Rewrite model ready.{C.RESET}")

        # Build generation kwargs from preset
        gen_kwargs = {}
        if args.rewrite_preset == "qwen":
            gen_kwargs.update(AIPartnerClient.QWEN_GENERATION_DEFAULTS)
            print(f"  Preset: qwen → {gen_kwargs}")

        label = args.rewrite_local_label or args.rewrite_local_model
        return (
            AIPartnerClient(
                model=label,
                chat_model=chat_model,
                generation_kwargs=gen_kwargs,
            ),
            f"Local — {args.rewrite_local_model}",
        )

    # ── API model mode ──
    if args.rewrite_api_model:
        return (
            AIPartnerClient(
                model=args.rewrite_api_model,
                reasoning=args.rewrite_reasoning,
                base_url=getattr(args, "rewrite_base_url", None) or None,
            ),
            args.rewrite_api_model,
        )

    # Default: use the same model that was the AI partner in the eval run
    ai_partner_model = run_metadata.get("ai_partner_model")
    if ai_partner_model:
        return (
            AIPartnerClient(model=ai_partner_model, reasoning=args.rewrite_reasoning),
            ai_partner_model,
        )

    raise ValueError(
        "No rewrite model available. The run metadata has no ai_partner_model, "
        "and no --rewrite-api-model or --rewrite-local-model was specified."
    )


def translate_feedback_to_guideline(
    translate_client: AIPartnerClient,
    turn: FlaggedAssistantTurn,
    judge_reason: str,
    persona_name: str = DEFAULT_PERSONA_NAME,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Translate the persona's private CoT + judge reason into an observable
    behavioral guideline that the rewrite model can use without mind-reading.

    Returns (guideline, None) on success, or (None, skip_reason) if the
    translation model determines this is a bad training example.
    """
    chat_history_text = format_chat_history(turn.chat_messages_before)

    system_prompt = TRANSLATE_SYSTEM_PROMPT.format(persona_name=persona_name)
    user_prompt = TRANSLATE_USER_TEMPLATE.format(
        chat_history=chat_history_text,
        original_response=turn.original_ai_response,
        chain_of_thought=turn.persona_chain_of_thought,
        judge_reason=judge_reason or "(no additional reason)",
        persona_name=persona_name,
    )

    raw = translate_client.generate(
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        # Headroom for reasoning models, which spend the completion
        # budget on hidden reasoning before emitting the guideline.
        max_tokens=4096,
    )
    result = raw.strip()

    if result.startswith("[SKIP]"):
        skip_reason = result[len("[SKIP]"):].strip().lstrip("—:- ").strip()
        return (None, skip_reason or "filtered by translation model")

    return (result, None)


MAX_REWRITE_LENGTH_RATIO = 10.0

VISIBLE_REWRITE_COT_STARTS = (
    "Thinking Process:",
    "The user is ",
    "I need to:",
    "Analyze the Request:",
    "Improvement guideline:",
    "Rewrite your response",
    "Plan:",
)

VISIBLE_REWRITE_COT_MARKERS = (
    "Analyze the Request:",
    "Evaluate the Previous Response:",
    "Improvement guideline:",
    "Rewrite your response following this guideline",
    "Constraints:",
    "Final Plan:",
)


def has_visible_rewrite_cot(text: str) -> bool:
    """Detect rewrites that leaked plain-text chain-of-thought.

    Some models emit reasoning outside <think> tags, so simple tag stripping is
    not enough. Keep this heuristic conservative to avoid throwing away normal
    assistant prose.
    """
    stripped = text.lstrip()
    if stripped.startswith("Thinking Process:"):
        return True

    if stripped.startswith("The user is ") and any(
        marker in stripped for marker in ("I need to:", "Improvement guideline:", "Plan:")
    ):
        return True

    if any(stripped.startswith(prefix) for prefix in VISIBLE_REWRITE_COT_STARTS[2:]):
        return True

    marker_count = sum(marker in stripped for marker in VISIBLE_REWRITE_COT_MARKERS)
    return marker_count >= 2


def rewrite_assistant_response(
    rewrite_client: AIPartnerClient,
    turn: FlaggedAssistantTurn,
    guideline: str,
    persona_name: str = DEFAULT_PERSONA_NAME,
) -> str:
    """Ask the assistant model to rewrite its response given an observable guideline."""
    chat_history_text = format_chat_history(turn.chat_messages_before)

    system_prompt = REWRITE_SYSTEM_PROMPT.format(persona_name=persona_name)
    user_prompt = REWRITE_USER_TEMPLATE.format(
        chat_history=chat_history_text,
        original_response=turn.original_ai_response,
        guideline=guideline,
        persona_name=persona_name,
    )

    rewritten = rewrite_client.generate(
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=4096,
    )
    # Strip <think>...</think> blocks — reasoning models can emit them
    # even when instructed not to.  DPO chosen/rejected must be clean.
    rewritten = strip_think_tokens(rewritten)
    return rewritten.strip()


def ensure_output_path(args, run_path: str) -> str:
    """Resolve the output dataset path."""
    if args.output:
        return os.path.abspath(args.output)

    run_dir = run_path if os.path.isdir(run_path) else os.path.dirname(run_path)

    # Use a distinct filename for local-model rewrites to avoid overwriting
    # API-model results (enables side-by-side comparison)
    if args.rewrite_local_model:
        label = args.rewrite_local_label or "local"
        return os.path.join(run_dir, f"dpo_dataset_{label}.json")

    return os.path.join(run_dir, "dpo_dataset.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a naive DPO dataset from an IMPersona eval run."
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to an eval run directory or a direct path to results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <run_dir>/dpo_dataset.json)",
    )
    parser.add_argument(
        "--rewrite-modal",
        action="store_true",
        help=(
            "Use Modal VLLMPartner for on-policy rewrites (recommended for DPO). "
            "Spins up the same model used as the AI partner in the eval run on a "
            "Modal GPU. Requires 'modal deploy modal_app.py' first."
        ),
    )
    parser.add_argument(
        "--rewrite-modal-model",
        type=str,
        default=None,
        help="Override the Modal model name (default: AI partner model from eval metadata)",
    )
    parser.add_argument(
        "--rewrite-modal-adapter",
        type=str,
        default=None,
        help="Override the Modal partner adapter name (default: ai_partner_adapter from eval metadata, "
             "or no adapter if missing). Use this to do on-policy rewrites with a LoRA-tuned partner.",
    )
    parser.add_argument(
        "--rewrite-api-model",
        type=str,
        default=None,
        help="API model for rewrites (any API model id; see configs/models.yaml)",
    )
    parser.add_argument(
        "--rewrite-reasoning",
        type=str,
        default=None,
        help="Optional reasoning setting for the rewrite API model",
    )
    parser.add_argument(
        "--rewrite-base-url",
        type=str,
        default=model_for("rewrite", "base_url"),
        help=(
            "OpenAI-compatible endpoint for --rewrite-api-model, e.g. "
            "https://openrouter.ai/api/v1 (default: rewrite.base_url in configs/models.yaml)"
        ),
    )
    # Local model rewrite options
    parser.add_argument(
        "--rewrite-local-model",
        type=str,
        default=None,
        help=(
            "Path to a local model for rewrites (loaded via LlamaFactory). "
            "A HuggingFace model id or local path. When set, rewrites are "
            "done in-process instead of via API."
        ),
    )
    parser.add_argument(
        "--rewrite-template",
        type=str,
        default=None,
        help="LlamaFactory chat template for the local rewrite model (e.g. 'qwen3_5_nothink')",
    )
    parser.add_argument(
        "--rewrite-preset",
        type=str,
        default=None,
        choices=["qwen"],
        help="Generation preset for the local rewrite model (e.g. 'qwen')",
    )
    parser.add_argument(
        "--rewrite-quantization-bit",
        type=int,
        default=None,
        choices=[4, 8],
        help="Quantize the local rewrite model to 4-bit or 8-bit",
    )
    parser.add_argument(
        "--rewrite-local-label",
        type=str,
        default=None,
        help="Short label for the local rewrite model (used in output metadata)",
    )
    parser.add_argument(
        "--scenario-ids",
        type=str,
        default=None,
        help="Comma-separated scenario ids to include",
    )
    parser.add_argument(
        "--limit-conversations",
        type=int,
        default=None,
        help="Optional cap on the number of conversations to process",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of DPO examples to emit",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Max parallel rewrite workers (default: 20). Set to 1 for sequential.",
    )
    parser.add_argument(
        "--allow-identical",
        action="store_true",
        help="Keep examples where the rewrite is identical to the original response",
    )
    parser.add_argument(
        "--empty-think-prefix",
        choices=("yes", "no"),
        default="no",
        help=(
            "Prepend an empty '<think>\\n\\n</think>\\n\\n' block to chosen and rejected. "
            "Default: no. Only set to 'yes' if you are training with LF's qwen3_5_nothink "
            "template (which strips think tokens) and want to restore them. For the "
            "recommended setup (qwen3_5 template + enable_thinking=False), leave as 'no' — "
            "the template will prefill the empty think on the prompt side automatically."
        ),
    )
    return parser.parse_args()


def generate_dpo_viewer_html(examples: List[Dict], run_metadata: Dict) -> str:
    """Generate a self-contained HTML viewer for inspecting DPO preference pairs."""
    import html as html_mod
    from collections import Counter

    scenarios = Counter(ex.get("_scenario_name", "unknown") for ex in examples)

    def _script_safe(s: str) -> str:
        """Make a JSON string safe to embed inside an inline <script> block.

        Message content can contain HTML snippets (``<!--``, ``<script``,
        ``</script>`` …) that would otherwise terminate or re-mode the script
        element and dump the raw JSON onto the page. Encoding every ``<``/``>``
        as a unicode escape is still valid JSON and parses back to the same
        characters, so the data is unchanged once the browser reads it.
        """
        return (
            s.replace("<", "\\u003c")
             .replace(">", "\\u003e")
             .replace("\u2028", "\\u2028")
             .replace("\u2029", "\\u2029")
        )

    scenarios_json = _script_safe(json.dumps(dict(scenarios.most_common()), ensure_ascii=False))
    examples_json = _script_safe(json.dumps(examples, ensure_ascii=False))
    run_label = run_metadata.get("run_label") or "unknown"
    run_id = run_metadata.get("run_id") or ""
    persona = run_metadata.get("persona_name", DEFAULT_PERSONA_NAME)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DPO Dataset Viewer — {html_mod.escape(run_label)}</title>
<style>
:root {{
  --bg: #0d1117; --surface: #161b22; --surface2: #21262d;
  --border: #30363d; --text: #e6edf3; --muted: #8b949e;
  --accent: #58a6ff; --green: #3fb950; --red: #f85149;
  --purple: #bc8cff; --yellow: #d29922;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
       background: var(--bg); color: var(--text); line-height:1.6; }}
.container {{ max-width:1400px; margin:0 auto; padding:20px; }}

/* Header */
.header {{ padding:20px 0; border-bottom:1px solid var(--border); margin-bottom:24px; }}
.header h1 {{ font-size:24px; }}
.header h1 span {{ color:var(--accent); }}
.header .meta {{ color:var(--muted); font-size:14px; margin-top:4px; }}

/* Stats */
.stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
               gap:12px; margin-bottom:24px; }}
.stat-card {{ background:var(--surface); border:1px solid var(--border);
             border-radius:8px; padding:14px; }}
.stat-card .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.5px; }}
.stat-card .value {{ font-size:22px; font-weight:600; margin-top:2px; }}

/* Scenario bar chart */
.chart-section {{ background:var(--surface); border:1px solid var(--border);
                  border-radius:8px; padding:20px; margin-bottom:24px; }}
.chart-section h3 {{ font-size:15px; margin-bottom:14px; }}
.bar-row {{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }}
.bar-label {{ width:240px; font-size:13px; color:var(--muted); text-align:right; flex-shrink:0;
              overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.bar-track {{ flex:1; background:var(--surface2); border-radius:4px; height:22px;
              position:relative; overflow:hidden; }}
.bar-fill {{ height:100%; border-radius:4px; background:var(--accent); display:flex;
             align-items:center; padding-left:8px; font-size:11px; font-weight:600;
             min-width:18px; }}
.bar-count {{ width:40px; font-size:13px; color:var(--muted); }}

/* Controls */
.controls {{ display:flex; gap:10px; margin-bottom:20px; flex-wrap:wrap; align-items:center; }}
.controls select, .controls input {{
  background:var(--surface); border:1px solid var(--border);
  color:var(--text); padding:8px 12px; border-radius:6px; font-size:14px;
}}
.controls input {{ flex:1; min-width:200px; }}
.toggle-btn {{
  background:var(--surface2); border:1px solid var(--border);
  color:var(--text); padding:6px 14px; border-radius:6px;
  cursor:pointer; font-size:13px; transition:all .2s;
}}
.toggle-btn.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}

/* Example cards */
.example-list {{ display:flex; flex-direction:column; gap:20px; }}
.example-card {{ background:var(--surface); border:1px solid var(--border);
                 border-radius:8px; overflow:hidden; }}
.example-card:hover {{ border-color:var(--accent); }}
.card-header {{ display:flex; justify-content:space-between; align-items:center;
                padding:12px 16px; background:var(--surface2); border-bottom:1px solid var(--border); }}
.card-header .idx {{ font-weight:600; color:var(--accent); }}
.card-header .scenario {{ color:var(--muted); font-size:13px; }}
.card-body {{ padding:16px; }}
.section-label {{ font-size:11px; text-transform:uppercase; letter-spacing:.8px;
                  color:var(--muted); margin-bottom:6px; margin-top:16px; }}
.section-label:first-child {{ margin-top:0; }}

/* Conversation context */
.context-box {{ background:var(--bg); padding:12px 14px; border-radius:6px;
                font-size:13px; white-space:pre-wrap; word-break:break-word;
                max-height:300px; overflow-y:auto; line-height:1.5; }}
.context-box .msg-me {{ color:var(--green); }}
.context-box .msg-other {{ color:var(--accent); }}
.context-box .msg-ts {{ color:var(--muted); font-size:11px; }}
.system-prompt {{ background:var(--surface2); padding:10px 14px; border-radius:6px;
                  font-size:12px; color:var(--muted); white-space:pre-wrap;
                  word-break:break-word; border-left:3px solid var(--accent);
                  max-height:120px; overflow-y:auto; }}

/* Side-by-side chosen / rejected */
.pair-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:6px; }}
@media (max-width:800px) {{ .pair-grid {{ grid-template-columns:1fr; }} }}
.pair-box {{ padding:12px 14px; border-radius:6px; font-size:14px;
             white-space:pre-wrap; word-break:break-word; line-height:1.5; }}
.pair-box.chosen {{ background:#1a2a1a; border:1px solid #2d4a2d; color:var(--green); }}
.pair-box.rejected {{ background:#2a1a1a; border:1px solid #4a2d2d; color:var(--red); }}
.pair-label {{ font-size:11px; text-transform:uppercase; letter-spacing:.8px;
               font-weight:700; margin-bottom:6px; }}
.pair-label.chosen {{ color:var(--green); }}
.pair-label.rejected {{ color:var(--red); }}

/* CoT + judge */
.cot-box {{ background:#2a1a2a; border:1px solid #4a2d4a; padding:12px 14px;
            border-radius:6px; font-size:13px; white-space:pre-wrap;
            word-break:break-word; color:#c9a0dc; font-style:italic;
            border-left:3px solid #9b59b6; }}
.judge-box {{ background:#1a1a2a; border:1px solid #2d2d4a; padding:10px 14px;
              border-radius:6px; font-size:13px; color:var(--yellow);
              border-left:3px solid var(--yellow); }}
.metadata-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
                  gap:8px; margin-top:10px; }}
.metadata-item {{ font-size:12px; color:var(--muted); }}
.metadata-item strong {{ color:var(--text); }}

/* Pagination */
.pagination {{ display:flex; justify-content:center; gap:8px; margin-top:24px; margin-bottom:40px; }}
.pagination button {{
  background:var(--surface); border:1px solid var(--border);
  color:var(--text); padding:8px 16px; border-radius:6px;
  cursor:pointer; font-size:14px;
}}
.pagination button:hover {{ background:var(--surface2); border-color:var(--accent); }}
.pagination button:disabled {{ opacity:.4; cursor:not-allowed; }}
.page-info {{ display:flex; align-items:center; color:var(--muted); font-size:14px; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>DPO Dataset <span>Viewer</span></h1>
  <div class="meta">{html_mod.escape(run_label)} &middot; {html_mod.escape(run_id)} &middot; persona: {html_mod.escape(persona)}</div>
</div>

<div class="stats-grid" id="stats-grid"></div>
<div class="chart-section" id="scenario-chart"></div>

<div class="controls">
  <select id="filter-scenario"><option value="">All scenarios</option></select>
  <input type="text" id="search-input" placeholder="Search chosen, rejected, CoT, judge reason…">
  <button class="toggle-btn active" id="toggle-system" onclick="toggleSystem()">System Prompt</button>
  <button class="toggle-btn active" id="toggle-cot" onclick="toggleCot()">CoT</button>
  <button class="toggle-btn active" id="toggle-meta" onclick="toggleMeta()">Metadata</button>
</div>

<div class="example-list" id="example-list"></div>
<div class="pagination" id="pagination"></div>

</div>

<script>
const ALL = {examples_json};
const SCENARIOS = {scenarios_json};
const PER_PAGE = 20;
let page = 0;
let showSystem = true, showCot = true, showMeta = true;

function escapeHtml(t) {{ const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }}

function fmtChat(msgs) {{
  if (!msgs || !msgs.length) return '<span style="color:var(--muted)">(no prior context)</span>';
  return msgs.map(m => {{
    const role = m.role || '?';
    if (role === 'user') {{
      return '<div style="margin-bottom:8px"><span class="msg-me" style="font-weight:600">USER:</span> ' + escapeHtml(m.content) + '</div>';
    }} else {{
      return '<div style="margin-bottom:8px"><span class="msg-other" style="font-weight:600">ASSISTANT:</span> ' + escapeHtml(m.content) + '</div>';
    }}
  }}).join('');
}}

function filtered() {{
  let list = ALL;
  const sc = document.getElementById('filter-scenario').value;
  if (sc) list = list.filter(e => e._scenario_name === sc);
  const q = document.getElementById('search-input').value.toLowerCase();
  if (q) list = list.filter(e => {{
    const blob = [e.chosen, e.rejected, e._persona_chain_of_thought||'', e._observable_guideline||'', e._judge_reason||'', e.input, e.instruction, e._persona_visible_response||''].join(' ').toLowerCase();
    return blob.includes(q);
  }});
  return list;
}}

function renderStats() {{
  const g = document.getElementById('stats-grid');
  const scenarios = new Set(ALL.map(e => e._scenario_id));
  const avgChosenWords = ALL.length ? Math.round(ALL.reduce((s,e) => s + e.chosen.split(/\\s+/).length, 0) / ALL.length) : 0;
  const avgRejWords = ALL.length ? Math.round(ALL.reduce((s,e) => s + e.rejected.split(/\\s+/).length, 0) / ALL.length) : 0;
  const rewriteModels = [...new Set(ALL.map(e => e._rewrite_model||'?'))];
  g.innerHTML = [
    {{l:'Total DPO Pairs', v:ALL.length}},
    {{l:'Scenarios', v:scenarios.size}},
    {{l:'Avg Chosen Words', v:avgChosenWords}},
    {{l:'Avg Rejected Words', v:avgRejWords}},
    {{l:'Rewrite Model', v:rewriteModels.join(', ')}},
  ].map(c => `<div class="stat-card"><div class="label">${{c.l}}</div><div class="value">${{c.v}}</div></div>`).join('');
}}

function renderChart() {{
  const el = document.getElementById('scenario-chart');
  const entries = Object.entries(SCENARIOS).sort((a,b)=>b[1]-a[1]);
  const max = entries.length ? entries[0][1] : 1;
  el.innerHTML = '<h3>Examples by Scenario</h3>' + entries.map(([name, count]) => {{
    const pct = Math.round(count/max*100);
    return `<div class="bar-row">
      <div class="bar-label">${{escapeHtml(name)}}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%">${{count}}</div></div>
    </div>`;
  }}).join('');
}}

function renderScenarioFilter() {{
  const sel = document.getElementById('filter-scenario');
  Object.keys(SCENARIOS).sort().forEach(name => {{
    const o = document.createElement('option');
    o.value = name; o.textContent = name + ' (' + SCENARIOS[name] + ')';
    sel.appendChild(o);
  }});
}}

function renderExamples() {{
  const list = filtered();
  const start = page * PER_PAGE;
  const slice = list.slice(start, start + PER_PAGE);
  const el = document.getElementById('example-list');

  if (!slice.length) {{
    el.innerHTML = '<div style="text-align:center;color:var(--muted);padding:40px;">No matching examples.</div>';
    renderPagination(list.length);
    return;
  }}

  el.innerHTML = slice.map((ex, i) => `
    <div class="example-card">
      <div class="card-header">
        <span class="idx">#${{start+i+1}}</span>
        <span class="scenario">${{escapeHtml(ex._scenario_name||'')}} &middot; turn ${{ex._turn_index}}</span>
      </div>
      <div class="card-body">
        ${{showSystem ? `<div class="section-label">System Prompt</div>
          <div class="system-prompt">${{escapeHtml(ex.instruction)}}</div>` : ''}}

        <div class="section-label">Conversation Context (multi-turn chat)</div>
        <div class="context-box">${{fmtChat(ex._chat_messages_before)}}</div>

        <div class="section-label">Last User Message (what the assistant responded to)</div>
        <div class="context-box"><span class="msg-me" style="font-weight:600">USER:</span> ${{escapeHtml(ex.input)}}</div>

        ${{showCot && ex._persona_chain_of_thought ? `
          <div class="section-label">💭 Persona's Internal Reaction (private — not seen by assistant)</div>
          <div class="cot-box">${{escapeHtml(ex._persona_chain_of_thought)}}</div>` : ''}}

        ${{ex._observable_guideline ? `
          <div class="section-label">🔍 Observable Guideline (translated for rewrite model)</div>
          <div style="background:#1a1a2a;border:1px solid #2d2d4a;padding:12px 14px;border-radius:6px;font-size:13px;color:#7eb8da;border-left:3px solid var(--accent);">${{escapeHtml(ex._observable_guideline)}}</div>` : ''}}

        ${{ex._persona_visible_response ? `
          <div class="section-label">👤 What the persona actually replied</div>
          <div style="background:var(--surface2);padding:10px 14px;border-radius:6px;font-size:13px;color:var(--green);border-left:3px solid var(--green);">${{escapeHtml(ex._persona_visible_response).replace(/&lt;msg&gt;/g, '<br>')}}</div>` : ''}}

        ${{ex._judge_reason ? `
          <div class="section-label">⚖️ Judge Reason</div>
          <div class="judge-box">${{escapeHtml(ex._judge_reason)}}</div>` : ''}}

        <div class="section-label">Preference Pair</div>
        <div class="pair-grid">
          <div>
            <div class="pair-label chosen">✓ Chosen (rewrite)</div>
            <div class="pair-box chosen">${{escapeHtml(ex.chosen).replace(/&lt;msg&gt;/g, '<br>')}}</div>
          </div>
          <div>
            <div class="pair-label rejected">✗ Rejected (original)</div>
            <div class="pair-box rejected">${{escapeHtml(ex.rejected).replace(/&lt;msg&gt;/g, '<br>')}}</div>
          </div>
        </div>

        ${{showMeta ? `
          <div class="section-label">Metadata</div>
          <div class="metadata-grid">
            <div class="metadata-item"><strong>Scenario:</strong> ${{escapeHtml(ex._scenario_id||'')}}</div>
            <div class="metadata-item"><strong>Turn:</strong> ${{ex._turn_index}}</div>
            <div class="metadata-item"><strong>Timestamp:</strong> ${{escapeHtml(ex._timestamp||'')}}</div>
            <div class="metadata-item"><strong>Rewrite model:</strong> ${{escapeHtml(ex._rewrite_model||'')}}</div>
            <div class="metadata-item"><strong>Run:</strong> ${{escapeHtml(ex._run_label||'')}}</div>
          </div>` : ''}}
      </div>
    </div>
  `).join('');

  renderPagination(list.length);
}}

function renderPagination(total) {{
  const pages = Math.ceil(total / PER_PAGE);
  const el = document.getElementById('pagination');
  if (pages <= 1) {{ el.innerHTML = ''; return; }}
  el.innerHTML = `
    <button onclick="page=0;renderExamples();scrollUp()" ${{page===0?'disabled':''}}>«</button>
    <button onclick="page--;renderExamples();scrollUp()" ${{page===0?'disabled':''}}>‹ Prev</button>
    <span class="page-info">&nbsp;${{page+1}} / ${{pages}}&nbsp;</span>
    <button onclick="page++;renderExamples();scrollUp()" ${{page>=pages-1?'disabled':''}}>Next ›</button>
    <button onclick="page=${{pages-1}};renderExamples();scrollUp()" ${{page>=pages-1?'disabled':''}}>»</button>
  `;
}}

function scrollUp() {{ window.scrollTo({{top:document.getElementById('example-list').offsetTop-10,behavior:'smooth'}}); }}

function toggleSystem() {{ showSystem=!showSystem; document.getElementById('toggle-system').classList.toggle('active'); renderExamples(); }}
function toggleCot() {{ showCot=!showCot; document.getElementById('toggle-cot').classList.toggle('active'); renderExamples(); }}
function toggleMeta() {{ showMeta=!showMeta; document.getElementById('toggle-meta').classList.toggle('active'); renderExamples(); }}

document.getElementById('filter-scenario').addEventListener('change', () => {{ page=0; renderExamples(); }});
document.getElementById('search-input').addEventListener('input', () => {{ page=0; renderExamples(); }});

renderStats();
renderChart();
renderScenarioFilter();
renderExamples();
</script>
</body>
</html>"""


def run_generation(args) -> Dict:
    """Run the full DPO-dataset generation pipeline given a parsed args namespace.

    Returns a summary dict::

        {
            "output_path":    str,   # absolute path to the written JSON
            "viewer_path":    str | None,  # HTML viewer path, or None if no examples
            "num_examples":   int,
            "num_conversations": int,
            "num_flagged_turns": int,
            "num_skipped_translator": int,
            "num_skipped_rewrite_cot": int,
            "num_skipped_identical": int,
            "num_errors": int,
            "rewrite_model": str,
        }

    The CLI entrypoint (`main()`) is a thin wrapper around this function,
    making the same logic callable from inside a Modal container for the
    automated DPO pipeline (see `modal_app.run_pipeline`).
    """
    run_input_path = os.path.abspath(args.run)
    results_path = resolve_results_path(run_input_path)
    output_path = ensure_output_path(args, run_input_path)

    print(f"\n{C.BOLD}{C.CYAN}{'═' * 66}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  IMPersona DPO Dataset Generator{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * 66}{C.RESET}")
    print(f"  Run:    {results_path}")
    print(f"  Output: {output_path}")

    results_bundle = load_results_bundle(results_path)
    conversations = results_bundle.get("conversations", [])
    run_metadata = results_bundle.get("metadata", {})
    persona_name = run_metadata.get("persona_name") or DEFAULT_PERSONA_NAME
    scenario_ids = (
        [item.strip() for item in args.scenario_ids.split(",") if item.strip()]
        if args.scenario_ids
        else None
    )
    conversations = choose_conversations(
        conversations,
        scenario_ids=scenario_ids,
        limit_conversations=args.limit_conversations,
    )

    if not conversations:
        raise RuntimeError(
            "No conversations selected. Check --scenario-ids / --limit-conversations "
            "filters and that the eval results actually contain a `conversations` list."
        )

    saved_details = load_saved_disagreement_details(results_bundle)

    # ── Phase 1: Collect all rewrite tasks ──
    @dataclass
    class RewriteTask:
        scenario_id: str
        scenario_name: str
        turn: FlaggedAssistantTurn
        judge_reason: str

    rewrite_tasks: List[RewriteTask] = []
    skipped_zero = 0
    flagged_turns = 0

    for conv_idx, conversation in enumerate(conversations, start=1):
        scenario = conversation.get("scenario", {})
        scenario_id = scenario.get("id", f"conversation_{conv_idx}")
        scenario_name = scenario.get("name", scenario_id)

        classifications = saved_details.get(scenario_id, [])
        if not classifications:
            skipped_zero += 1
            print(
                f"  {C.DIM}Skipping {scenario_id}: 0 disagreement turns in saved eval.{C.RESET}"
            )
            continue

        assistant_turns = extract_flagged_assistant_turns(conversation, persona_name)
        if not assistant_turns:
            print(f"  {C.DIM}Skipping {scenario_id}: no persona turns with CoT.{C.RESET}")
            continue

        flagged = [row for row in classifications if row.get("is_disagreement")]
        flagged_turns += len(flagged)

        if not flagged:
            print(
                f"  {C.DIM}Skipping {scenario_id}: 0 flagged after filtering.{C.RESET}"
            )
            continue

        print(
            f"  {C.GREEN}{scenario_name}{C.RESET}  {C.DIM}({scenario_id}){C.RESET}: "
            f"{len(flagged)} flagged / {len(assistant_turns)} CoT turns"
        )

        for row in flagged:
            idx = row["turn_index"]
            if idx >= len(assistant_turns):
                continue
            turn = assistant_turns[idx]
            # Skip opening turns that have no preceding AI response to rewrite
            if turn.original_ai_response is None:
                continue
            rewrite_tasks.append(
                RewriteTask(
                    scenario_id=scenario_id,
                    scenario_name=scenario_name,
                    turn=turn,
                    judge_reason=row.get("reason", ""),
                )
            )

    if args.max_examples is not None:
        rewrite_tasks = rewrite_tasks[: args.max_examples]

    if not rewrite_tasks:
        print(f"\n{C.YELLOW}No flagged turns to rewrite.{C.RESET}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f)
            f.write("\n")
        print(f"  Saved empty dataset to: {output_path}\n")
        return {
            "output_path": output_path,
            "viewer_path": None,
            "num_examples": 0,
            "num_conversations": len(conversations),
            "num_flagged_turns": flagged_turns,
            "num_skipped_translator": 0,
            "num_skipped_rewrite_cot": 0,
            "num_skipped_identical": 0,
            "num_errors": 0,
            "rewrite_model": "",
        }

    # ── Phase 2: Initialize models ──
    translate_client = AIPartnerClient(
        model=TRANSLATE_MODEL, reasoning=TRANSLATE_REASONING, base_url=TRANSLATE_BASE_URL
    )
    rewrite_client, rewrite_model_name = resolve_rewrite_model(args, run_metadata)

    # Local models are not thread-safe — force sequential execution
    max_workers = args.max_workers
    if args.rewrite_local_model and max_workers > 1:
        print(f"  {C.YELLOW}⚠ Local model detected: forcing --max-workers 1 "
              f"(was {max_workers}){C.RESET}")
        max_workers = 1

    # Resolve the empty-think prefix mode.
    if args.empty_think_prefix == "yes":
        empty_think_prefix = "<think>\n\n</think>\n\n"
    else:
        empty_think_prefix = ""

    print(f"\n  Translate model: {TRANSLATE_MODEL} (CoT → observable guideline)")
    print(f"  Rewrite model:   {rewrite_model_name}")
    print(f"  Empty-think prefix: "
          f"{'ON — prepending <think></think> to chosen/rejected' if empty_think_prefix else 'OFF'}")
    print(f"  Tasks to process: {len(rewrite_tasks)}")
    print(f"  Max workers: {max_workers}\n")

    # ── Phase 3: Translate + rewrite in parallel ──
    emitted_examples: List[Dict] = []
    skipped_identical = 0
    skipped_by_translator = 0
    skipped_rewrite_cot = 0
    errors = 0

    def _execute_translate_and_rewrite(
        task: RewriteTask,
    ) -> Tuple[RewriteTask, Optional[str], Optional[str], Optional[str]]:
        """Worker: translate CoT → guideline, then rewrite.
        Returns (task, rewritten, guideline, error).
        If guideline is "[SKIP:reason]", rewritten will be None."""
        try:
            guideline, skip_reason = translate_feedback_to_guideline(
                translate_client=translate_client,
                turn=task.turn,
                judge_reason=task.judge_reason,
                persona_name=persona_name,
            )
            if skip_reason:
                return (task, None, f"[SKIP] {skip_reason}", None)

            rewritten = rewrite_assistant_response(
                rewrite_client=rewrite_client,
                turn=task.turn,
                guideline=guideline,
                persona_name=persona_name,
            )
            return (task, rewritten, guideline, None)
        except Exception as e:
            return (task, None, None, f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_execute_translate_and_rewrite, task): task
            for task in rewrite_tasks
        }

        for future in as_completed(futures):
            task, rewritten, guideline, err = future.result()

            if err:
                errors += 1
                print(
                    f"  {C.YELLOW}⚠ Failed for {task.scenario_id} "
                    f"turn {task.turn.turn_index}: {err}{C.RESET}"
                )
                continue

            # Translation model filtered this example
            if guideline and guideline.startswith("[SKIP]"):
                skipped_by_translator += 1
                reason = guideline[len("[SKIP]"):].strip()
                print(
                    f"  {C.DIM}Filtered {task.scenario_id} turn {task.turn.turn_index}: "
                    f"{reason}{C.RESET}"
                )
                continue

            if not rewritten:
                errors += 1
                print(
                    f"  {C.YELLOW}⚠ Empty rewrite for {task.scenario_id} "
                    f"turn {task.turn.turn_index}; skipping.{C.RESET}"
                )
                continue

            if has_visible_rewrite_cot(rewritten):
                skipped_rewrite_cot += 1
                print(
                    f"  {C.DIM}Filtered {task.scenario_id} turn {task.turn.turn_index}: "
                    f"rewrite leaked visible chain-of-thought.{C.RESET}"
                )
                continue

            # Length-ratio guard: a rewrite that is drastically longer than
            # the original is almost certainly leaked chain-of-thought that
            # slipped past the heuristic pattern check above.  Legitimate
            # rewrites stay close to the original's length.
            orig_len = len(task.turn.original_ai_response)
            if orig_len > 0 and len(rewritten) / orig_len > MAX_REWRITE_LENGTH_RATIO:
                skipped_rewrite_cot += 1
                print(
                    f"  {C.DIM}Filtered {task.scenario_id} turn {task.turn.turn_index}: "
                    f"rewrite is {len(rewritten)/orig_len:.1f}x longer than original "
                    f"(likely leaked CoT).{C.RESET}"
                )
                continue

            if (not args.allow_identical) and normalize_text(rewritten) == normalize_text(task.turn.original_ai_response):
                skipped_identical += 1
                print(
                    f"  {C.DIM}Skipping identical rewrite for {task.scenario_id} "
                    f"turn {task.turn.turn_index}.{C.RESET}"
                )
                continue

            # Build the input for the DPO pair: the user message(s) the
            # assistant was responding to (the last user turn in the chat)
            chat_input = task.turn.user_message_before

            emitted_examples.append(
                {
                    "instruction": task.turn.ai_partner_system_prompt,
                    "input": chat_input,
                    "chosen": empty_think_prefix + rewritten,
                    "rejected": empty_think_prefix + strip_think_tokens(task.turn.original_ai_response),
                    "_run_id": run_metadata.get("run_id"),
                    "_run_label": run_metadata.get("run_label"),
                    "_scenario_id": task.scenario_id,
                    "_scenario_name": task.scenario_name,
                    "_persona_name": persona_name,
                    "_turn_index": task.turn.turn_index,
                    "_transcript_ai_msg_index": task.turn.transcript_ai_msg_index,
                    "_persona_chain_of_thought": task.turn.persona_chain_of_thought,
                    "_observable_guideline": guideline,
                    "_persona_visible_response": task.turn.persona_visible_response,
                    "_judge_reason": task.judge_reason,
                    "_rewrite_model": rewrite_model_name,
                    "_chat_messages_before": task.turn.chat_messages_before,
                }
            )
            print(
                f"  {C.GREEN}✓{C.RESET} {task.scenario_id} turn {task.turn.turn_index} "
                f"({len(emitted_examples)} total)"
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(emitted_examples, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Generate interactive HTML visualizer alongside the dataset
    viewer_path = output_path.replace(".json", "_viewer.html")
    viewer_html = generate_dpo_viewer_html(emitted_examples, run_metadata)
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(viewer_html)

    print(f"\n{C.BOLD}Summary{C.RESET}")
    print(f"  Conversations processed: {len(conversations)}")
    print(f"  Conversations skipped (0 disagreements): {skipped_zero}")
    print(f"  Flagged disagreement turns: {flagged_turns}")
    print(f"  Rewrite tasks submitted: {len(rewrite_tasks)}")
    print(f"  Filtered by translator (bad training signal): {skipped_by_translator}")
    print(f"  Filtered after rewrite (visible CoT): {skipped_rewrite_cot}")
    print(f"  Rewrite errors: {errors}")
    print(f"  Identical rewrites skipped: {skipped_identical}")
    print(f"  Final DPO examples: {len(emitted_examples)}")
    print(f"  Saved to: {output_path}")
    print(f"  Viewer: {viewer_path}")

    if isinstance(translate_client, AIPartnerClient):
        usage = translate_client.get_usage()
        print(
            f"  Translate usage ({TRANSLATE_MODEL}): {usage['num_calls']} calls, "
            f"{usage['total_tokens']} tokens, "
            f"${usage['estimated_cost_usd']:.4f}"
        )
    if isinstance(rewrite_client, AIPartnerClient):
        usage = rewrite_client.get_usage()
        print(
            f"  Rewrite usage ({rewrite_model_name}): {usage['num_calls']} calls, "
            f"{usage['total_tokens']} tokens, "
            f"${usage['estimated_cost_usd']:.4f}"
        )

    print()

    return {
        "output_path": output_path,
        "viewer_path": viewer_path,
        "num_examples": len(emitted_examples),
        "num_conversations": len(conversations),
        "num_flagged_turns": flagged_turns,
        "num_skipped_translator": skipped_by_translator,
        "num_skipped_rewrite_cot": skipped_rewrite_cot,
        "num_skipped_identical": skipped_identical,
        "num_errors": errors,
        "rewrite_model": rewrite_model_name,
    }


def main():
    """CLI entrypoint: parse args and run the generation pipeline."""
    args = parse_args()
    run_generation(args)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
