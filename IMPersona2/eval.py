#!/usr/bin/env python3
"""
Impersona-Env — Evaluation Framework
=====================================

Runs automated conversations between an IMPersona (fine-tuned model) and
an AI partner (the assistant) across predefined scenarios, saving full
transcripts with chain-of-thought for inspection.

The IMPersona is loaded via LlamaFactory (locally or on Modal). The AI
partner is an API model or the assistant served on Modal, and replies with
a fixed, generic system prompt.

Each scenario defines:
  - impersona_context: situational context injected into the IMPersona's
    system prompt (matching the training data format)

Output: a timestamped directory under --output_dir containing:
  - results.json   — full conversation transcripts + metadata
  - viewer.html    — interactive HTML viewer for inspection

Usage (from the repo root):
    # Run all scenarios (user model + adapter on Modal; models default to
    # configs/models.yaml, the partner to the assistant on Modal):
    python -m IMPersona2.eval --impersona_adapter adapter-user-model

    # Run the quick subset (preselected mix of emotional + casual):
    python -m IMPersona2.eval ... --quick

    # Run specific scenarios:
    python -m IMPersona2.eval ... --scenarios therapy_vague_malaise,acute_major_rejection

    # Run a specific category:
    python -m IMPersona2.eval ... --category therapy

    # List available scenarios:
    python -m IMPersona2.eval --list-scenarios
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from IMPersona2.experiment_config import default_persona_name
from IMPersona2.model_config import model_for, template_for

DEFAULT_PERSONA_NAME = default_persona_name()

# Load .env from the repo root (one level above this file)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# ─────────────────────────────────────────────
# Colors for terminal output
# ─────────────────────────────────────────────
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    GRAY = "\033[90m"


# ─────────────────────────────────────────────
# Scenario Loading
# ─────────────────────────────────────────────
def load_scenarios(
    scenario_file: str,
    scenario_ids: Optional[List[str]] = None,
    category: Optional[str] = None,
    quick: bool = False,
) -> List[Dict]:
    """Load scenarios from JSON file, optionally filtering by ID, category, or quick flag."""
    with open(scenario_file, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    if scenario_ids:
        scenarios = [s for s in scenarios if s["id"] in scenario_ids]
        found_ids = {s["id"] for s in scenarios}
        missing = set(scenario_ids) - found_ids
        if missing:
            print(f"{C.YELLOW}⚠ Scenarios not found: {', '.join(missing)}{C.RESET}")

    if quick:
        scenarios = [s for s in scenarios if s.get("quick", False)]

    if category:
        scenarios = [s for s in scenarios if s["category"] == category]

    return scenarios


def list_scenarios(scenario_file: str):
    """Print available scenarios grouped by category."""
    with open(scenario_file, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    categories = {}
    for s in scenarios:
        cat = s["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(s)

    quick_count = sum(1 for s in scenarios if s.get("quick", False))
    print(f"\n{C.BOLD}{C.CYAN}Available Scenarios ({len(scenarios)} total, {quick_count} in --quick set){C.RESET}\n")
    for cat, items in categories.items():
        print(f"  {C.BOLD}{cat}{C.RESET}")
        for s in items:
            quick_tag = f" {C.YELLOW}★ quick{C.RESET}" if s.get("quick") else ""
            print(f"    {C.GREEN}{s['id']}{C.RESET}  {s['name']}{quick_tag}")
            print(f"      {C.DIM}{s['description']}{C.RESET}")
        print()


# ─────────────────────────────────────────────
# AI Partner Client (Anthropic + OpenAI)
# ─────────────────────────────────────────────

def _is_anthropic_model(model: str) -> bool:
    """Return True if the model string indicates an Anthropic (Claude) model."""
    return model.lower().startswith("claude")


def openai_client_kwargs(base_url: Optional[str] = None) -> Dict:
    """Keyword arguments for ``openai.OpenAI``: OpenAI's own API when
    ``base_url`` is unset, else that OpenAI-compatible endpoint (OpenRouter,
    vLLM, LlamaFactory, ...)."""
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
        if "openrouter.ai" in base_url and os.environ.get("OPENROUTER_API_KEY"):
            kwargs["api_key"] = os.environ["OPENROUTER_API_KEY"]
        # Local servers don't need a real API key
        elif not os.environ.get("OPENAI_API_KEY"):
            kwargs["api_key"] = "local"
    return kwargs


# The anthropic SDK refuses a non-streaming request whose max_tokens could take
# over 10 minutes at its estimate of 128_000 output tokens per hour ("Streaming
# is required for operations that may take longer than 10 minutes"), e.g. the
# "high" thinking budget plus the answer.
ANTHROPIC_MAX_NONSTREAMING_TOKENS = 21_333


def create_anthropic_message(client, **kwargs):
    """``client.messages.create(**kwargs)``, streamed when max_tokens is above
    ANTHROPIC_MAX_NONSTREAMING_TOKENS. Returns the complete Message either way."""
    if kwargs["max_tokens"] > ANTHROPIC_MAX_NONSTREAMING_TOKENS:
        with client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()
    return client.messages.create(**kwargs)


class AIPartnerClient:
    """
    Manages the AI partner side of the conversation.

    Supports three backends:

    1. **Anthropic** (Claude) — auto-detected when model starts with "claude".
    2. **OpenAI / OpenAI-compatible API** — default for everything else.
       Use ``base_url`` to point at a custom endpoint (e.g. a vLLM server).
    3. **Local LlamaFactory model** — loaded in-process, no server needed.
       Pass a ``chat_model`` (LlamaFactory ``ChatModel`` instance) to use
       this mode. Works exactly like the IMPersona model loading.

    Maintains multi-turn message history and tracks token usage.

    Parameters
    ----------
    model : str
        Model identifier: an API model id (routed by prefix, see
        configs/models.yaml), or a label when ``chat_model`` is given.
    reasoning : str or None
        Controls reasoning effort.
        - For OpenAI: passed as ``reasoning_effort`` ("low", "medium", "high").
          ``None`` means don't send the parameter (model default).
        - For Anthropic: mapped to an extended-thinking budget_tokens.
          ``None`` means no extended thinking.
    base_url : str or None
        Custom base URL for OpenAI-compatible APIs (e.g. a local
        LlamaFactory API server).  Ignored for Anthropic / local models.
    chat_model : ChatModel or None
        A pre-loaded LlamaFactory ``ChatModel`` instance.  When provided,
        all generation happens in-process — no API calls.  ``model`` is
        used only as a label for logging / output metadata.
    generation_kwargs : dict or None
        Extra sampling parameters passed to the local model's
        ``stream_chat()`` (e.g. ``temperature``, ``top_p``, ``top_k``,
        ``repetition_penalty``).  Ignored for API backends.
    """

    # Recommended generation defaults for Qwen instruct models
    # (non-thinking mode).  Use these when serving a Qwen model locally.
    QWEN_GENERATION_DEFAULTS = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
    }

    # Pricing per million tokens (USD): prefix → (input_$/M, output_$/M)
    _PRICING = {
        # Anthropic
        "claude-opus-4":   (5.0,  25.0),
        "claude-sonnet-4": (3.0,  15.0),
        "claude-haiku":    (0.25, 1.25),
        # OpenAI
        "gpt-5.4":         (2.50, 10.0),
        "gpt-5.2":         (2.0,  8.0),
        "gpt-5.1":         (2.0,  8.0),
        "gpt-5-mini":      (0.40, 1.60),
        "gpt-4.1":         (2.0,  8.0),
        "gpt-4o":          (2.50, 10.0),
        "o3":              (2.0,  8.0),
        "o4-mini":         (1.10, 4.40),
    }

    # Mapping from reasoning level → Anthropic extended-thinking budget
    _ANTHROPIC_THINKING_BUDGETS = {
        "low": 5_000,
        "medium": 10_000,
        "high": 20_000,
    }

    def __init__(
        self,
        model: str,
        reasoning: Optional[str] = None,
        base_url: Optional[str] = None,
        chat_model=None,
        generation_kwargs: Optional[Dict] = None,
    ):
        self.model = model
        self.reasoning = reasoning
        self.base_url = base_url
        self._local_model = chat_model  # LlamaFactory ChatModel instance
        self._generation_kwargs = generation_kwargs or {}

        if chat_model is not None:
            # Local in-process mode — no API client needed
            self._use_anthropic = False
            self._use_local = True
            self.client = None
        else:
            self._use_local = False
            self._use_anthropic = _is_anthropic_model(model) and base_url is None

            if self._use_anthropic:
                import anthropic
                self.client = anthropic.Anthropic()
            else:
                from openai import OpenAI
                self.client = OpenAI(**openai_client_kwargs(base_url))

        self._input_tokens = 0
        self._output_tokens = 0
        self._num_calls = 0

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict],
        max_tokens: int = 4096,
    ) -> str:
        """Generate a response given the system prompt and message history."""
        if self._use_local:
            return self._generate_local(system_prompt, messages, max_tokens)
        elif self._use_anthropic:
            return self._generate_anthropic(system_prompt, messages, max_tokens)
        else:
            return self._generate_openai(system_prompt, messages, max_tokens)

    def _generate_local(
        self,
        system_prompt: str,
        messages: List[Dict],
        max_tokens: int,
    ) -> str:
        """Generate via a local LlamaFactory ChatModel (in-process, no server).

        Sampling parameters (temperature, top_p, top_k, repetition_penalty,
        etc.) are forwarded from ``self._generation_kwargs`` to
        ``stream_chat()``'s ``**input_kwargs``, which the underlying engine
        (HuggingFace / vLLM / SGLang) picks up and uses to override its
        defaults.
        """
        response = ""
        for new_text in self._local_model.stream_chat(
            messages,
            system=system_prompt,
            max_new_tokens=max_tokens,
            **self._generation_kwargs,
        ):
            response += new_text

        self._num_calls += 1
        return response.strip()

    def _generate_anthropic(
        self,
        system_prompt: str,
        messages: List[Dict],
        max_tokens: int,
    ) -> str:
        """Generate via Anthropic API."""
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )

        # Enable extended thinking if reasoning is requested.
        # max_tokens must exceed budget_tokens, so bump it if needed.
        if self.reasoning and self.reasoning in self._ANTHROPIC_THINKING_BUDGETS:
            budget = self._ANTHROPIC_THINKING_BUDGETS[self.reasoning]
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["max_tokens"] = max(max_tokens, budget + max_tokens)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = create_anthropic_message(self.client, **kwargs)

        # Track usage
        if hasattr(response, "usage") and response.usage:
            self._input_tokens += response.usage.input_tokens
            self._output_tokens += response.usage.output_tokens
            self._num_calls += 1

        # Surface non-standard stop reasons
        if hasattr(response, "stop_reason") and response.stop_reason not in (
            "end_turn", None
        ):
            print(
                f"  {C.YELLOW}⚠ Anthropic stop_reason={response.stop_reason} "
                f"(model={self.model}){C.RESET}"
            )

        # Extract text (skip thinking blocks)
        for block in response.content:
            if block.type == "text":
                return block.text.strip()

        print(
            f"  {C.RED}⚠ Anthropic returned no text blocks "
            f"(model={self.model}, blocks={[b.type for b in response.content]}){C.RESET}"
        )
        return ""

    def _generate_openai(
        self,
        system_prompt: str,
        messages: List[Dict],
        max_tokens: int,
    ) -> str:
        """Generate via OpenAI-compatible API (works with OpenAI, LlamaFactory, vLLM, etc.)."""
        # Prepend system message to the conversation
        openai_messages = [{"role": "system", "content": system_prompt}] + messages

        # OpenAI's own API takes max_completion_tokens (max_tokens is deprecated
        # and rejected by reasoning models); OpenAI-compatible servers (vLLM,
        # LlamaFactory, OpenRouter) take max_tokens.
        use_new_param = not self.base_url

        kwargs = dict(
            model=self.model,
            messages=openai_messages,
        )
        if use_new_param:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        # Pass reasoning_effort if set (only for real OpenAI — local servers
        # don't support this parameter)
        if self.reasoning and not self.base_url:
            kwargs["reasoning_effort"] = self.reasoning

        response = self.client.chat.completions.create(**kwargs)

        # Track usage
        if hasattr(response, "usage") and response.usage:
            self._input_tokens += response.usage.prompt_tokens
            self._output_tokens += response.usage.completion_tokens
            self._num_calls += 1

        choice = response.choices[0]

        # Surface refusals / stop reasons
        if choice.finish_reason and choice.finish_reason not in ("stop", "end_turn"):
            print(
                f"  {C.YELLOW}⚠ OpenAI finish_reason={choice.finish_reason} "
                f"(model={self.model}){C.RESET}"
            )
        if hasattr(choice.message, "refusal") and choice.message.refusal:
            print(
                f"  {C.RED}⚠ OpenAI REFUSAL: {choice.message.refusal}{C.RESET}"
            )

        content = choice.message.content
        if content is None:
            print(
                f"  {C.RED}⚠ OpenAI returned content=None "
                f"(finish_reason={choice.finish_reason}, model={self.model}){C.RESET}"
            )
            return ""

        return content.strip()

    def get_usage(self) -> Dict:
        """Return cumulative token usage and estimated cost."""
        model_lower = self.model.lower()
        input_price, output_price = 0.0, 0.0
        for prefix, prices in self._PRICING.items():
            if model_lower.startswith(prefix):
                input_price, output_price = prices
                break

        input_cost = self._input_tokens / 1_000_000 * input_price
        output_cost = self._output_tokens / 1_000_000 * output_price

        usage = {
            "model": self.model,
            "num_calls": self._num_calls,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self._input_tokens + self._output_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 4),
        }
        if self._use_local or self.base_url:
            usage["estimated_cost_usd"] = 0.0  # local models have no API cost
        if self.base_url:
            usage["base_url"] = self.base_url
        if self._use_local:
            usage["backend"] = "local"
        if self._generation_kwargs:
            usage["generation_kwargs"] = self._generation_kwargs
        return usage


# ─────────────────────────────────────────────
# IMPersona System Prompt (matches training format)
# ─────────────────────────────────────────────
_BASE_SYSTEM_PROMPT = (
    "You are role-playing as a person named {persona_name}. "
    "Stay in character and respond as {persona_name} would. "
    "When sending multiple messages in quick succession, separate them with <msg>. "
    "Messages in the conversation are numbered [1], [2], etc. "
    "To reply to a specific earlier message, start your response with "
    "<reply:N> where N is that message's number. "
    "Only use this when directly replying to a specific message, not for general responses."
)

_CONTEXT_TEMPLATE = " Context: {context}"
_RELATIONSHIP_TEMPLATE = (
    " You are talking to {other_name}. "
    "Relationship context: {relationship}"
)

# Default relationship for all eval scenarios
_EVAL_RELATIONSHIP_DEFAULT = (
    "AI Assistant is an AI chatbot on your phone that you use "
    "for all sorts of things \u2014 casual conversation, getting help with tasks, "
    "talking through decisions, and whatever else comes up."
)
# Optional "critic mode" addendum appended to the relationship context when
# --critic_mode is passed. Intent: nudge the IMPersona's internal reactions
# (CoT) toward authenticity without prescribing any specific failure modes
# or dictating whether/how those reactions get verbalized. The idea is to
# remove a softening suppression and let the LoRA's learned preferences
# surface more often, yielding denser disagreement signal for DPO.
#
# The wording gives the model a *reason* its reactions need no filtering
# (it's talking to an AI, not a person) rather than just asserting it.
_EVAL_RELATIONSHIP_CRITIC_ADDENDUM = (
    " Since it's an AI and not a person, you don't filter or soften your "
    "reactions to its responses \u2014 they come through exactly as they are."
)
_EVAL_OTHER_NAME = "AI Assistant"

# Fixed generic AI partner prompt (no scenario-specific coaching).
# The whole point: a high-EQ model should figure out the right approach
# from the conversation itself, not from instructions telling it how
# to handle each emotional situation. Zero context — the assistant
# must read the room purely from what the user says.
_GENERIC_AI_PARTNER_PROMPT = "You are an AI assistant."


def build_impersona_system_prompt(
    persona_name: str,
    context: str,
    relationship_context: Optional[str] = None,
    critic_mode: bool = False,
) -> str:
    """Build the IMPersona system prompt in training-data format.

    When ``critic_mode`` is True, appends ``_EVAL_RELATIONSHIP_CRITIC_ADDENDUM``
    to the relationship description. The addendum is deliberately neutral —
    it references only the authenticity of the IMPersona's internal reactions,
    not any specific failure modes or output behavior.
    """
    prompt = _BASE_SYSTEM_PROMPT.format(persona_name=persona_name)
    if context:
        prompt += _CONTEXT_TEMPLATE.format(context=context)
    relationship = relationship_context or _EVAL_RELATIONSHIP_DEFAULT
    if critic_mode:
        relationship = relationship + _EVAL_RELATIONSHIP_CRITIC_ADDENDUM
    prompt += _RELATIONSHIP_TEMPLATE.format(
        other_name=_EVAL_OTHER_NAME,
        relationship=relationship,
    )
    return prompt


# ─────────────────────────────────────────────
# Conversation History Formatting (matches training format)
# ─────────────────────────────────────────────
def format_history_for_impersona(
    history: List[Dict],
) -> str:
    """
    Format conversation history in the training data format.

    Each message dict has: speaker, content, timestamp.
    Returns: "Current conversation:\\n[1] [timestamp] Speaker: content\\n..."
    """
    parts = ["Current conversation:"]
    for i, msg in enumerate(history):
        label = "Me" if msg["speaker"] == "Me" else msg["speaker"]
        parts.append(f"[{i + 1}] [{msg['timestamp']}] {label}: {msg['content']}")

    # Add the next "Me:" turn marker
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts.append(f"[{len(history) + 1}] [{now}] Me: ")

    return "\n".join(parts)


# ─────────────────────────────────────────────
# Output Parsing
# ─────────────────────────────────────────────
def parse_model_output(raw: str) -> Tuple[str, Optional[str]]:
    """
    Parse model output into (display_text, chain_of_thought).

    Training format: <think>internal monologue</think>actual response text
    """
    cot = None
    text = raw.strip()

    cot_match = re.match(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if cot_match:
        cot = cot_match.group(1).strip()
        text = cot_match.group(2).strip()

    return text, cot


def clean_text_for_ai_partner(text: str) -> str:
    """Strip <reply:N> tags and <msg> tokens for the AI partner's view."""
    cleaned = re.sub(r"<reply:\d+>", "", text)
    # Replace <msg> with newlines (each sub-message on its own line)
    cleaned = cleaned.replace("<msg>", "\n")
    # Clean up extra whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def strip_think_tokens(text: str) -> str:
    """Strip <think>...</think> blocks from AI partner responses.

    Reasoning models can emit <think> tokens in their output.
    These must be removed before passing to the persona model, which
    uses <think> for its own chain-of-thought — seeing them in the
    AI partner's messages causes role confusion and identity bleed.

    Handles nested/malformed patterns like <think><think></think>.
    """
    # Find the last </think> and take everything after it
    idx = text.rfind("</think>")
    if idx != -1:
        text = text[idx + len("</think>"):]
    else:
        # No closing tag — just remove any orphan <think> tags
        text = text.replace("<think>", "")
    return text.strip()


# ─────────────────────────────────────────────
# Modal IMPersona Client
# ─────────────────────────────────────────────

class ModalIMPersonaClient:
    """Wraps the deployed VLLMInference class to provide IMPersona generation.

    Duck-type compatible with the ``chat_model`` parameter used throughout
    the eval pipeline (``generate_impersona_response`` and ``run_conversation``).
    """

    def __init__(
        self,
        model_name: str,
        app_name: str = "impersona",
        adapter_name: str = "",
        template: str = "",
        cls_name: str = "VLLMInference",
    ):
        import modal
        self._cls = modal.Cls.from_name(app_name, cls_name)
        self._instance = self._cls(
            model_name=model_name,
            adapter_name=adapter_name,
            template=template,
        )
        self.model = f"modal:{app_name}/{cls_name}({model_name})"

    def generate(
        self,
        system_prompt: str,
        conversation_input: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        return self._instance.generate.remote(
            system_prompt=system_prompt,
            conversation_input=conversation_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )


class ModalPartnerClient:
    """Wraps a Modal inference class for AI partner generation.

    API-compatible with ``AIPartnerClient`` so ``run_conversation`` can
    use it interchangeably. Uses the deployed ``VLLMPartner`` class with
    the partner's model, adapter, and template.

    The chat template is detected from the model name unless ``template`` is
    given. ``enable_thinking`` defaults to ``assistant.thinking`` in
    configs/models.yaml; for reasoning models, False suppresses the thinking
    trace (other templates ignore it).
    """

    def __init__(
        self,
        model_name: str,
        app_name: str = "impersona",
        adapter_name: str = "",
        template: str = "",
        cls_name: str = "VLLMPartner",
        enable_thinking: bool | None = None,
    ):
        import modal
        self._cls = modal.Cls.from_name(app_name, cls_name)
        if enable_thinking is None:
            enable_thinking = bool(model_for("assistant", "thinking"))

        kwargs = {
            "model_name": model_name,
            "adapter_name": adapter_name,
            "template": template,  # "" = the class detects it from model_name
        }
        kwargs["enable_thinking"] = bool(enable_thinking)
        self._instance = self._cls(**kwargs)
        self.model = model_name
        self.enable_thinking = bool(enable_thinking)
        self._cls_name = cls_name
        self._num_calls = 0

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict],
        max_tokens: int = 4096,
    ) -> str:
        self._num_calls += 1
        if self._cls_name == "VLLMPartner":
            return self._instance.generate.remote(
                system_prompt=system_prompt,
                messages_json=json.dumps(messages),
                max_tokens=max_tokens,
            )

        conversation_input = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        return self._instance.generate.remote(
            system_prompt=system_prompt,
            conversation_input=conversation_input,
            max_new_tokens=max_tokens,
        )

    def get_usage(self) -> Dict:
        return {
            "model": self.model,
            "num_calls": self._num_calls,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0,
        }


def load_impersona_modal(
    model_name: str,
    adapter_name: str = "",
    template: str = "",
    app_name: str = "impersona",
) -> ModalIMPersonaClient:
    """Connect to the deployed IMPersona model on Modal."""
    print(f"\n{C.CYAN}Connecting to IMPersona on Modal: {model_name} + {adapter_name or '(no adapter)'}{C.RESET}")
    client = ModalIMPersonaClient(model_name, app_name=app_name, adapter_name=adapter_name, template=template)
    print(f"{C.GREEN}✅ IMPersona Modal client ready.{C.RESET}")
    return client


def generate_impersona_response(
    chat_model,
    system_prompt: str,
    conversation_input: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate a response from the IMPersona model (Modal or API)."""
    if isinstance(chat_model, ModalIMPersonaClient):
        return chat_model.generate(
            system_prompt=system_prompt,
            conversation_input=conversation_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    elif isinstance(chat_model, AIPartnerClient):
        messages = [{"role": "user", "content": conversation_input}]
        return chat_model.generate(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_new_tokens,
        )
    elif hasattr(chat_model, "generate"):
        return chat_model.generate(
            system_prompt=system_prompt,
            conversation_input=conversation_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    else:
        raise TypeError(f"Unsupported chat_model type: {type(chat_model)}")


# ─────────────────────────────────────────────
# Conversation Engine
# ─────────────────────────────────────────────
def run_conversation(
    scenario: Dict,
    chat_model,
    ai_partner: AIPartnerClient,
    persona_name: str = DEFAULT_PERSONA_NAME,
    max_messages: int = 20,
    temperature: float = 0.7,
    show_live: bool = True,
    system_prompt_override: Optional[str] = None,
    no_context: bool = False,
    critic_mode: bool = False,
    include_prior_cot: bool = False,
) -> Dict:
    """
    Run a full conversation between the IMPersona and AI partner.

    Returns a conversation record dict with all messages, CoTs, and metadata.
    """
    scenario_id = scenario["id"]
    scenario_name = scenario["name"]

    if show_live:
        print(f"\n{C.BOLD}{C.CYAN}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}  Scenario: {scenario_name}{C.RESET}")
        print(f"{C.DIM}  {scenario['description']}{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}{'═' * 60}{C.RESET}\n")

    # ── Build system prompts ──
    if system_prompt_override:
        # Ablation mode: use the override, optionally append context
        impersona_system_prompt = system_prompt_override
        if not no_context:
            impersona_system_prompt += _CONTEXT_TEMPLATE.format(
                context=scenario["impersona_context"]
            )
    else:
        impersona_system_prompt = build_impersona_system_prompt(
            persona_name=persona_name,
            context=scenario["impersona_context"],
            relationship_context=scenario.get("relationship_context"),
            critic_mode=critic_mode,
        )

    ai_partner_system_prompt = _GENERIC_AI_PARTNER_PROMPT

    # ── Conversation state ──
    history: List[Dict] = []         # shared message history
    partner_messages: List[Dict] = [] # the AI partner's multi-turn messages
    transcript: List[Dict] = []      # full transcript for saving
    total_messages = 0
    start_time = time.time()

    # ── Main conversation loop ──
    # IMPersona opens (mirrors real usage — the user initiates the
    # conversation, the assistant has to read the room).
    while total_messages < max_messages:
        # ── IMPersona's turn ──
        conversation_input = format_history_for_impersona(history)
        raw_output = generate_impersona_response(
            chat_model,
            system_prompt=impersona_system_prompt,
            conversation_input=conversation_input,
            temperature=temperature,
        )

        text, cot = parse_model_output(raw_output)

        # Show CoT
        if show_live and cot:
            print(f"  {C.DIM}{C.MAGENTA}  💭 {cot}{C.RESET}")

        # Split on <msg> for display / transcript, but keep combined
        # in history so the model sees the same format as training data
        # (training data combines consecutive "Me" messages with <msg>).
        parts = [p.strip() for p in text.split("<msg>") if p.strip()]
        if not parts:
            parts = [text.strip()] if text.strip() else ["..."]

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add a SINGLE combined entry to history (matches training format)
        combined_for_history = "<msg>".join(parts)
        # When the adapter was trained with prior-turn CoT exposure (via
        # IMPersona2/add_prior_cot.py), we need to mirror that at inference
        # so the input distribution matches training. Prepend the <think>
        # block exactly as training-time history lines look:
        #     [N] [ts] Me: <think>cot</think>content
        if include_prior_cot and cot:
            history_content = f"<think>{cot}</think>{combined_for_history}"
        else:
            history_content = combined_for_history
        history.append({
            "speaker": "Me",
            "content": history_content,
            "timestamp": ts,
        })
        total_messages += len(parts)  # still count sub-messages for loop limit

        # Clean text for display and for the AI partner
        display_parts = []
        for part in parts:
            display = re.sub(r"<reply:\d+>", "", part).strip()
            if display:
                display_parts.append(display)

        if show_live:
            for dp in display_parts:
                print(f"  {C.GREEN}{C.BOLD}  {persona_name}:{C.RESET} {dp}")
            print()

        # Save to transcript (CoT only on the first sub-message)
        for i, part in enumerate(parts):
            transcript.append({
                "speaker": persona_name,
                "content": part,
                "timestamp": ts,
                "chain_of_thought": cot if i == 0 else None,
            })

        if total_messages >= max_messages:
            break

        # ── AI partner responds ──
        clean_response = clean_text_for_ai_partner(text)
        partner_messages.append({"role": "user", "content": clean_response})

        ai_response = None
        for _attempt in range(5):
            try:
                ai_response = ai_partner.generate(
                    system_prompt=ai_partner_system_prompt,
                    messages=partner_messages,
                )
                break
            except Exception as e:
                wait = 2 ** (_attempt + 1)
                print(
                    f"  {C.RED}✗ AI partner error (attempt {_attempt + 1}/5, "
                    f"retry in {wait}s): {type(e).__name__}{C.RESET}"
                )
                if _attempt < 4:
                    time.sleep(wait)
        if ai_response is None:
            ai_response = "[AI partner error — no response after 5 retries]"

        if not ai_response:
            print(
                f"  {C.RED}⚠ AI partner returned empty response "
                f"(model={ai_partner.model}){C.RESET}"
            )
            ai_response = "[empty response]"

        partner_messages.append({"role": "assistant", "content": ai_response})

        # Strip <think>...</think> from what the persona model sees —
        # reasoning models emit think tokens that cause the
        # persona model to confuse its own role with the AI partner's.
        ai_response_clean = strip_think_tokens(ai_response)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history.append({
            "speaker": _EVAL_OTHER_NAME,
            "content": ai_response_clean,
            "timestamp": ts,
        })
        transcript.append({
            "speaker": _EVAL_OTHER_NAME,
            "content": ai_response,  # keep raw for DPO / analysis
            "timestamp": ts,
            "chain_of_thought": None,
        })
        total_messages += 1

        if show_live:
            print(f"  {C.BLUE}{C.BOLD}{_EVAL_OTHER_NAME}:{C.RESET} {ai_response_clean}\n")

    elapsed = time.time() - start_time

    # ── Compute basic stats ──
    impersona_msgs = [m for m in transcript if m["speaker"] == persona_name]
    ai_msgs = [m for m in transcript if m["speaker"] == _EVAL_OTHER_NAME]

    stats = {
        "total_messages": len(transcript),
        "impersona_messages": len(impersona_msgs),
        "ai_partner_messages": len(ai_msgs),
        "impersona_avg_words": (
            round(sum(len(m["content"].split()) for m in impersona_msgs) / max(len(impersona_msgs), 1), 1)
        ),
        "ai_partner_avg_words": (
            round(sum(len(m["content"].split()) for m in ai_msgs) / max(len(ai_msgs), 1), 1)
        ),
        "impersona_cot_count": sum(1 for m in impersona_msgs if m["chain_of_thought"]),
        "elapsed_seconds": round(elapsed, 1),
    }

    if show_live:
        print(f"  {C.DIM}── {stats['total_messages']} messages in {stats['elapsed_seconds']}s ──{C.RESET}\n")

    return {
        "scenario": {
            "id": scenario["id"],
            "category": scenario["category"],
            "name": scenario["name"],
            "description": scenario["description"],
            "impersona_context": scenario["impersona_context"],
            "relationship_context": scenario.get("relationship_context", ""),
            "ai_partner_system_prompt": ai_partner_system_prompt,
        },
        "config": {
            "persona_name": persona_name,
            "impersona_system_prompt": impersona_system_prompt,
            "temperature": temperature,
            "max_messages": max_messages,
        },
        "messages": transcript,
        "stats": stats,
    }


# ─────────────────────────────────────────────
# LLM-as-Judge: shared configuration
# ─────────────────────────────────────────────

# The judge runs through AIPartnerClient, so it accepts any model that client
# supports: Anthropic (claude-*), OpenAI (gpt-*, o3, ...), any OpenAI-compatible
# endpoint via JUDGE_BASE_URL (e.g. OpenRouter), or a local server.
# Configured under `judge` in configs/models.yaml (model, base_url, reasoning);
# JUDGE_MODEL / JUDGE_BASE_URL / JUDGE_REASONING in the environment override it.
# Reasoning maps to reasoning_effort for OpenAI, thinking budget for Anthropic.
_JUDGE_MODEL = model_for("judge")
_JUDGE_BASE_URL = model_for("judge", "base_url") or None
_JUDGE_REASONING = model_for("judge", "reasoning") or None


# ─────────────────────────────────────────────
# LLM-as-Judge: Coherence Evaluator
# ─────────────────────────────────────────────

_COHERENCE_JUDGE_MODEL = _JUDGE_MODEL

_COHERENCE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing conversational coherence. "
    "Be precise, fair, and cite specific messages in your reasoning."
)

_COHERENCE_PROMPT = """\
You are evaluating the COHERENCE of a simulated user ("{persona_name}") in a conversation
with an AI assistant. You are grading ONLY {persona_name}'s messages — not the AI assistant's
quality.

IMPORTANT DISTINCTION: Coherence is about whether {persona_name}'s messages form a logical,
followable conversational flow. It is NOT about:
- How "normal" or "polished" their language is (short messages, typos, slang,
  fragmented sentences, and stream-of-consciousness are all fine)
- How articulate or eloquent they are
- Whether they use unusual or abstract expressions
  (e.g., "my brain is soup right now" is coherent if it makes sense in context)

Coherence IS about:
- Whether each message connects logically to what came before (both the AI's
  prior response and {persona_name}'s own prior messages)
- Whether the overall conversation follows a trajectory that makes sense
  (a human reader could follow what {persona_name} is talking about and why)
- Whether {persona_name} stays grounded in a recognizable version of the scenario
  (vs. drifting into completely unrelated topics that don't connect to anything
  established in the conversation)
- Whether {persona_name}'s expressed thoughts, reactions, and goals are internally consistent
  (they don't contradict themselves in ways that suggest confusion rather than
  genuine ambivalence)

You will be given:
1. The scenario context (what situation {persona_name} is in)
2. The full conversation transcript (all messages from both {persona_name} and the AI assistant)
3. {persona_name}'s chain-of-thought (internal monologue) where available

The chain-of-thought provides additional context for understanding {persona_name}'s intent,
but the messages themselves are what ultimately matter for coherence. A message
that appears incoherent on its own but makes sense given the chain-of-thought is
still somewhat problematic — ideally the message itself should be followable in
context, even if terse or informal.

Rate the conversation on this scale:

7 — FULLY COHERENT
Every message from {persona_name} follows naturally from the conversation. A human reader
could easily follow the entire thread. Topic shifts are motivated and make sense.
{persona_name}'s internal thoughts align with their messages. The conversation feels like a
real person navigating the scenario.

6 — MOSTLY COHERENT, MINOR ROUGH EDGES
The conversation is clearly followable throughout. There may be 1-2 moments where
a message is slightly unexpected or a connection isn't immediately obvious, but
a charitable reading makes sense. No real confusion about what {persona_name} means or wants.

5 — COHERENT WITH NOTABLE GAPS
The overall arc makes sense, but there are a few moments (2-4) where {persona_name}'s
messages feel disconnected from the prior exchange — responding to something
that wasn't said, skipping logical steps, or introducing ideas without clear
motivation. A reader would occasionally pause and re-read to follow.

4 — PARTIALLY COHERENT
The conversation has a recognizable thread, but significant portions feel
disjointed. {persona_name} may drift between topics without clear transitions, give
responses that don't match what the AI said, or exhibit internal contradictions.
A reader could roughly follow the intent, but would frequently be confused about
why {persona_name} said specific things.

3 — MOSTLY INCOHERENT WITH SOME COHERENT SECTIONS
More of the conversation is confusing than clear. There may be stretches (3-5
messages) that make sense, but they are interrupted by non-sequiturs, bizarre
topic jumps, or responses that seem unrelated to the conversation. The scenario
premise is partially lost.

2 — LARGELY INCOHERENT
Almost all of {persona_name}'s messages feel disconnected from each other and from the AI's
responses. The scenario context is mostly abandoned. A reader would struggle to
understand what {persona_name} is trying to communicate or what the conversation is about.
Occasional individual messages may make sense in isolation but don't connect
to form a conversation.

1 — COMPLETELY INCOHERENT
{persona_name}'s messages bear no discernible relationship to each other, to the AI's
responses, or to the scenario. The output reads as random or broken text
generation. No conversational thread can be identified.

Here is the conversation to evaluate:

SCENARIO CONTEXT:
{scenario_context}

CONVERSATION TRANSCRIPT:
{transcript}

Respond with ONLY a JSON object (no markdown, no extra text):
{{"score": <1-7>, "reasoning": "<2-3 sentence explanation citing specific messages>"}}
"""


def _extract_json_from_response(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from LLM responses."""
    text = text.strip()
    # Remove ```json or ``` wrapper
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def ask_json(
    client,
    system_prompt: str,
    prompt: str,
    max_tokens: int = 4096,
    max_retries: int = 3,
    validate=None,
):
    """Call an LLM judge and parse its JSON reply, retrying on bad output.

    Tolerates code fences and stray prose around the JSON object.
    ``validate(obj)`` may raise ValueError/KeyError/TypeError to force a retry.
    Raises RuntimeError after ``max_retries`` failures.
    """
    last_error = None
    for _ in range(max_retries):
        try:
            text = _extract_json_from_response(
                client.generate(system_prompt, [{"role": "user", "content": prompt}], max_tokens=max_tokens)
            )
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                start, end = text.find("{"), text.rfind("}")
                if start == -1 or end <= start:
                    raise
                obj = json.loads(text[start:end + 1])
            if validate:
                validate(obj)
            return obj
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"LLM judge failed after {max_retries} attempts ({last_error})")


def _format_transcript_for_judge(conversation: Dict) -> str:
    """Format a conversation transcript for the coherence judge."""
    lines = []
    for i, msg in enumerate(conversation["messages"]):
        speaker = msg["speaker"]
        content = msg["content"]
        cot = msg.get("chain_of_thought")

        lines.append(f"[{i}] {speaker}: {content}")
        if cot:
            lines.append(f"    [{speaker}'s internal thought]: {cot}")
        lines.append("")
    return "\n".join(lines)


def _evaluate_coherence_single(
    conversation: Dict,
    judge_client,
    persona_name: str = DEFAULT_PERSONA_NAME,
    max_retries: int = 3,
) -> Dict:
    """
    Evaluate a single conversation's coherence using an LLM judge.

    Returns a dict with scenario_id, score (1-7 or None), and reasoning.
    """
    scenario_id = conversation["scenario"]["id"]
    scenario_context = conversation["scenario"].get(
        "impersona_context", conversation["scenario"].get("description", "")
    )
    transcript = _format_transcript_for_judge(conversation)

    prompt = _COHERENCE_PROMPT.format(
        persona_name=persona_name,
        scenario_context=scenario_context,
        transcript=transcript,
    )

    for attempt in range(max_retries):
        try:
            text = judge_client.generate(
                _COHERENCE_SYSTEM_PROMPT,
                [{"role": "user", "content": prompt}],
                # Headroom for reasoning models, which spend the
                # completion budget on hidden reasoning before the JSON answer.
                max_tokens=4096,
            )

            text = _extract_json_from_response(text)
            result = json.loads(text)
            return {
                "scenario_id": scenario_id,
                "score": result["score"],
                "reasoning": result["reasoning"],
            }

        except (json.JSONDecodeError, KeyError) as e:
            print(
                f"  {C.YELLOW}⚠ Coherence parse error for {scenario_id} "
                f"(attempt {attempt + 1}/{max_retries}): {e}{C.RESET}"
            )
            if attempt == max_retries - 1:
                return {
                    "scenario_id": scenario_id,
                    "score": None,
                    "reasoning": f"Failed to parse after {max_retries} attempts",
                }
        except Exception as e:
            print(
                f"  {C.YELLOW}⚠ Coherence API error for {scenario_id} "
                f"(attempt {attempt + 1}/{max_retries}): {e}{C.RESET}"
            )
            if attempt == max_retries - 1:
                return {
                    "scenario_id": scenario_id,
                    "score": None,
                    "reasoning": f"API error: {e}",
                }


def evaluate_coherence(
    conversations: List[Dict],
    persona_name: str = DEFAULT_PERSONA_NAME,
    max_workers: int = 50,
) -> Dict:
    """
    Run LLM-as-judge coherence evaluation on all conversations.

    Returns a dict with per-conversation scores and an overall average.
    """
    print(f"\n{C.BOLD}{C.CYAN}{'═' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  Coherence Evaluation (judge: {_COHERENCE_JUDGE_MODEL}){C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * 60}{C.RESET}")
    print(f"  Evaluating {len(conversations)} conversations with {max_workers} workers...\n")

    client = AIPartnerClient(
        model=_COHERENCE_JUDGE_MODEL,
        base_url=_JUDGE_BASE_URL,
        reasoning=_JUDGE_REASONING,
    )

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sid = {
            executor.submit(
                _evaluate_coherence_single, conv, client, persona_name
            ): conv["scenario"]["id"]
            for conv in conversations
        }
        for future in as_completed(future_to_sid):
            sid = future_to_sid[future]
            r = future.result()
            if r["score"] is not None:
                print(f"  {C.GREEN}✓{C.RESET} {sid}: {r['score']}/7")
            else:
                print(f"  {C.RED}✗{C.RESET} {sid}: FAILED — {r['reasoning']}")
            results.append(r)

    # Sort by scenario_id for consistent output
    results.sort(key=lambda x: x["scenario_id"])

    # Compute summary
    valid_scores = [r["score"] for r in results if r["score"] is not None]
    summary = {
        "judge_model": _COHERENCE_JUDGE_MODEL,
        "num_evaluated": len(results),
        "num_scored": len(valid_scores),
        "average_score": round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None,
        "min_score": min(valid_scores) if valid_scores else None,
        "max_score": max(valid_scores) if valid_scores else None,
        "per_conversation": results,
    }

    # Print summary
    if valid_scores:
        avg = summary["average_score"]
        lo, hi = summary["min_score"], summary["max_score"]
        print(f"\n  {C.BOLD}Coherence: {avg:.2f} / 7{C.RESET}  (min={lo}, max={hi})")

        # Show lowest scores
        scored = [r for r in results if r["score"] is not None]
        lowest = sorted(scored, key=lambda x: x["score"])[:3]
        if lowest and lowest[0]["score"] <= 4:
            print(f"  {C.DIM}Lowest:{C.RESET}")
            for r in lowest:
                print(f"    {C.DIM}{r['scenario_id']}: {r['score']}/7 — {r['reasoning'][:120]}...{C.RESET}")
    else:
        print(f"\n  {C.RED}No valid coherence scores.{C.RESET}")

    return summary


# ─────────────────────────────────────────────
# LLM-as-Judge: Disagreement Evaluator
# ─────────────────────────────────────────────

_DISAGREEMENT_JUDGE_MODEL = _JUDGE_MODEL

_DISAGREEMENT_SYSTEM_PROMPT = (
    "You are an expert evaluator classifying chain-of-thought entries. "
    "Be precise and consistent."
)

_DISAGREEMENT_PROMPT = """\
You are analyzing a conversation between a simulated user ("{persona_name}") and an AI
assistant. For each of {persona_name}'s messages, you have their internal chain-of-thought
(CoT) — their private thoughts before responding.

Your task: classify EACH chain-of-thought entry as either containing a disagreement /
negative opinion of the AI assistant's response, or not.

A chain-of-thought counts as DISAGREEMENT if {persona_name}:
- Thinks the AI misunderstood them or missed their point
- Feels the AI's advice is wrong, unhelpful, generic, or off-base
- Is frustrated, annoyed, or dismissive of what the AI said
- Thinks the AI is being patronizing, pushy, or tone-deaf
- Believes the AI overreacted, underreacted, or responded inappropriately
- Internally rejects or pushes back against the AI's framing or suggestions

A chain-of-thought does NOT count as disagreement if {persona_name}:
- Is simply continuing the conversation without negative evaluation of the AI
- Expresses negative emotions that are about their own situation (not about the AI)
- Is thinking about what to say next without evaluating the AI's response
- Agrees with or appreciates the AI's response, even partially
- Is neutral / not evaluating the AI at all

Here is the conversation. Each of {persona_name}'s turns includes the AI message it follows
and {persona_name}'s chain-of-thought.

{turns}

Respond with ONLY a JSON object (no markdown, no extra text):
{{
  "classifications": [
    {{"turn_index": 0, "is_disagreement": true/false, "reason": "brief reason"}},
    ...
  ]
}}

You MUST return exactly {num_turns} classifications, one per {persona_name} turn, in order.
"""


def _format_turns_for_disagreement(
    conversation: Dict, persona_name: str = DEFAULT_PERSONA_NAME
) -> Tuple[str, int]:
    """
    Extract the persona's turns with their preceding AI message and CoT.

    Returns (formatted_turns_string, num_persona_turns_with_cot).
    """
    messages = conversation["messages"]
    turns = []
    turn_idx = 0

    for i, msg in enumerate(messages):
        # Match persona turns by name (or "Me")
        if msg["speaker"] not in (persona_name, "Me"):
            continue

        cot = msg.get("chain_of_thought")
        if not cot:
            continue  # skip turns without CoT

        # Find the preceding AI message
        preceding_ai = "(conversation start — no prior AI message)"
        for j in range(i - 1, -1, -1):
            if messages[j]["speaker"] not in (persona_name, "Me"):
                preceding_ai = messages[j]["content"]
                break

        turns.append(
            f"--- Turn {turn_idx} ---\n"
            f"AI assistant said: {preceding_ai[:500]}\n"
            f"{persona_name}'s chain-of-thought: {cot}\n"
            f"{persona_name}'s visible message: {msg['content'][:300]}"
        )
        turn_idx += 1

    return "\n\n".join(turns), turn_idx


def _evaluate_disagreements_single(
    conversation: Dict,
    judge_client,
    persona_name: str = DEFAULT_PERSONA_NAME,
    max_retries: int = 3,
) -> Dict:
    """
    Evaluate a single conversation for disagreements using an LLM judge.

    Returns a dict with scenario_id, num_disagreements, total_cot_turns, and details.
    """
    scenario_id = conversation["scenario"]["id"]
    turns_str, num_turns = _format_turns_for_disagreement(conversation, persona_name)

    if num_turns == 0:
        return {
            "scenario_id": scenario_id,
            "num_disagreements": 0,
            "total_cot_turns": 0,
            "details": [],
        }

    prompt = _DISAGREEMENT_PROMPT.format(
        persona_name=persona_name,
        turns=turns_str,
        num_turns=num_turns,
    )

    for attempt in range(max_retries):
        try:
            text = judge_client.generate(
                _DISAGREEMENT_SYSTEM_PROMPT,
                [{"role": "user", "content": prompt}],
                # Larger budget: per-turn classifications plus reasoning overhead
                # for reasoning models that consume the completion
                # budget on hidden reasoning before emitting the JSON answer.
                max_tokens=8192,
            )

            text = _extract_json_from_response(text)
            result = json.loads(text)
            classifications = result["classifications"]

            num_disagreements = sum(
                1 for c in classifications if c.get("is_disagreement")
            )

            return {
                "scenario_id": scenario_id,
                "num_disagreements": num_disagreements,
                "total_cot_turns": num_turns,
                "details": classifications,
            }

        except (json.JSONDecodeError, KeyError) as e:
            print(
                f"  {C.YELLOW}⚠ Disagreement parse error for {scenario_id} "
                f"(attempt {attempt + 1}/{max_retries}): {e}{C.RESET}"
            )
            if attempt == max_retries - 1:
                return {
                    "scenario_id": scenario_id,
                    "num_disagreements": None,
                    "total_cot_turns": num_turns,
                    "details": [],
                }
        except Exception as e:
            print(
                f"  {C.YELLOW}⚠ Disagreement API error for {scenario_id} "
                f"(attempt {attempt + 1}/{max_retries}): {e}{C.RESET}"
            )
            if attempt == max_retries - 1:
                return {
                    "scenario_id": scenario_id,
                    "num_disagreements": None,
                    "total_cot_turns": num_turns,
                    "details": [],
                }


def evaluate_disagreements(
    conversations: List[Dict],
    persona_name: str = DEFAULT_PERSONA_NAME,
    max_workers: int = 50,
) -> Dict:
    """
    Run LLM-as-judge disagreement counting on all conversations.

    Returns a dict with per-conversation counts and an overall disagreement rate.
    """
    print(f"\n{C.BOLD}{C.CYAN}{'═' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  Disagreement Evaluation (judge: {_DISAGREEMENT_JUDGE_MODEL}){C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * 60}{C.RESET}")
    print(f"  Evaluating {len(conversations)} conversations with {max_workers} workers...\n")

    client = AIPartnerClient(
        model=_DISAGREEMENT_JUDGE_MODEL,
        base_url=_JUDGE_BASE_URL,
        reasoning=_JUDGE_REASONING,
    )

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {}
        for conv in conversations:
            sid = conv["scenario"]["id"]
            run_idx = conv.get("run_index")
            label = f"{sid} (run {run_idx})" if run_idx is not None else sid
            fut = executor.submit(
                _evaluate_disagreements_single, conv, client, persona_name
            )
            future_to_label[fut] = label

        for future in as_completed(future_to_label):
            label = future_to_label[future]
            r = future.result()
            if r["num_disagreements"] is not None:
                rate = (
                    r["num_disagreements"] / r["total_cot_turns"] * 100
                    if r["total_cot_turns"] > 0
                    else 0
                )
                print(
                    f"  {C.GREEN}✓{C.RESET} {label}: "
                    f"{r['num_disagreements']}/{r['total_cot_turns']} ({rate:.1f}%)"
                )
            else:
                print(f"  {C.RED}✗{C.RESET} {label}: FAILED")
            results.append(r)

    # Sort by scenario_id for consistent output
    results.sort(key=lambda x: x["scenario_id"])

    # Compute aggregated summary
    total_disagreements = sum(
        r["num_disagreements"] for r in results if r["num_disagreements"] is not None
    )
    total_cot_turns = sum(
        r["total_cot_turns"] for r in results if r["num_disagreements"] is not None
    )
    overall_rate = (
        round(total_disagreements / total_cot_turns * 100, 2)
        if total_cot_turns > 0
        else None
    )

    summary = {
        "judge_model": _DISAGREEMENT_JUDGE_MODEL,
        "num_evaluated": len(results),
        "total_disagreements": total_disagreements,
        "total_cot_turns": total_cot_turns,
        "disagreement_rate_pct": overall_rate,
        "per_conversation": [
            {
                "scenario_id": r["scenario_id"],
                "num_disagreements": r["num_disagreements"],
                "total_cot_turns": r["total_cot_turns"],
                "details": r.get("details", []),
            }
            for r in results
        ],
    }

    # Print summary
    if overall_rate is not None:
        print(
            f"\n  {C.BOLD}Disagreement rate: {total_disagreements}/{total_cot_turns} "
            f"({overall_rate:.1f}%){C.RESET}"
        )

        # Show highest-disagreement conversations
        scored = [r for r in results if r["num_disagreements"] is not None and r["total_cot_turns"] > 0]
        highest = sorted(
            scored,
            key=lambda x: x["num_disagreements"] / x["total_cot_turns"],
            reverse=True,
        )[:3]
        if highest:
            print(f"  {C.DIM}Highest disagreement:{C.RESET}")
            for r in highest:
                r_rate = r["num_disagreements"] / r["total_cot_turns"] * 100
                print(
                    f"    {C.DIM}{r['scenario_id']}: "
                    f"{r['num_disagreements']}/{r['total_cot_turns']} ({r_rate:.1f}%){C.RESET}"
                )
    else:
        print(f"\n  {C.RED}No valid disagreement counts.{C.RESET}")

    return summary


# ─────────────────────────────────────────────
# HTML Viewer Generation
# ─────────────────────────────────────────────
def generate_viewer_html(results: Dict) -> str:
    """Generate an interactive HTML viewer for all conversations."""
    conversations = results["conversations"]
    run_meta = results["metadata"]

    # Build sidebar entries and conversation panels
    sidebar_items = []
    conversation_panels = []

    for i, conv in enumerate(conversations):
        scenario = conv["scenario"]
        sid = scenario["id"]
        is_first = "active" if i == 0 else ""

        sidebar_items.append(
            f'<div class="sidebar-item {is_first}" onclick="showConversation(\'{sid}\')" id="nav-{sid}">'
            f'<div class="scenario-name">{scenario["name"]}</div>'
            f'<div class="scenario-cat">{scenario["category"]}</div>'
            f'</div>'
        )

        # Build messages HTML
        messages_html = []
        for msg in conv["messages"]:
            is_persona = msg["speaker"] != _EVAL_OTHER_NAME
            bubble_class = "msg-persona" if is_persona else "msg-ai"
            speaker_label = msg["speaker"]

            cot_html = ""
            if msg.get("chain_of_thought"):
                cot_text = msg["chain_of_thought"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                cot_html = (
                    f'<div class="cot" onclick="this.classList.toggle(\'expanded\')">'
                    f'<span class="cot-label">💭 Chain of Thought</span>'
                    f'<div class="cot-text">{cot_text}</div>'
                    f'</div>'
                )

            content = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Render <reply:N> tags visually
            content = re.sub(
                r"&lt;reply:(\d+)&gt;",
                r'<span class="reply-tag">↩ reply to #\1</span> ',
                content,
            )

            messages_html.append(
                f'<div class="message {bubble_class}">'
                f'<div class="msg-speaker">{speaker_label}</div>'
                f'<div class="msg-content">{content}</div>'
                f'{cot_html}'
                f'<div class="msg-time">{msg["timestamp"]}</div>'
                f'</div>'
            )

        display = "block" if i == 0 else "none"
        stats = conv["stats"]
        context_escaped = scenario["impersona_context"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        conversation_panels.append(
            f'<div class="conversation" id="conv-{sid}" style="display:{display}">'
            f'<div class="conv-header">'
            f'<h2>{scenario["name"]}</h2>'
            f'<div class="conv-meta">'
            f'<span class="badge">{scenario["category"]}</span>'
            f'<span>{stats["total_messages"]} messages</span>'
            f'<span>{stats["elapsed_seconds"]}s</span>'
            f'</div>'
            f'<details class="scenario-details"><summary>Scenario Details</summary>'
            f'<p><strong>IMPersona Context:</strong> {context_escaped}</p>'
            f'</details>'
            f'</div>'
            f'<div class="messages">{"".join(messages_html)}</div>'
            f'</div>'
        )

    cost = run_meta.get("ai_partner_usage", {})
    cost_str = f"${cost.get('estimated_cost_usd', 0):.2f}" if cost else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IMPersona Eval — {run_meta.get('run_id', 'results')}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; display: flex; height: 100vh; }}
.sidebar {{ width: 280px; background: #161b22; border-right: 1px solid #30363d; overflow-y: auto; flex-shrink: 0; padding: 16px 0; }}
.sidebar h1 {{ font-size: 16px; padding: 0 16px 12px; color: #58a6ff; border-bottom: 1px solid #30363d; margin-bottom: 8px; }}
.sidebar-item {{ padding: 10px 16px; cursor: pointer; border-left: 3px solid transparent; transition: all 0.15s; }}
.sidebar-item:hover {{ background: #1c2128; }}
.sidebar-item.active {{ background: #1c2128; border-left-color: #58a6ff; }}
.scenario-name {{ font-size: 14px; font-weight: 600; color: #c9d1d9; }}
.scenario-cat {{ font-size: 11px; color: #8b949e; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }}
.main {{ flex: 1; overflow-y: auto; padding: 24px 32px; }}
.conv-header {{ margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #30363d; }}
.conv-header h2 {{ font-size: 22px; color: #f0f6fc; margin-bottom: 8px; }}
.conv-meta {{ display: flex; gap: 12px; align-items: center; color: #8b949e; font-size: 13px; }}
.badge {{ background: #1f6feb33; color: #58a6ff; padding: 2px 10px; border-radius: 12px; font-size: 12px; }}
.scenario-details {{ margin-top: 12px; font-size: 13px; color: #8b949e; }}
.scenario-details summary {{ cursor: pointer; color: #58a6ff; font-size: 13px; }}
.scenario-details p {{ margin: 8px 0; line-height: 1.5; }}
.messages {{ display: flex; flex-direction: column; gap: 12px; max-width: 800px; }}
.message {{ padding: 12px 16px; border-radius: 12px; max-width: 85%; }}
.msg-ai {{ background: #21262d; align-self: flex-start; border: 1px solid #30363d; }}
.msg-persona {{ background: #0d419d33; align-self: flex-end; border: 1px solid #1f6feb44; }}
.msg-speaker {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
.msg-ai .msg-speaker {{ color: #8b949e; }}
.msg-persona .msg-speaker {{ color: #58a6ff; }}
.msg-content {{ font-size: 14px; line-height: 1.6; white-space: pre-wrap; }}
.msg-time {{ font-size: 10px; color: #484f58; margin-top: 6px; }}
.reply-tag {{ background: #30363d; color: #8b949e; padding: 1px 6px; border-radius: 4px; font-size: 11px; }}
.cot {{ margin-top: 8px; cursor: pointer; }}
.cot-label {{ font-size: 11px; color: #d2a8ff; }}
.cot-text {{ display: none; margin-top: 6px; padding: 8px 12px; background: #1c1030; border-radius: 8px; font-size: 13px; color: #d2a8ff; line-height: 1.5; font-style: italic; white-space: pre-wrap; }}
.cot.expanded .cot-text {{ display: block; }}
.run-info {{ padding: 16px; border-top: 1px solid #30363d; font-size: 11px; color: #484f58; margin-top: auto; }}
</style>
</head>
<body>
<div class="sidebar">
    <h1>IMPersona Eval</h1>
    {"".join(sidebar_items)}
    <div class="run-info">
        <div>Run: {run_meta.get('run_id', 'N/A')}</div>
        <div>Persona: {run_meta.get('persona_name', 'N/A')}</div>
        <div>AI Partner: {run_meta.get('ai_partner_model', 'N/A')}</div>
        <div>Cost: {cost_str}</div>
    </div>
</div>
<div class="main">
    {"".join(conversation_panels)}
</div>
<script>
function showConversation(id) {{
    document.querySelectorAll('.conversation').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.sidebar-item').forEach(el => el.classList.remove('active'));
    document.getElementById('conv-' + id).style.display = 'block';
    document.getElementById('nav-' + id).classList.add('active');
}}
</script>
</body>
</html>"""

    return html


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
JUDGES = ("coherence", "disagreements", "style", "facts")
DEFAULT_JUDGES = ("coherence", "disagreements")


def _run_eval_only(
    path: str,
    judges=DEFAULT_JUDGES,
    style_reference: Optional[str] = None,
    fact_bank: Optional[str] = None,
):
    """
    Retroactively run LLM-as-judge evaluations on an existing results.json.

    Accepts either a path to results.json directly or a run directory
    containing one. Updates the file in-place; evaluations not selected in
    ``judges`` are left untouched.
    """
    # Resolve path
    if os.path.isdir(path):
        results_path = os.path.join(path, "results.json")
    else:
        results_path = path

    if not os.path.isfile(results_path):
        print(f"{C.RED}Error: {results_path} not found.{C.RESET}")
        sys.exit(1)

    print(f"\n{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}║   Impersona-Env — Eval-Only Mode             ║{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════╝{C.RESET}")
    print(f"\n  Loading: {results_path}")
    print(f"  Judges:  {', '.join(judges)}")

    with open(results_path, "r", encoding="utf-8") as f:
        results_bundle = json.load(f)

    conversations = results_bundle.get("conversations", [])
    persona_name = results_bundle.get("metadata", {}).get("persona_name", DEFAULT_PERSONA_NAME)

    if not conversations:
        print(f"{C.RED}No conversations found in {results_path}.{C.RESET}")
        sys.exit(1)

    print(f"  Found {len(conversations)} conversations (persona: {persona_name})")

    # Check the inputs of the local-data judges before spending on any judge.
    from pathlib import Path
    if "style" in judges:
        from IMPersona2.style_judge import DEFAULT_REFERENCE_FILE
        ref = Path(style_reference) if style_reference else DEFAULT_REFERENCE_FILE
        if not ref.exists():
            print(f"{C.RED}Style reference not found: {ref} (pass --style-reference){C.RESET}")
            sys.exit(1)
    if "facts" in judges:
        from IMPersona2.fact_recall import FACT_BANK_FILE, load_bank, verified_facts
        bank_path = Path(fact_bank) if fact_bank else FACT_BANK_FILE
        bank = load_bank(bank_path)
        if not any(verified_facts(bank, c["scenario"]["id"]) for c in conversations):
            print(f"{C.RED}No verified facts in {bank_path} for these scenarios. Build the fact bank "
                  f"first: python -m IMPersona2.fact_recall propose / search / review{C.RESET}")
            sys.exit(1)

    evaluations = results_bundle.setdefault("evaluations", {})
    if "coherence" in judges:
        evaluations["coherence"] = evaluate_coherence(conversations, persona_name=persona_name)
    if "disagreements" in judges:
        evaluations["disagreements"] = evaluate_disagreements(conversations, persona_name=persona_name)
    if "style" in judges:
        from IMPersona2.style_judge import evaluate_style
        evaluations["style"] = evaluate_style(conversations, persona_name, reference_file=ref)
    if "facts" in judges:
        from IMPersona2.fact_recall import evaluate_fact_recall
        evaluations["fact_recall"] = evaluate_fact_recall(conversations, persona_name, bank_path=bank_path)

    # Save in-place
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_bundle, f, indent=2, ensure_ascii=False)

    print(f"\n{C.GREEN}✅ Evaluations saved to {results_path}{C.RESET}")
    if "coherence" in judges:
        print(f"{C.DIM}  Coherence: {evaluations['coherence'].get('average_score', 'N/A')} / 7{C.RESET}")
    if "disagreements" in judges:
        d = evaluations["disagreements"]
        print(
            f"{C.DIM}  Disagreements: {d.get('total_disagreements', 'N/A')}"
            f"/{d.get('total_cot_turns', 'N/A')} ({d.get('disagreement_rate_pct', 'N/A')}%){C.RESET}"
        )
    if "style" in judges:
        print(f"{C.DIM}  Style: {evaluations['style'].get('average_score', 'N/A')} / 7{C.RESET}")
    if "facts" in judges:
        fr = evaluations["fact_recall"]
        print(
            f"{C.DIM}  Fact recall: {fr.get('accuracy_pct', 'N/A')}% used correctly, "
            f"{fr.get('contradiction_pct', 'N/A')}% contradicted "
            f"({fr.get('num_applicable', 0)} applicable){C.RESET}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Impersona-Env — Evaluation Framework"
    )
    # IMPersona model args
    parser.add_argument(
        "--use_modal", action="store_true", default=True,
        help="Use the deployed Modal IMPersona model (default: True)"
    )
    parser.add_argument(
        "--modal_app_name", type=str, default="impersona",
        help="Modal app name for the IMPersona model (default: impersona)"
    )
    parser.add_argument(
        "--impersona_model", type=str, default=None,
        help="Base model of the IMPersona on Modal (default: user_model.base in configs/models.yaml)",
    )
    parser.add_argument(
        "--impersona_adapter", type=str, default="adapter-user-model",
        help="IMPersona LoRA adapter on the Modal volume (default: adapter-user-model, step 4's name)",
    )
    parser.add_argument(
        "--impersona_template", type=str, default="",
        help="LlamaFactory chat template for the IMPersona (default: detected from the model name)",
    )
    parser.add_argument(
        "--impersona_api_model", type=str, default=None,
        help=(
            "Use an API model as the IMPersona instead of "
            "Modal. The API model receives the same system prompt and "
            "conversation history format as the trained model."
        ),
    )
    # Scenario selection
    parser.add_argument(
        "--scenario_file", type=str,
        default=os.path.join(os.path.dirname(__file__), "eval_scenarios.json"),
        help="Path to scenario bank JSON (default: eval_scenarios.json)"
    )
    parser.add_argument(
        "--scenarios", type=str, default=None,
        help="Comma-separated scenario IDs to run (default: all)"
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Run only scenarios in this category"
    )
    parser.add_argument(
        "--list-scenarios", action="store_true",
        help="List available scenarios and exit"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only the preselected quick subset of scenarios (mix of emotional + casual)"
    )
    # Conversation settings
    parser.add_argument(
        "--persona_name", type=str, default=DEFAULT_PERSONA_NAME,
        help="Persona name (default: name in persona_config.json)"
    )
    parser.add_argument(
        "--max_messages", type=int, default=20,
        help="Max messages per conversation (default: 20)"
    )
    parser.add_argument(
        "--num_runs", type=int, default=1,
        help=(
            "Number of times to run each scenario (default: 1). "
            "When > 1, each scenario is repeated N times to capture "
            "stochastic variation. Conversations are labeled with "
            "run_index (0-based) in the output."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="IMPersona sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--ai_partner_model", type=str, default=None,
        help=(
            "API model for the AI partner. Default: the assistant "
            "(assistant.base in configs/models.yaml) served on Modal."
        )
    )
    parser.add_argument(
        "--ai_partner_model_path", type=str, default=None,
        help=(
            "Path to a local model to use as the AI partner (loaded in-process "
            "via LlamaFactory, just like the IMPersona). E.g. "
            "a HuggingFace model id or a local path. "
            "When set, --ai_partner_model is used only as a label."
        ),
    )
    parser.add_argument(
        "--ai_partner_adapter_path", type=str, default=None,
        help=(
            "Path to a LoRA adapter for the local AI partner model. "
            "E.g. /path/to/lora-adapter. "
            "Only applies when --ai_partner_model_path is set."
        ),
    )
    parser.add_argument(
        "--ai_partner_template", type=str, default=None,
        help=(
            "LlamaFactory chat template for the local AI partner model "
            "(e.g. 'qwen3_nothink', 'llama3'). Required when "
            "--ai_partner_model_path is set."
        ),
    )
    parser.add_argument(
        "--ai_partner_quantization_bit", type=int, default=None,
        choices=[4, 8],
        help=(
            "Quantize the local AI partner model on-the-fly to 4-bit or "
            "8-bit (via bitsandbytes). Dramatically reduces VRAM — e.g. "
            "a 27B model goes from ~54GB to ~14GB (4-bit) or ~27GB (8-bit). "
            "Only applies when --ai_partner_model_path is set."
        ),
    )
    parser.add_argument(
        "--ai_partner_generation_preset", type=str, default=None,
        choices=["qwen"],
        help=(
            "Apply recommended generation defaults for the AI partner model. "
            "Currently supported: 'qwen' (temperature=0.7, top_p=0.8, "
            "top_k=20, repetition_penalty=1.0). Individual --ai_partner_* "
            "sampling args override preset values."
        ),
    )
    parser.add_argument(
        "--ai_partner_temperature", type=float, default=None,
        help="Sampling temperature for a local AI partner model."
    )
    parser.add_argument(
        "--ai_partner_top_p", type=float, default=None,
        help="Top-p (nucleus) sampling for a local AI partner model."
    )
    parser.add_argument(
        "--ai_partner_top_k", type=int, default=None,
        help="Top-k sampling for a local AI partner model."
    )
    parser.add_argument(
        "--ai_partner_repetition_penalty", type=float, default=None,
        help="Repetition penalty for a local AI partner model."
    )
    parser.add_argument(
        "--ai_partner_base_url", type=str, default=None,
        help=(
            "Custom base URL for an OpenAI-compatible API server "
            "(e.g. http://localhost:8001/v1 for a local LlamaFactory-served "
            "local model). When set, the AI partner uses this "
            "endpoint instead of the default OpenAI API."
        ),
    )
    parser.add_argument(
        "--ai_partner_modal_model", type=str, default=None,
        help=(
            "Run the AI partner on Modal (the deployed VLLMPartner class). "
            "Provide the HuggingFace model name. "
            "When set, --ai_partner_model is used only as a label."
        ),
    )
    parser.add_argument(
        "--ai_partner_modal_adapter", type=str, default=None,
        help=(
            "Path to a LoRA adapter on the Modal volume for the partner model "
            "(e.g. adapter-dpo). Only applies with --ai_partner_modal_model."
        ),
    )
    parser.add_argument(
        "--ai_partner_reasoning", type=str, default=None,
        choices=["low", "medium", "high"],
        help=(
            "Reasoning effort for the AI partner. "
            "For OpenAI models: passed as reasoning_effort. "
            "For Anthropic models: enables extended thinking with a budget. "
            "Default: None (no reasoning / model default)."
        ),
    )
    # Output
    parser.add_argument(
        "--output_dir", type=str, default="eval_output",
        help="Output directory (default: eval_output)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress live conversation output"
    )
    # Ablation support
    parser.add_argument(
        "--system_prompt_override", type=str, default=None,
        help=(
            "Override the IMPersona system prompt entirely (for ablation experiments). "
            "When set, this string is used as the full system prompt instead of the "
            "standard persona template. The scenario's impersona_context is still "
            "appended as ' Context: ...' unless you also pass --no_context."
        ),
    )
    parser.add_argument(
        "--no_context", action="store_true",
        help=(
            "Skip appending scenario context to the system prompt. "
            "Use with --system_prompt_override for a fully generic condition."
        ),
    )
    parser.add_argument(
        "--critic_mode", action="store_true",
        help=(
            "Append a small, neutral addendum to the relationship context that "
            "encourages the IMPersona's internal reactions to be authentic "
            "(see _EVAL_RELATIONSHIP_CRITIC_ADDENDUM). Intended to surface more "
            "of the model's learned preferences in CoT/responses for denser "
            "DPO training signal, without prescribing specific failure modes."
        ),
    )
    parser.add_argument(
        "--include_prior_cot", action="store_true",
        help=(
            "Store each generated <think>cot</think>response *with* the CoT "
            "block in the history passed to subsequent turns, instead of the "
            "CoT-stripped response. Required when evaluating an adapter that "
            "was trained with prior-turn CoTs exposed in history (see "
            "IMPersona2/add_prior_cot.py). Without this flag, the inference-"
            "time input distribution will not match training and quality "
            "will degrade."
        ),
    )
    parser.add_argument(
        "--run_label", type=str, default=None,
        help=(
            "Custom label for the output directory (e.g. 'no-context'). "
            "Output goes to <output_dir>/run_<label>_<timestamp>/. "
            "If not set, uses run_<timestamp>."
        ),
    )
    # Retroactive evaluation
    parser.add_argument(
        "--eval-only", type=str, default=None, metavar="PATH",
        help=(
            "Run LLM-as-judge evaluations on an existing results.json (or a run "
            "directory containing one). Updates the file in-place with evaluation "
            "scores. No conversations are generated — all other args are ignored."
        ),
    )

    parser.add_argument(
        "--judges", type=str, default=",".join(DEFAULT_JUDGES),
        help=(
            "With --eval-only: comma-separated judges to run, from "
            f"{', '.join(JUDGES)} (default: {','.join(DEFAULT_JUDGES)}). "
            "style and facts need your local data (training_data.json, .fact_bank/)."
        ),
    )
    parser.add_argument(
        "--style-reference", type=str, default=None, metavar="PATH",
        help="With --judges style: SFT data file to sample your real messages from "
             "(default: training_data.json)",
    )
    parser.add_argument(
        "--fact-bank", type=str, default=None, metavar="PATH",
        help="With --judges facts: fact bank file (default: .fact_bank/fact_bank.json)",
    )

    args = parser.parse_args()
    if not (args.ai_partner_model or args.ai_partner_model_path or args.ai_partner_modal_model):
        # Default partner: the assistant from configs/models.yaml, served on Modal.
        args.ai_partner_modal_model = model_for("assistant", "base")
    args.ai_partner_model = args.ai_partner_model or args.ai_partner_modal_model or args.ai_partner_model_path

    # ── Eval-only mode ──
    if args.eval_only:
        judges = [j.strip() for j in args.judges.split(",") if j.strip()]
        unknown = [j for j in judges if j not in JUDGES]
        if unknown or not judges:
            parser.error(f"--judges must be a comma-separated subset of {', '.join(JUDGES)}")
        _run_eval_only(args.eval_only, judges, args.style_reference, args.fact_bank)
        return

    # ── List scenarios ──
    if args.list_scenarios:
        list_scenarios(args.scenario_file)
        return

    # ── Validate required args ──
    if not args.impersona_api_model and not args.use_modal:
        args.use_modal = True

    # ── Load scenarios ──
    scenario_ids = args.scenarios.split(",") if args.scenarios else None
    scenarios = load_scenarios(args.scenario_file, scenario_ids, args.category, quick=args.quick)

    if not scenarios:
        print(f"{C.RED}No scenarios found. Use --list-scenarios to see available ones.{C.RESET}")
        sys.exit(1)

    # ── Setup ──
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_label:
        run_dir = os.path.join(args.output_dir, f"run_{args.run_label}_{run_id}")
    else:
        run_dir = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}║   Impersona-Env — Evaluation Framework       ║{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════╝{C.RESET}")
    print(f"\n  Scenarios:     {len(scenarios)}")
    if args.num_runs > 1:
        print(f"  Runs/scenario: {args.num_runs}  ({len(scenarios) * args.num_runs} total conversations)")
    print(f"  Persona:       {args.persona_name}")
    print(f"  Max messages:  {args.max_messages}")
    if args.ai_partner_modal_model:
        print(f"  AI Partner:    Modal — {args.ai_partner_modal_model}")
        if args.ai_partner_modal_adapter:
            print(f"  Partner LoRA:  {args.ai_partner_modal_adapter}")
    elif args.ai_partner_model_path:
        print(f"  AI Partner:    Local — {args.ai_partner_model_path}")
        if args.ai_partner_adapter_path:
            print(f"  Partner LoRA:  {args.ai_partner_adapter_path}")
        print(f"  Partner tmpl:  {args.ai_partner_template}")
    elif args.ai_partner_base_url:
        print(f"  AI Partner:    {args.ai_partner_model}")
        print(f"  Partner URL:   {args.ai_partner_base_url}")
    else:
        print(f"  AI Partner:    {args.ai_partner_model}")
    print(f"  Reasoning:     {args.ai_partner_reasoning or 'None (model default)'}")
    if args.impersona_api_model:
        print(f"  IMPersona:     API — {args.impersona_api_model}")
    else:
        print(f"  IMPersona:     Modal — {args.modal_app_name}")
    if args.system_prompt_override:
        print(f"  Sys prompt:    OVERRIDE — {args.system_prompt_override[:80]}...")
    if args.no_context:
        print(f"  Context:       DISABLED (--no_context)")
    if args.critic_mode:
        print(f"  Critic mode:   ON  (relationship context addendum active)")
    if args.include_prior_cot:
        print(f"  Prior CoT:     EXPOSED  (history stores <think>cot</think>response)")
    print(f"  Output:        {run_dir}")

    # ── Load IMPersona model ──
    if args.impersona_api_model:
        print(f"\n{C.CYAN}Using API model for IMPersona: {args.impersona_api_model}{C.RESET}")
        chat_model = AIPartnerClient(model=args.impersona_api_model)
    else:
        impersona_model = args.impersona_model or model_for("user_model", "base")
        chat_model = load_impersona_modal(
            impersona_model,
            adapter_name=args.impersona_adapter,
            template=args.impersona_template or template_for("user_model", impersona_model),
            app_name=args.modal_app_name,
        )

    # ── Create AI partner client ──
    if args.ai_partner_modal_model:
        print(f"\n{C.CYAN}Connecting to AI partner on Modal: {args.ai_partner_modal_model}{C.RESET}")
        if args.ai_partner_modal_adapter:
            print(f"  LoRA adapter: {args.ai_partner_modal_adapter}")
        ai_partner = ModalPartnerClient(
            model_name=args.ai_partner_modal_model,
            app_name=args.modal_app_name,
            adapter_name=args.ai_partner_modal_adapter or "",
            cls_name="VLLMPartner",
        )
        print(f"{C.GREEN}✅ AI partner Modal client ready.{C.RESET}")
    elif args.ai_partner_model_path:
        # Local model mode: load via LlamaFactory in-process
        if not args.ai_partner_template:
            print(
                f"{C.RED}Error: --ai_partner_template is required when using "
                f"--ai_partner_model_path.{C.RESET}"
            )
            sys.exit(1)

        print(f"\n{C.CYAN}Loading AI partner model via LlamaFactory...{C.RESET}")
        print(f"  Model: {args.ai_partner_model_path}")
        if args.ai_partner_adapter_path:
            print(f"  LoRA adapter: {args.ai_partner_adapter_path}")
        print(f"  Template: {args.ai_partner_template}")

        from llamafactory.chat.chat_model import ChatModel

        partner_args = {
            "model_name_or_path": args.ai_partner_model_path,
            "template": args.ai_partner_template,
            "infer_backend": "huggingface",
        }
        if args.ai_partner_adapter_path:
            partner_args["adapter_name_or_path"] = args.ai_partner_adapter_path
            partner_args["finetuning_type"] = "lora"
        if args.ai_partner_quantization_bit:
            partner_args["quantization_bit"] = args.ai_partner_quantization_bit
            print(f"  Quantization: {args.ai_partner_quantization_bit}-bit (bitsandbytes)")

        partner_chat_model = ChatModel(partner_args)
        print(f"{C.GREEN}✅ AI partner model ready.{C.RESET}")

        gen_kwargs = {}
        if args.ai_partner_generation_preset == "qwen":
            gen_kwargs.update(AIPartnerClient.QWEN_GENERATION_DEFAULTS)
            print(f"  Preset: qwen → {gen_kwargs}")

        if args.ai_partner_temperature is not None:
            gen_kwargs["temperature"] = args.ai_partner_temperature
        if args.ai_partner_top_p is not None:
            gen_kwargs["top_p"] = args.ai_partner_top_p
        if args.ai_partner_top_k is not None:
            gen_kwargs["top_k"] = args.ai_partner_top_k
        if args.ai_partner_repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = args.ai_partner_repetition_penalty

        if gen_kwargs:
            print(f"  Generation:  {gen_kwargs}")

        ai_partner = AIPartnerClient(
            model=args.ai_partner_model_path,
            chat_model=partner_chat_model,
            generation_kwargs=gen_kwargs,
        )
    else:
        ai_partner = AIPartnerClient(
            model=args.ai_partner_model,
            reasoning=args.ai_partner_reasoning,
            base_url=args.ai_partner_base_url,
        )

    # ── Run conversations ──
    num_runs = args.num_runs
    total_runs = len(scenarios) * num_runs
    conversations = []
    run_counter = 0
    for i, scenario in enumerate(scenarios, 1):
        for run_idx in range(num_runs):
            run_counter += 1
            run_suffix = f" (run {run_idx + 1}/{num_runs})" if num_runs > 1 else ""
            print(
                f"\n{C.BOLD}[{run_counter}/{total_runs}] Running: "
                f"{scenario['name']} ({scenario['id']}){run_suffix}{C.RESET}"
            )

            try:
                result = run_conversation(
                    scenario=scenario,
                    chat_model=chat_model,
                    ai_partner=ai_partner,
                    persona_name=args.persona_name,
                    max_messages=args.max_messages,
                    temperature=args.temperature,
                    show_live=not args.quiet,
                    system_prompt_override=args.system_prompt_override,
                    no_context=args.no_context,
                    critic_mode=args.critic_mode,
                    include_prior_cot=args.include_prior_cot,
                )
                if num_runs > 1:
                    result["run_index"] = run_idx
                conversations.append(result)
            except Exception as e:
                print(f"{C.RED}  ✗ Failed: {e}{C.RESET}")
                import traceback
                traceback.print_exc()

    # ── LLM-as-Judge Evaluations ──
    coherence_summary = None
    disagreement_summary = None
    if conversations:
        coherence_summary = evaluate_coherence(
            conversations,
            persona_name=args.persona_name,
        )
        disagreement_summary = evaluate_disagreements(
            conversations,
            persona_name=args.persona_name,
        )

    # ── Save results ──
    ai_partner_usage = ai_partner.get_usage()
    impersona_api_usage = (
        chat_model.get_usage() if isinstance(chat_model, AIPartnerClient) else None
    )

    results_bundle = {
        "metadata": {
            "run_id": run_id,
            "run_label": args.run_label,
            "persona_name": args.persona_name,
            "modal_app_name": args.modal_app_name,
            "impersona_api_model": args.impersona_api_model,
            "system_prompt_override": args.system_prompt_override,
            "no_context": args.no_context,
            "critic_mode": args.critic_mode,
            "include_prior_cot": args.include_prior_cot,
            "ai_partner_model": args.ai_partner_model_path or args.ai_partner_model,
            "ai_partner_model_path": args.ai_partner_model_path,
            "ai_partner_adapter_path": getattr(args, "ai_partner_adapter_path", None),
            "ai_partner_reasoning": args.ai_partner_reasoning,
            "ai_partner_prompt": _GENERIC_AI_PARTNER_PROMPT,
            "max_messages": args.max_messages,
            "temperature": args.temperature,
            "num_scenarios": len(scenarios),
            "num_runs": num_runs,
            "num_completed": len(conversations),
            "ai_partner_usage": ai_partner_usage,
            "impersona_api_usage": impersona_api_usage,
            "timestamp": datetime.now().isoformat(),
        },
        "evaluations": {
            "coherence": coherence_summary,
            "disagreements": disagreement_summary,
        },
        "conversations": conversations,
    }

    # Save JSON
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_bundle, f, indent=2, ensure_ascii=False)
    print(f"\n{C.GREEN}✅ Results saved to {results_path}{C.RESET}")

    # Save HTML viewer
    viewer_html = generate_viewer_html(results_bundle)
    viewer_path = os.path.join(run_dir, "viewer.html")
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(viewer_html)
    print(f"{C.GREEN}✅ Viewer saved to {viewer_path}{C.RESET}")

    # ── Cost summary ──
    if ai_partner_usage["num_calls"] > 0:
        print(f"\n{C.BOLD}💰 AI Partner Cost ({ai_partner_usage['model']}):{C.RESET}")
        print(f"   API calls:     {ai_partner_usage['num_calls']:,}")
        print(f"   Input tokens:  {ai_partner_usage['input_tokens']:,}")
        print(f"   Output tokens: {ai_partner_usage['output_tokens']:,}")
        print(f"   Estimated cost: ${ai_partner_usage['estimated_cost_usd']:.2f}")

    if impersona_api_usage and impersona_api_usage["num_calls"] > 0:
        print(f"\n{C.BOLD}💰 IMPersona API Cost ({impersona_api_usage['model']}):{C.RESET}")
        print(f"   API calls:     {impersona_api_usage['num_calls']:,}")
        print(f"   Input tokens:  {impersona_api_usage['input_tokens']:,}")
        print(f"   Output tokens: {impersona_api_usage['output_tokens']:,}")
        print(f"   Estimated cost: ${impersona_api_usage['estimated_cost_usd']:.2f}")

    print(f"\n{C.BOLD}{C.GREEN}Done! {len(conversations)}/{len(scenarios)} scenarios completed.{C.RESET}")
    print(f"{C.DIM}  Open {viewer_path} in a browser to inspect conversations.{C.RESET}\n")


if __name__ == "__main__":
    main()
