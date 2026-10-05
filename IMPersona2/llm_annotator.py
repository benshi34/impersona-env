"""
Impersona-Env — LLM-Based Conversation Annotator & CoT Generator
=================================================================

Three capabilities:

1. **Relationship analysis**: Produces a holistic relationship summary from full
   chat history (injected as system-prompt context during training).

2. **Conversation-level context**: Generates a brief situational context for each
   conversation (what was true about the world before the conversation started).

3. **Chain-of-Thought generation**: Generates a free-form internal monologue for
   each training example — what is "Me" noticing, feeling, and intending before
   they respond. Prepended as <think>...</think> so the model learns to
   reason about social context before responding.

Usage as a module:
    from IMPersona2.llm_annotator import LLMAnnotator
    annotator = LLMAnnotator()

    # Relationship analysis
    summary = annotator.analyze_relationship(conversations, other_speaker, ...)

    # Conversation-level context
    goals = annotator.generate_conversation_goals_batch(conversations, ...)

    # Chain-of-thought
    cot = annotator.generate_cot(history, target, ...)

Usage standalone (from the repo root):
    python -m IMPersona2.llm_annotator goals --conversations path/to/conversations.json
    python -m IMPersona2.llm_annotator relationship --conversations path/to/conversations.json
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from IMPersona2.experiment_config import default_persona_name
from IMPersona2.model_config import model_for

DEFAULT_PERSONA_NAME = default_persona_name()


def _is_anthropic_model(model: str) -> bool:
    """Return True if the model string indicates an Anthropic (Claude) model."""
    return model.lower().startswith("claude")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class AnnotatorConfig:
    """All knobs for the LLM annotator."""

    # Model & inference
    model: str = field(default_factory=lambda: model_for("annotation"))
    reasoning_effort: str = field(default_factory=lambda: model_for("annotation", "reasoning") or "medium")
    # OpenAI-compatible endpoint (e.g. OpenRouter); None routes by model prefix
    base_url: Optional[str] = field(default_factory=lambda: model_for("annotation", "base_url"))

    # Concurrency
    max_concurrency: int = 100         # max parallel API calls


# ─────────────────────────────────────────────
# Chain-of-Thought Prompts
# ─────────────────────────────────────────────

COT_SYSTEM_PROMPT = """You are generating internal monologue for a person named "Me" in a text conversation with "{other_speaker}".

PERSONA IDENTITY:
- "Me" is {persona_full_name} (also known as: {persona_nicknames}).
- If "Me" sends their own name (e.g. "{persona_full_name}!") as the first message, that is a SELF-INTRODUCTION to a new contact — "Me" is telling the recipient who they are. It is NOT the name of the other person.

You will be given:
1. A CONVERSATION CONTEXT — situational background that was true before the conversation started
2. A conversation history between "Me" and "{other_speaker}" — where prior "Me" messages may include inline 💭 annotations showing what "Me" was thinking before sending that message
3. The actual response that "Me" sent

Your job: write a brief, natural internal monologue that "Me" might have had RIGHT BEFORE sending that response.

If earlier "Me" messages have inline 💭 monologues, treat them as already-established reasoning. Do NOT repeat or re-state context, observations, or situational framing that was already covered in a prior monologue. Focus on what is NEW in this specific moment — new information, shifts in feeling, a new decision, or a reaction to something that just happened in the conversation.

This monologue should read like a real person's inner voice — not a clinical analysis. It should capture whatever is ACTUALLY driving the response in this specific moment. That might include:

- What I notice or read into {other_speaker}'s messages
- How I feel right now (interested, bored, anxious, amused, annoyed, neutral...)
- Whether I agree or disagree with something said, and how strongly
- How engaged I am — am I invested in this conversation or just going through the motions?
- Why I chose this particular tone, length, or level of effort
- Whether I'm being sarcastic, ironic, or playfully exaggerating — and what the REAL meaning is underneath. Analyze holistically: if the literal words don't match the tone cues (excessive "darn"s, over-the-top phrasing, emojis like 🥴, conspicuous overstatement), that's likely sarcasm or teasing. The monologue should capture the actual intent, not the surface-level reading.
- Any decision being made and why (even small ones like "I'll keep this vague" or "I actually want to push back here")
- If the response starts with <reply:N>, WHY I'm replying to that specific message rather than sending a general response — what about message N made me want to address it directly?

CRITICAL RULES:
- ALWAYS write in first person ("I", "me", "my"). You ARE "Me". Never refer to yourself as "they" or in third person.
- Refer to the other person naturally, the way you would in your own head. Use their first name, a nickname, or pronouns (she/he/they) — whatever feels most natural for the moment. You do NOT need to use their full name every time. The other person's name/label is "{other_speaker}".
- Only include dimensions that are genuinely present. A simple "sure, sounds good" does NOT need deep emotional analysis. If the response is mundane, the thought should be mundane too.
- Do NOT manufacture drama, opinions, or engagement signals that aren't supported by the actual conversation. If I'm just confirming logistics, the thought is just about logistics.
- Match the WEIGHT of the thought to the weight of the moment. A one-word reply gets a light thought. A long emotional response gets a richer one.
- Be specific — reference actual things from the conversation, not generic filler.
- 1-4 sentences. Scale length to how much is actually going on.
- Do NOT include the actual response text in the monologue.
- Output ONLY the monologue text. No quotes, no labels, no JSON.

QUALITY GATE — output [SKIP] + brief reason instead of a monologue if ANY of these apply:
1. GIBBERISH AT START: The response is the FIRST message in a conversation (no prior history) AND it is gibberish, random typing, a test message, or an accidental keyboard mash. (Gibberish in the MIDDLE of a conversation can be expressive — do NOT skip those.)
2. MISSING CONTEXT: The response clearly references something that happened OUTSIDE this conversation (another group chat, a phone call, an in-person interaction, a shared experience not mentioned anywhere — not in the conversation history AND not in the conversation context) and you CANNOT plausibly explain why "Me" sent this response. If you would have to invent or guess at the reasoning, that means the context is missing — output [SKIP].
3. INTERRUPTED THOUGHT FRAGMENT: The response is the beginning of a thought that was cut short because {other_speaker} interjected before "Me" finished. This typically looks like: "Me" sends a short fragment that reads like the start of a sentence (e.g., "Ok so", "But like", "And then I"), and the very next message from {other_speaker} arrives within just a few seconds — meaning both people were typing at the same time, and {other_speaker}'s message landed before "Me" could finish their thought. The fragment is NOT a self-contained message; it only makes sense as the beginning of something longer.

   How to detect this: Look at the "WHAT HAPPENED RIGHT AFTER" section (if provided). If {other_speaker}'s next message arrives almost immediately (within a few seconds of "Me"'s response), that's a strong signal of overlapping typing. Combined with the response reading like an incomplete sentence (ending with a conjunction, preposition, article, or trailing off mid-clause), this is an interrupted fragment — skip it.

   CRITICAL — do NOT skip these, even if they are short:
   - Genuine acknowledgments and reactions: "Yea", "Ok", "Lol", "Oh shit", "Damn", "Nah"
   - Expressive fragments that stand on their own: "Stop<msg>Literally just<msg>You're so" (expressing exasperation), "Who<msg>The<msg>Fuck" (dramatic emphasis)
   - Casual abbreviations: "Comin", "Doin", "Somethin" (these are complete messages with dropped g's)
   - Keyboard mashes and elongated words: "HEHEHEHE", "Mmmmmm", "alsdkjfa" (these are expressive)
   - One-word answers to direct questions: Q: "What grade?" A: "C"

   The litmus test: Would {other_speaker} need to say "go on..." or "finish your thought" after hearing JUST this response? If yes, it's an interrupted fragment. If {other_speaker} could reasonably respond to it as a complete message, keep it.

IMPORTANT: Do NOT over-filter. These are fine and should NOT be skipped:
- Short responses ("ok", "lol", emojis, casual slang)
- Responses where the reasoning is simple/obvious even if brief
- Responses that naturally follow from the conversation even if the topic is mundane
- Responses where you can reasonably infer the intent from the conversation flow OR from the conversation context
- Conversation openers where the context explains why "Me" is reaching out

Only skip when you genuinely cannot connect the response to EITHER the conversation history OR the conversation context without fabricating details."""


def _build_cot_user_prompt(
    conversation_history: List[Dict],
    target_response: str,
    other_speaker: str = "the other person",
    conversation_context: str = "",
    prior_cots: Optional[Dict[str, str]] = None,
    after_context: Optional[List[Dict]] = None,
) -> str:
    """Build the user prompt for CoT generation.

    Args:
        prior_cots: Mapping of timestamp → CoT text for previous "Me" messages
                    in this conversation.  CoTs are shown inline next to their
                    corresponding "Me" message in the history.
        after_context: Optional list of 1-3 messages that came AFTER "Me"'s
                       response.  Used by the quality gate to detect interrupted
                       fragments (overlapping typing where the other person's
                       message landed seconds after this response).
    """
    history_lines = []
    for idx, msg in enumerate(conversation_history):
        msg_num = idx + 1  # 1-based to match <reply:N> tokens
        ts = msg.get("timestamp", "")
        speaker = msg.get("speaker", "Unknown")
        content = msg.get("content", "")
        content = content.replace("<msg>", "\n    ")

        # Inline prior CoT with the corresponding "Me" message
        if speaker == "Me" and prior_cots and ts in prior_cots:
            cot_text = prior_cots[ts]
            history_lines.append(f"[{msg_num}] [{ts}] Me (💭 {cot_text}): {content}")
        else:
            history_lines.append(f"[{msg_num}] [{ts}] {speaker}: {content}")

    history_text = "\n".join(history_lines) if history_lines else "(no prior history)"
    target_display = target_response.replace("<msg>", "\n    ")

    context_block = ""
    if conversation_context:
        context_block = f"CONVERSATION CONTEXT (what was going on before this conversation started):\n{conversation_context}\n\n"

    # Build after-context block (for fragment detection)
    after_block = ""
    if after_context:
        after_lines = []
        for msg in after_context:
            ts = msg.get("timestamp", "")
            speaker = msg.get("speaker", "Unknown")
            content = msg.get("content", "")
            content = content.replace("<msg>", " / ")
            after_lines.append(f"[{ts}] {speaker}: {content}")
        after_block = (
            "\n\nWHAT HAPPENED RIGHT AFTER (for quality-gate context only — "
            "do NOT reference this in the monologue):\n"
            + "\n".join(after_lines)
        )

    return f"""{context_block}CONVERSATION BETWEEN "Me" AND "{other_speaker}":
{history_text}

WHAT "Me" ACTUALLY RESPONDED:
{target_display}{after_block}

Write the brief internal monologue "Me" had right before sending that response. Use first person ("I") for Me and refer to the other person as "{other_speaker}". If the response starts with <reply:N>, explain why I chose to reply to message [N] specifically."""


# ─────────────────────────────────────────────
# Relationship Analyzer Prompts
# ─────────────────────────────────────────────

RELATIONSHIP_SYSTEM_PROMPT = """You are analyzing a complete text message history between two people to produce a relationship summary that will be used as a system-prompt injection for a language model.

PERSONA IDENTITY:
- The person labeled "Me" in the transcripts is: {persona_full_name} (also known as: {persona_nicknames})
- Bio: {persona_bio}
- If "Me" sends their own name as the first message in a conversation, that is a self-introduction to a new contact, NOT the name of the other person.

Your job is to write a brief **relational context** summary. It should ONLY capture:

1. **Relationship type** — how they know each other at a high level (college friends, coworkers, family, mentor/mentee, etc.)
2. **Closeness & comfort level** — how intimate or guarded the relationship is; how much vulnerability is present
3. **Power/status dynamics** — are they equals? Is there deference, mentoring, or asymmetry?
4. **Emotional dynamics** — who initiates emotional support? How do they handle conflict or stress?

DO NOT INCLUDE — the model must learn these from the actual conversation data, not from a description:
- Communication style (message length, slang, emoji patterns, humor style)
- Texting habits or patterns (short bursts, rapid-fire, etc.)
- Typical response behaviors (banter, one-liners, teasing, etc.)
- Interaction categories (what they coordinate, plan, discuss)

The reason: this summary is injected as a system prompt during model training. If it prescribes HOW the persona communicates (e.g. "you text in short bursts with slang"), the model learns to follow that instruction rather than internalizing the style from data. We need the model to learn communication style from the training examples themselves.

CRITICAL RULES:
- Do NOT cite specific people, names, places, events, or conversation content from the messages. The summary must be ABSTRACT.
- Do NOT describe how the relationship evolved or changed over time. Write a timeless snapshot.
- Do NOT describe communication style, texting patterns, or how either person typically responds.

GUIDELINES:
- Write from "Me"'s perspective, using "you" to address the persona. Refer to the other person by their name/label ("{other_speaker}"), NEVER as "this person" or "the other person."
- Focus purely on the RELATIONSHIP: who are these two people to each other? How close? What's the emotional dynamic?
- Keep it to 2-4 sentences (50-80 words). Tight and relational.
- Output ONLY the relationship summary text. No JSON, no headers, no formatting."""


def _build_relationship_user_prompt(
    sampled_conversations: List[List[Dict]],
    other_speaker: str,
    total_conversations: int,
    date_range: tuple,
) -> str:
    """Build the user prompt with sampled conversations for relationship analysis."""
    parts = []
    parts.append(f"CHAT HISTORY OVERVIEW:")
    parts.append(f"- Between: \"Me\" and \"{other_speaker}\"")
    parts.append(f"- Total conversations: {total_conversations}")
    parts.append(f"- Date range: {date_range[0]} to {date_range[1]}")
    parts.append(f"- Sampled conversations shown below: {len(sampled_conversations)}")
    parts.append("")

    for i, conv in enumerate(sampled_conversations):
        parts.append(f"── Conversation {i+1} ({conv[0]['timestamp']}) ──")
        for msg in conv:
            content = msg["content"].replace("<msg>", " / ")
            # Truncate very long messages
            if len(content) > 200:
                content = content[:200] + "..."
            parts.append(f"[{msg['timestamp']}] {msg['speaker']}: {content}")
        parts.append("")

    parts.append("Based on these sampled conversations, write the relationship summary.")
    return "\n".join(parts)


def _sample_conversations_for_analysis(
    conversations: List[List[Dict]],
    max_conversations: int = 15,
    max_messages_per_conv: int = 20,
) -> List[List[Dict]]:
    """
    Strategically sample conversations to give a representative view.
    Picks from early, middle, and late in the timeline, preferring longer conversations.
    """
    if len(conversations) <= max_conversations:
        sampled = conversations
    else:
        # Split timeline into thirds and sample from each
        n = len(conversations)
        third = n // 3
        early = conversations[:third]
        middle = conversations[third:2*third]
        late = conversations[2*third:]

        per_section = max_conversations // 3

        def pick_best(section, k):
            # Prefer longer conversations (more signal)
            sorted_by_len = sorted(section, key=len, reverse=True)
            return sorted_by_len[:k]

        sampled = (
            pick_best(early, per_section) +
            pick_best(middle, per_section) +
            pick_best(late, max_conversations - 2 * per_section)
        )

    # Truncate long conversations
    truncated = []
    for conv in sampled:
        if len(conv) <= max_messages_per_conv:
            truncated.append(conv)
        else:
            # Keep first few and last few messages
            head = conv[:max_messages_per_conv // 2]
            tail = conv[-(max_messages_per_conv // 2):]
            truncated.append(head + tail)

    return truncated


# ─────────────────────────────────────────────
# Conversation-Level Context Generation Prompts
# ─────────────────────────────────────────────

CONVERSATION_GOAL_SYSTEM_PROMPT = """You are generating a brief **pre-conversation context** for a person named "{persona_name}".

PERSONA IDENTITY:
- Full name: {persona_full_name}
- Also known as: {persona_nicknames}
- Bio: {persona_bio}
- In the transcript below, this person is ALWAYS labeled "Me".
- If "Me" sends their own name (e.g. "{persona_full_name}!") as the first message, that is a SELF-INTRODUCTION to a new contact, NOT the name of the recipient.

You will be given:
1. The CURRENT conversation between "{persona_name}" and "{other_speaker}"
2. (Optionally) The tail end of 1-2 RECENT PRIOR CONVERSATIONS between the same people

IMPORTANT — SPEAKER IDENTIFICATION:
- "Me" = {persona_full_name} (the persona). ALWAYS.
- "{other_speaker}" = the other person.
- You are writing the context from "Me" ({persona_name})'s perspective ONLY.

Your job: write the SITUATIONAL CONTEXT that {persona_name} ("Me") would have known BEFORE this conversation started. This is purely the background state of the world — NOT what {persona_name} is about to do, and NOT a summary of the conversation.

Think of it as: "What was true about the world right before this exchange started?"

USING PRIOR CONVERSATIONS:
- Prior conversations are provided so you can identify CONTINUITY — ongoing plans, unresolved topics, or shared context that carries into the current conversation.
- ONLY mention information from prior conversations if it is DIRECTLY RELEVANT to the current conversation. If the current conversation is about a completely different topic, ignore the prior conversations entirely.
- Do NOT summarize the prior conversations. Do NOT mention topics from prior conversations that don't connect to the current one.
- Example: If the prior conversation ended with "let's get dinner tomorrow" and the current conversation opens with "hey are we still on?", the context should mention the dinner plans. But if the prior conversation was about homework and the current one is "MERRY CHRISTMAS", the prior conversation is irrelevant — ignore it.

CRITICAL RULES:
1. Write in second person: "You..." / "{other_speaker} just..."
2. ONLY describe the SITUATION — facts, events, or circumstances that exist independent of this conversation. Do NOT describe what {persona_name} is about to do, why they're reaching out, or their intent. The persona's decision-making belongs in internal reasoning, not here.
3. You may infer the situation from the first 1-2 messages of the current conversation (e.g., if the conversation opens with "congrats on the new job!", the context is that the other person recently got a new job).
4. Keep it to 1-3 sentences. Brief and factual. Only go to 3 sentences if prior conversation context adds genuinely relevant information.
5. If the trigger is unclear or the conversation is casual small talk, say something simple like "{other_speaker} texted you" or "You and {other_speaker} haven't talked in a while."
6. Output ONLY the context text.

EXAMPLES OF GOOD CONTEXT (purely situational):
- "{other_speaker} is moving out of her dorm this week."
- "You and {other_speaker} haven't talked in a while."
- "You and {other_speaker} made plans to meet up today."
- "A mutual friend just posted something you both saw."
- "{other_speaker} recently got a new job."
- "You and {other_speaker} were discussing a group trip to Hawaii in your last conversation and tentatively agreed to go." (ONLY if the current conversation continues this topic.)

EXAMPLES OF BAD CONTEXT (leaks intent or action):
- BAD: "You're following up to confirm plans." (This describes what the persona IS ABOUT TO DO — that's intent, not context.)
- BAD: "You want to reconnect and catch up." (This is a goal/desire, not a situational fact.)
- BAD: "You're reaching out to share some photos." (This describes the persona's action.)
- BAD: "You coordinate picking up your belongings, then ask about housing options." (This describes what HAPPENS in the conversation.)
- BAD: "Your goal is to share your travel timeline and keep the conversation warm." (This is a goal/arc summary.)
- BAD: "In your last conversation you talked about quiz bowl and school gossip." (Irrelevant prior context — only include if the current conversation continues these topics.)"""


def _build_conversation_goal_user_prompt(
    conversation: List[Dict],
    prior_conversations: Optional[List[List[Dict]]] = None,
) -> str:
    """Format a full conversation (and optional prior conversations) for the goal generation prompt.

    Args:
        conversation: The current conversation to generate context for.
        prior_conversations: Optional list of 1-2 recent prior conversations.
            Only the tail end (last ~10 messages) of each is included to keep
            the prompt compact.
    """
    parts = []

    # Include tail of prior conversations if provided
    if prior_conversations:
        for pc_idx, pc in enumerate(prior_conversations):
            # Take the last 10 messages of each prior conversation
            tail = pc[-10:] if len(pc) > 10 else pc
            pc_lines = []
            for msg in tail:
                content = msg["content"].replace("<msg>", " / ")
                if len(content) > 200:
                    content = content[:200] + "..."
                pc_lines.append(f"[{msg['timestamp']}] {msg['speaker']}: {content}")

            label = f"RECENT PRIOR CONVERSATION (ended {pc[-1]['timestamp'] if pc else '?'}):"
            if len(pc) > 10:
                label += f" (showing last 10 of {len(pc)} messages)"
            parts.append(label + "\n" + "\n".join(pc_lines))

        parts.append("")  # blank line separator

    # Current conversation
    lines = []
    for msg in conversation:
        content = msg["content"].replace("<msg>", " / ")
        if len(content) > 250:
            content = content[:250] + "..."
        lines.append(f"[{msg['timestamp']}] {msg['speaker']}: {content}")
    parts.append("CURRENT CONVERSATION:\n" + "\n".join(lines))

    return "\n".join(parts)


# ─────────────────────────────────────────────
# Annotator Class
# ─────────────────────────────────────────────

class LLMAnnotator:
    """
    LLM-based conversation annotator & CoT generator.

    Supports both OpenAI models (gpt-*) and Anthropic models (claude-*).
    The provider is auto-detected from the model name in the config, unless
    ``base_url`` is set: then every model goes to that OpenAI-compatible
    endpoint (e.g. OpenRouter), as in AIPartnerClient.

    Capabilities:
        - Relationship analysis (analyze_relationship)
        - Conversation-level context generation (generate_conversation_goal[s_batch])
        - Chain-of-thought internal monologue (generate_cot)
    """

    # Mapping from reasoning_effort → Anthropic extended-thinking budget
    _ANTHROPIC_THINKING_BUDGETS = {
        "low": 5_000,
        "medium": 10_000,
        "high": 20_000,
    }

    # Pricing per million tokens (USD).  Update as prices change.
    # Format: model_prefix → (input_$/M, output_$/M)
    _PRICING = {
        # Anthropic
        "claude-opus-4":     (5.0,  25.0),
        "claude-sonnet-4":   (3.0,  15.0),
        "claude-haiku":      (0.25, 1.25),
        # OpenAI
        "gpt-5.5":           (5.0,  30.0),
        "gpt-5.2":           (2.0,  8.0),
        "gpt-5.1":           (2.0,  8.0),
        "gpt-4.1":           (2.0,  8.0),
        "gpt-4o":            (2.50, 10.0),
        "o3":                (2.0,  8.0),
        "o4-mini":           (1.10, 4.40),
    }

    def __init__(self, config: Optional[AnnotatorConfig] = None):
        self.config = config or AnnotatorConfig()
        self._use_anthropic = _is_anthropic_model(self.config.model) and not self.config.base_url

        if self._use_anthropic:
            import anthropic
            self.client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY
        else:
            from IMPersona2.eval import openai_client_kwargs
            self.client = OpenAI(**openai_client_kwargs(self.config.base_url))

        # Thread-safe token tracking
        self._usage_lock = threading.Lock()
        self._input_tokens = 0
        self._output_tokens = 0
        self._num_calls = 0

    def _track_usage(self, input_tokens: int, output_tokens: int):
        """Thread-safe accumulation of token usage."""
        with self._usage_lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._num_calls += 1

    def _get_price_per_million(self):
        """Look up (input_$/M, output_$/M) for the current model."""
        model = self.config.model.lower()
        # Try exact match first, then prefix match
        for prefix, prices in self._PRICING.items():
            if model.startswith(prefix):
                return prices
        return (0.0, 0.0)  # unknown model

    def get_usage(self) -> dict:
        """Return cumulative token usage and estimated cost for this annotator."""
        with self._usage_lock:
            input_tok = self._input_tokens
            output_tok = self._output_tokens
            num_calls = self._num_calls

        input_price, output_price = self._get_price_per_million()
        input_cost = input_tok / 1_000_000 * input_price
        output_cost = output_tok / 1_000_000 * output_price
        total_cost = input_cost + output_cost

        return {
            "model": self.config.model,
            "num_calls": num_calls,
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "total_tokens": input_tok + output_tok,
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "total_cost_usd": round(total_cost, 4),
        }

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Unified LLM call that works with both OpenAI and Anthropic."""
        if self._use_anthropic:
            from IMPersona2.eval import create_anthropic_message

            budget = self._ANTHROPIC_THINKING_BUDGETS.get(
                self.config.reasoning_effort, 10_000
            )
            # max_tokens covers thinking and must exceed budget_tokens: "high"
            # gets the same 6_000 tokens for the answer as "medium"
            max_tokens = max(16_000, budget + 6_000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                response = create_anthropic_message(
                    self.client,
                    model=self.config.model,
                    max_tokens=max_tokens,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": budget,
                    },
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt},
                    ],
                )
            # Track token usage
            if hasattr(response, "usage") and response.usage:
                self._track_usage(
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )
            # Extract the text block (skip thinking blocks)
            for block in response.content:
                if block.type == "text":
                    return block.text.strip()
            return ""
        else:
            kwargs = {}
            # Only OpenAI's own API takes reasoning_effort (as in AIPartnerClient)
            if not self.config.base_url:
                kwargs["reasoning_effort"] = self.config.reasoning_effort
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs,
            )
            # Track token usage
            if hasattr(response, "usage") and response.usage:
                self._track_usage(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
            return response.choices[0].message.content.strip()

    # ─────────────────────────────────────────
    # Relationship Analysis
    # ─────────────────────────────────────────

    def analyze_relationship(
        self,
        conversations: List[List[Dict]],
        other_speaker: str = "Friend",
        max_sample_conversations: int = 15,
        persona_full_name: str = "",
        persona_nicknames: str = "",
        persona_bio: str = "",
    ) -> str:
        """
        Analyze the full chat history to produce a holistic relationship summary.

        Args:
            conversations: All parsed conversations (list of list of message dicts)
            other_speaker: Name/identifier of the other person
            max_sample_conversations: How many conversations to sample for analysis
            persona_full_name: Full name of the persona (for identity grounding)
            persona_nicknames: Comma-separated nicknames
            persona_bio: Brief bio for context

        Returns:
            A 1-2 paragraph relationship summary string
        """
        if not conversations:
            return "No conversation history available."

        # Get date range
        all_timestamps = []
        for conv in conversations:
            for msg in conv:
                all_timestamps.append(msg.get("timestamp", ""))
        all_timestamps = [t for t in all_timestamps if t]
        date_range = (
            min(all_timestamps) if all_timestamps else "unknown",
            max(all_timestamps) if all_timestamps else "unknown",
        )

        # Sample conversations
        sampled = _sample_conversations_for_analysis(
            conversations, max_conversations=max_sample_conversations
        )

        user_prompt = _build_relationship_user_prompt(
            sampled, other_speaker, len(conversations), date_range
        )

        # Format system prompt with persona identity and other speaker name
        sys_prompt = RELATIONSHIP_SYSTEM_PROMPT.format(
            persona_full_name=persona_full_name or "unknown",
            persona_nicknames=persona_nicknames or "unknown",
            persona_bio=persona_bio or "(not provided)",
            other_speaker=other_speaker,
        )

        for attempt in range(3):
            try:
                summary = self._call_llm(sys_prompt, user_prompt)
                # Clean up quotes if present
                if summary.startswith('"') and summary.endswith('"'):
                    summary = summary[1:-1]
                return summary

            except Exception as e:
                if attempt < 3 - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return f"[RELATIONSHIP ANALYSIS FAILED: {e}]"

    # ─────────────────────────────────────────
    # Conversation-Level Goal Generation
    # ─────────────────────────────────────────

    def generate_conversation_goal(
        self,
        conversation: List[Dict],
        persona_name: str = DEFAULT_PERSONA_NAME,
        other_speaker: str = "Friend",
        persona_full_name: str = "",
        persona_nicknames: str = "",
        persona_bio: str = "",
        prior_conversations: Optional[List[List[Dict]]] = None,
    ) -> str:
        """
        Generate a forward-looking conversational goal for a single conversation.

        Returns a 1-3 sentence directive describing what the persona is trying
        to accomplish, without leaking outcomes from the conversation.

        Args:
            prior_conversations: Optional list of 1-2 recent prior conversations
                between the same people.  Only the tail is included in the prompt.
        """
        system_prompt = CONVERSATION_GOAL_SYSTEM_PROMPT.format(
            persona_name=persona_name,
            other_speaker=other_speaker,
            persona_full_name=persona_full_name or persona_name,
            persona_nicknames=persona_nicknames or persona_name,
            persona_bio=persona_bio or "(not provided)",
        )
        user_prompt = _build_conversation_goal_user_prompt(
            conversation,
            prior_conversations=prior_conversations,
        )

        for attempt in range(3):
            try:
                goal = self._call_llm(system_prompt, user_prompt)
                # If empty due to token budget exhaustion, retry
                if not goal:
                    if attempt < 3 - 1:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    return "[GOAL GENERATION FAILED: empty response]"
                # Clean up quotes if present
                if goal.startswith('"') and goal.endswith('"'):
                    goal = goal[1:-1]
                return goal
            except Exception as e:
                if attempt < 3 - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return f"[GOAL GENERATION FAILED: {e}]"

    @staticmethod
    def goal_prompt_hash(
        conversation: List[Dict],
        prior_conversations: Optional[List[List[Dict]]] = None,
    ) -> str:
        """Compute a content-based hash for a conversation goal prompt.

        The hash is derived from the user prompt text (which includes both
        the current conversation and any prior-conversation context).  Two
        conversations with identical content produce the same hash, so the
        cache survives conversation re-indexing.
        """
        user_prompt = _build_conversation_goal_user_prompt(
            conversation, prior_conversations=prior_conversations,
        )
        return hashlib.sha256(user_prompt.encode()).hexdigest()[:16]

    def generate_conversation_goals_batch(
        self,
        conversations: List[List[Dict]],
        persona_name: str = DEFAULT_PERSONA_NAME,
        other_speaker: str = "Friend",
        progress: bool = True,
        max_concurrency: Optional[int] = None,
        persona_full_name: str = "",
        persona_nicknames: str = "",
        persona_bio: str = "",
        prior_conversation_count: int = 1,
        cache: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Generate conversation-level goals for all conversations concurrently.

        Args:
            prior_conversation_count: Number of prior conversations to include
                as context for each conversation (default 1). Set to 0 to
                disable prior conversation context.
            cache: Optional content-keyed cache dict ``{hash: goal_text}``.
                If provided, conversations whose prompt hash is already in
                the cache are skipped (no LLM call).  Newly generated goals
                are added to this dict in-place.

        Returns a list of goal strings, one per conversation.
        """
        total = len(conversations)
        workers = max_concurrency or self.config.max_concurrency
        results: List[Optional[str]] = [None] * total
        completed = 0
        cached_hits = 0
        errors = 0
        lock = threading.Lock()

        if cache is None:
            cache = {}

        # Pre-compute hashes and resolve cache hits
        prompt_hashes: List[str] = []
        to_generate: List[int] = []
        for i, conv in enumerate(conversations):
            prior_convs = None
            if prior_conversation_count > 0 and i > 0:
                start = max(0, i - prior_conversation_count)
                prior_convs = conversations[start:i]
            h = self.goal_prompt_hash(conv, prior_convs)
            prompt_hashes.append(h)
            if h in cache:
                results[i] = cache[h]
                cached_hits += 1
            else:
                to_generate.append(i)

        if progress:
            print(f"       Cache: {cached_hits} hits, {len(to_generate)} to generate")

        if not to_generate:
            if progress:
                print(f"  ✅ All {total} goals loaded from cache.")
            return results

        def _process(idx: int) -> None:
            nonlocal completed, errors
            conv = conversations[idx]
            # Build prior conversation context
            prior_convs = None
            if prior_conversation_count > 0 and idx > 0:
                start = max(0, idx - prior_conversation_count)
                prior_convs = conversations[start:idx]

            goal = self.generate_conversation_goal(
                conv, persona_name, other_speaker,
                persona_full_name=persona_full_name,
                persona_nicknames=persona_nicknames,
                persona_bio=persona_bio,
                prior_conversations=prior_convs,
            )
            results[idx] = goal

            # Cache the result (unless it's an error)
            if "[GOAL GENERATION FAILED" not in goal:
                with lock:
                    cache[prompt_hashes[idx]] = goal

            with lock:
                completed += 1
                if "[GOAL GENERATION FAILED" in goal:
                    errors += 1
                if progress:
                    pct = completed / len(to_generate) * 100
                    sys.stdout.write(
                        f"\r  Generating goals [{completed}/{len(to_generate)}] ({pct:.0f}%) "
                        f"— errors: {errors}"
                    )
                    sys.stdout.flush()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process, i): i
                for i in to_generate
            }
            for future in as_completed(futures):
                future.result()

        if progress:
            print(f"\n  ✅ Done. {len(to_generate) - errors} generated, {cached_hits} cached, {errors} errors.")
        return results

    # ─────────────────────────────────────────
    # Chain-of-Thought Generation
    # ─────────────────────────────────────────

    def generate_cot(
        self,
        conversation_history: List[Dict],
        target_response: str,
        other_speaker: str = "the other person",
        persona_full_name: str = "",
        persona_nicknames: str = "",
        conversation_context: str = "",
        prior_cots: Optional[Dict[str, str]] = None,
        after_context: Optional[List[Dict]] = None,
    ) -> str:
        """
        Generate a free-form chain-of-thought internal monologue for a single example.

        Given the conversation history and the persona's actual response, produces a natural
        language "thinking" trace that explains what the persona is noticing, feeling, and
        intending before they respond.

        If ``prior_cots`` is provided (a mapping of timestamp → CoT), these are
        inlined next to the corresponding "Me" messages in the conversation history
        so the LLM can see what was already reasoned about and avoid redundancy.

        If ``after_context`` is provided (list of 1-3 messages after the response),
        it's appended to the user prompt so the quality gate can detect interrupted
        fragments from overlapping typing.

        Returns the CoT string (without delimiters — caller wraps in <think>).
        """
        sys_prompt = COT_SYSTEM_PROMPT.format(
            other_speaker=other_speaker,
            persona_full_name=persona_full_name or "unknown",
            persona_nicknames=persona_nicknames or "unknown",
        )
        user_prompt = _build_cot_user_prompt(
            conversation_history, target_response,
            other_speaker=other_speaker,
            conversation_context=conversation_context,
            prior_cots=prior_cots,
            after_context=after_context,
        )

        for attempt in range(3):
            try:
                cot = self._call_llm(sys_prompt, user_prompt)
                # Clean up: remove quotes if the model wrapped it
                if cot.startswith('"') and cot.endswith('"'):
                    cot = cot[1:-1]
                # Remove any <think> tags the model might have added
                cot = cot.replace("<think>", "").replace("</think>", "").strip()
                return cot

            except Exception as e:
                if attempt < 3 - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return f"[COT GENERATION FAILED: {e}]"

    def _parse_history_from_input(self, input_text: str) -> List[Dict]:
        """
        Parse conversation history from the formatted input text back into
        a list of message dicts.

        Handles both numbered format ``[N] [timestamp] Speaker: content``
        and unnumbered format ``[timestamp] Speaker: content``.
        """
        import re
        messages = []
        # Match lines like:
        #   [1] [2024-08-21 15:52:59] Me: Hello       (numbered)
        #   [2024-08-21 15:52:59] Me: Jane Doe!         (unnumbered)
        pattern = r'(?:\[\d+\] )?\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] ([^:]+): (.+)'

        for line in input_text.split("\n"):
            line = line.strip()
            match = re.match(pattern, line)
            if match:
                messages.append({
                    "timestamp": match.group(1),
                    "speaker": match.group(2),
                    "content": match.group(3),
                })

        return messages

    @staticmethod
    def _extract_conversation_context(instruction: str) -> str:
        """Extract the conversation context string from a system prompt instruction.

        The context is embedded as ' Context: ...' after the base system prompt,
        and ends before ' You are talking to' or ' Relationship context:'.
        """
        import re
        # Match " Context: <text>" up to the next known section or end of string
        match = re.search(
            r" Context: (.+?)(?= You are talking to| Relationship context:|$)",
            instruction,
        )
        return match.group(1).strip() if match else ""


# ─────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────

def cmd_goals(args):
    """Generate conversation-level goals for all conversations."""
    if not os.path.exists(args.conversations):
        print(f"❌ Conversations file not found: {args.conversations}")
        sys.exit(1)

    print(f"📂 Loading conversations from {args.conversations} ...")
    with open(args.conversations, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    print(f"   {len(conversations)} conversations loaded")

    # Detect other speaker
    from collections import Counter
    speaker_counts = Counter()
    for conv in conversations:
        for msg in conv:
            if msg["speaker"] != "Me":
                speaker_counts[msg["speaker"]] += 1
    other_speaker = speaker_counts.most_common(1)[0][0] if speaker_counts else "Friend"
    print(f"   Other speaker: {other_speaker}")

    config = AnnotatorConfig(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_concurrency=args.max_concurrency or 50,
    )
    annotator = LLMAnnotator(config)

    # Load persona identity from config if available
    persona_full_name = getattr(args, "persona_full_name", "") or args.persona_name
    persona_nicknames = getattr(args, "persona_nicknames", "") or args.persona_name
    persona_bio = getattr(args, "persona_bio", "") or ""

    print(f"\n🎯 Generating conversation-level goals with {config.model} ...")
    print(f"   Persona: {persona_full_name} (nicknames: {persona_nicknames})")
    goals = annotator.generate_conversation_goals_batch(
        conversations, args.persona_name, other_speaker,
        max_concurrency=config.max_concurrency,
        persona_full_name=persona_full_name,
        persona_nicknames=persona_nicknames,
        persona_bio=persona_bio,
    )

    # Build result
    result = []
    for i, (conv, goal) in enumerate(zip(conversations, goals)):
        result.append({
            "conversation_idx": i,
            "num_turns": len(conv),
            "timestamp_start": conv[0]["timestamp"] if conv else None,
            "timestamp_end": conv[-1]["timestamp"] if conv else None,
            "initiator": conv[0]["speaker"] if conv else None,
            "goal": goal,
        })

    # Print sample
    print(f"\n{'═' * 60}")
    print("SAMPLE GOALS")
    print(f"{'═' * 60}")
    for r in result[:5]:
        print(f"\nConv {r['conversation_idx']} ({r['num_turns']} turns, {r['timestamp_start']}):")
        print(f"  {r['goal']}")
    print(f"{'═' * 60}")

    # Save
    output_path = args.output or os.path.join(
        os.path.dirname(args.conversations), "conversation_goals.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved {len(result)} goals to {output_path}")


def cmd_relationship(args):
    """Analyze relationship from full chat history."""
    if not os.path.exists(args.conversations):
        print(f"❌ Conversations file not found: {args.conversations}")
        sys.exit(1)

    print(f"📂 Loading conversations from {args.conversations} ...")
    with open(args.conversations, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    print(f"   {len(conversations)} conversations loaded")

    # Detect other speaker
    from collections import Counter
    speaker_counts = Counter()
    for conv in conversations:
        for msg in conv:
            if msg["speaker"] != "Me":
                speaker_counts[msg["speaker"]] += 1
    other_speaker = speaker_counts.most_common(1)[0][0] if speaker_counts else "Friend"
    print(f"   Other speaker: {other_speaker}")

    config = AnnotatorConfig(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    annotator = LLMAnnotator(config)

    print(f"\n🔍 Analyzing relationship with {config.model} ...")
    summary = annotator.analyze_relationship(conversations, other_speaker)

    print(f"\n{'═' * 60}")
    print(f"RELATIONSHIP SUMMARY")
    print(f"{'═' * 60}")
    print(summary)
    print(f"{'═' * 60}")

    # Save
    output_path = args.output or os.path.join(
        os.path.dirname(args.conversations), "relationship_summary.json"
    )
    result = {
        "other_speaker": other_speaker,
        "summary": summary,
        "model": config.model,
        "num_conversations_analyzed": len(conversations),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Impersona-Env — LLM-based conversation annotator & CoT generator"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # goals subcommand
    goals_parser = subparsers.add_parser("goals", help="Generate conversation-level context")
    goals_parser.add_argument("--conversations", type=str,
                              default="experiment_output/parsed_conversations.json",
                              help="Path to parsed conversations JSON")
    goals_parser.add_argument("--output", type=str, default=None,
                              help="Output path for conversation goals")
    goals_parser.add_argument("--persona-name", type=str, default=DEFAULT_PERSONA_NAME)
    goals_parser.add_argument("--persona-full-name", type=str, default=DEFAULT_PERSONA_NAME,
                              help="Full name of the persona (Me)")
    goals_parser.add_argument("--persona-nicknames", type=str, default=DEFAULT_PERSONA_NAME,
                              help="Comma-separated nicknames/aliases")
    goals_parser.add_argument("--persona-bio", type=str, default="",
                              help="Brief persona bio for context")
    goals_parser.add_argument("--model", type=str, default=model_for("annotation"))
    goals_parser.add_argument("--reasoning-effort", type=str, default="medium",
                              choices=["low", "medium", "high"])
    goals_parser.add_argument("--max-concurrency", type=int, default=None)

    # relationship subcommand
    rel_parser = subparsers.add_parser("relationship", help="Analyze relationship from full chat history")
    rel_parser.add_argument("--conversations", type=str,
                            default="experiment_output/parsed_conversations.json",
                            help="Path to parsed conversations JSON")
    rel_parser.add_argument("--output", type=str, default=None,
                            help="Output path for relationship summary")
    rel_parser.add_argument("--model", type=str, default=model_for("annotation"))
    rel_parser.add_argument("--reasoning-effort", type=str, default="medium",
                            choices=["low", "medium", "high"])

    args = parser.parse_args()

    if args.command == "goals":
        cmd_goals(args)
    elif args.command == "relationship":
        cmd_relationship(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
