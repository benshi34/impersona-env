"""
Configuration for the Impersona-Env dataset generation pipeline.

This file contains all tunable parameters for:
  - Parsing raw iMessage exports
  - Splitting into conversations
  - Generating training examples (with goal/intent annotation)
  - Filtering
  - Output formatting

Change anything here and re-run generate_dataset.py to see the effect.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PERSONA_CONFIG_FILE = Path(__file__).resolve().parent.parent / "persona_config.json"


def default_persona_name() -> str:
    """First name from persona_config.json (written by `run.py`), else "User"."""
    try:
        return json.loads(PERSONA_CONFIG_FILE.read_text()).get("name") or "User"
    except (OSError, ValueError):
        return "User"


def _default_annotation_model() -> str:
    from IMPersona2.model_config import model_for
    return model_for("annotation")


def _default_annotation_reasoning() -> str:
    from IMPersona2.model_config import model_for
    return model_for("annotation", "reasoning") or "medium"


# ─────────────────────────────────────────────
# 0. Persona Identity
# ─────────────────────────────────────────────
@dataclass
class PersonaConfig:
    """
    Global identity metadata for the persona ("Me" in the chat).

    This is the single source of truth for who "Me" is.  Every LLM prompt
    that needs to understand the persona's identity should draw from here.

    Edge-case notes (add any special handling rules below):
      - "Me" sometimes texts their own name as the first message to a new
        contact (e.g. "Jane Doe!") as a self-introduction.  This is NOT
        the name of the recipient.
    """

    # Short first name (used in system prompts, e.g. "You are role-playing as {name}")
    name: str = field(default_factory=default_persona_name)

    # Full name — helps LLMs distinguish the persona from other people
    full_name: str = field(default_factory=default_persona_name)

    # All known nicknames / aliases the persona goes by or might be called
    nicknames: List[str] = field(default_factory=list)

    # Brief context about the persona (keeps LLM grounded)
    bio: str = ""


# ─────────────────────────────────────────────
# 0b. LLM Settings
# ─────────────────────────────────────────────
@dataclass
class LLMConfig:
    """
    Model and inference settings for all LLM-powered pipeline steps
    (annotation, CoT generation, relationship analysis, goal generation).

    The model and reasoning effort come from configs/models.yaml
    (`annotation`); every LLM call in the pipeline picks them up.
    """

    # Which model to use for all LLM calls
    model: str = field(default_factory=_default_annotation_model)

    # Reasoning effort ("low", "medium", "high") — controls thinking budget
    reasoning_effort: str = field(default_factory=_default_annotation_reasoning)

    # Max parallel API calls
    max_concurrency: int = 100


# ─────────────────────────────────────────────
# 1. Parsing & Conversation Splitting
# ─────────────────────────────────────────────
@dataclass
class ParsingConfig:
    """Controls how raw iMessage text files are parsed into conversations."""

    # Maximum gap (hours) between messages before a new conversation starts
    max_gap_hours: float = 48.0

    # Combine consecutive messages from the same sender into one turn
    combine_consecutive: bool = True

    # Delimiter used when combining consecutive messages
    combine_delimiter: str = "<msg>"

    # Maximum gap (hours) between consecutive same-speaker messages before
    # they are treated as separate turns.  Prevents grouping a cluster of
    # messages sent at 10 AM with a follow-up "why aren't you responding?"
    # sent the next day into one training example.  Set to 0 to disable.
    max_combine_gap_hours: float = 4.0


# ─────────────────────────────────────────────
# 2. Dataset Example Generation
# ─────────────────────────────────────────────
@dataclass
class GenerationConfig:
    """Controls how parsed conversations become training examples."""

    # Persona name (the "Me" in the chat) — should match PersonaConfig.name
    persona_name: str = field(default_factory=default_persona_name)

    # Number of prior conversations to include as context (0 = none)
    conversation_buffer: int = 0

    # ── Goal / Intent Annotation ──
    # Whether to annotate each example with a conversational goal
    annotate_goals: bool = True

    # ── Relationship Context ──
    # Whether to inject relationship context into the system prompt
    inject_relationship_context: bool = True

    # ── System Prompt Templates ──
    # Base system prompt (always included)
    base_system_prompt: str = (
        "You are role-playing as a person named {persona_name}. "
        "Stay in character and respond as {persona_name} would. "
        "When sending multiple messages in quick succession, separate them with <msg>. "
        "Messages in the conversation are numbered [1], [2], etc. "
        "To reply to a specific earlier message, start your response with "
        "<reply:N> where N is that message's number. "
        "Only use this when directly replying to a specific message, not for general responses."
    )

    # Conversation context addendum (appended when annotate_goals=True)
    goal_prompt_template: str = (
        " Context: {goal}"
    )

    # Relationship context addendum (appended when inject_relationship_context=True)
    relationship_prompt_template: str = (
        " You are talking to {other_name}. "
        "Relationship context: {relationship_description}"
    )


# ─────────────────────────────────────────────
# 3. Filtering
# ─────────────────────────────────────────────
@dataclass
class FilterConfig:
    """Controls which training examples are kept vs. discarded."""

    # Minimum number of words in the output (Me's response).
    # Set to 0 to preserve short/disengaged responses ("ok", "👍") —
    # these are important engagement signals for the reward mechanism.
    min_output_words: int = 0

    # Minimum number of turns in the conversation history (input).
    # Set to 0 to include conversation openers / first responses —
    # the conversation context (goal) gives the model enough signal.
    min_input_turns: int = 0

    # Minimum number of TOTAL messages in a conversation for it to
    # produce any training examples.  Conversations with fewer messages
    # are skipped entirely.  Set to 2 to require at least a real exchange.
    min_conversation_messages: int = 2

    # Maximum number of turns in the conversation history (input)
    max_input_turns: int = 80

    # Filter out examples where the output is only emoji.
    # Set to False to preserve emoji-only responses as engagement signals.
    filter_emoji_only: bool = False

    # Filter out examples containing URLs in the output
    filter_urls_in_output: bool = True

    # Filter out examples whose output is just a media attachment filename
    # (e.g. "IMG_9790.HEIC", "video.mov").  The model can't produce files.
    filter_attachment_filenames: bool = True

    # Strings in the OUTPUT that cause an example to be filtered out.
    # These are things the model should never learn to produce.
    blocked_strings: List[str] = field(default_factory=lambda: [
        "GamePigeon message:",
        "Library/Messages/",
        "Find My:",               # location sharing — no tool support at inference
        "Started sharing location",   # Find My app system message
        "Stopped sharing location",   # Find My app system message
        "is inviting you to a scheduled Zoom meeting",  # copy-paste Zoom invite
    ])

    # Filter out examples where the output contains consecutive identical
    # <msg> parts (e.g. the same text sent twice due to iMessage sync issues).
    filter_duplicate_messages: bool = True

    # Filter group chats (keep only 1-on-1)
    filter_group_chats: bool = True

    # Minimum date for conversations.  Any conversation containing a message
    # dated before this cutoff is excluded entirely.  Format: "YYYY-MM-DD".
    # Set to "" to disable.
    min_date: str = "2020-01-01"


# ─────────────────────────────────────────────
# 4. Pipeline Config (top-level)
# ─────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Top-level config combining all sub-configs."""

    persona: PersonaConfig = field(default_factory=PersonaConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)

    # ── Contacts ──
    # Path to a vCard (.vcf) file mapping phone numbers → names.
    # Used to replace phone numbers with human-readable names everywhere
    # (speaker labels, output folder names, etc.).  Set to "" to disable.
    contacts_file: str = "experiment_output/All Contacts.vcf"

    # ── Input / Output ──
    # List of raw iMessage export files to process.
    # Each file gets its own subfolder under output_dir.
    # Normally filled in by pipeline.py from selected_conversations.json.
    input_files: List[str] = field(default_factory=list)

    # Root output directory.  Per-conversation outputs go in
    # output_dir/<conversation_id>/<model_version>/dataset.json  (and llm_cache.json).
    output_dir: str = "experiment_output"

    # Dataset version suffix.  Appended to the model name when creating
    # output folders, e.g. "<annotation-model>_v4".  This makes it easy to
    # keep old and new versions side by side for comparison.
    dataset_version: str = "v4"  # keep in sync with pipeline.DEFAULT_DATASET_VERSION

    # Skip conversations whose dataset.json already exists.
    # Set to True or pass --force to reprocess everything.
    force_reprocess: bool = False

    # Whether to use LLM-based relationship analysis (uses the annotation model)
    use_llm_relationship: bool = True

    # Whether to use LLM-generated conversation goals (uses the annotation model)
    use_llm_goals: bool = True

    # Whether to generate chain-of-thought internal monologues (uses the annotation model)
    use_llm_cot: bool = True

    # Max number of contacts/conversations to process in parallel.
    # Each contact's pipeline (parse → goals → CoT → save) runs independently.
    # Set to 1 for sequential processing.
    max_parallel_contacts: int = 3
