#!/usr/bin/env python3
"""
Impersona-Env — Dataset Generation Pipeline
===========================================

Processes one or more raw iMessage export files and produces per-conversation
training datasets with optional goal/intent annotation and relationship
context injection.

Usage (from the repo root):
    python -m IMPersona2.generate_dataset                            # process all input_files from config
    python -m IMPersona2.generate_dataset --input_file path.txt      # process a single file
    python -m IMPersona2.generate_dataset --force                    # reprocess even if output exists
    python -m IMPersona2.generate_dataset --no-goals --no-cot        # disable LLM features

Each input file gets its own subfolder under output_dir/.  Already-processed
conversations are skipped unless --force is passed.

Inspect results with:
    python -m IMPersona2.dataset_visualizer --list                        # list available datasets
    python -m IMPersona2.dataset_visualizer --dataset "experiment_output/Jane Doe/dataset.json"
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from IMPersona2.experiment_config import (
    FilterConfig,
    GenerationConfig,
    LLMConfig,
    ParsingConfig,
    PersonaConfig,
    PipelineConfig,
)


# ═══════════════════════════════════════════════
# Contact Resolution (phone → name)
# ═══════════════════════════════════════════════

def _build_contacts_lookup(vcf_path: str) -> Dict[str, str]:
    """
    Parse a vCard (.vcf) file and build a phone→name lookup dict.

    Keys are normalized phone numbers (digits only, no country code).
    Uses parse_vcard_file from IMPersona2/utils.py.
    """
    if not vcf_path or not os.path.exists(vcf_path):
        return {}

    from IMPersona2.utils import parse_vcard_file

    raw = parse_vcard_file(vcf_path)

    # Build a lookup that handles +1 prefix and bare 10-digit forms
    lookup: Dict[str, str] = {}
    for phone, name in raw.items():
        digits = re.sub(r"[^\d]", "", phone)
        # Store with and without leading "1" country code
        lookup[digits] = name
        if len(digits) == 10:
            lookup["1" + digits] = name
        elif digits.startswith("1") and len(digits) == 11:
            lookup[digits[1:]] = name
        lookup[digits] = name

    return lookup


def resolve_contact_name(phone_or_label: str, contacts: Dict[str, str]) -> str:
    """
    Resolve a phone-number-style label or literal name to a human name.

    Returns the contact name if found, otherwise the original string unchanged.
    """
    if not contacts:
        return phone_or_label
    # Check literal match first (handles renamed contacts, emails, nicknames)
    if phone_or_label in contacts:
        return contacts[phone_or_label]
    # Then try phone-digit matching
    digits = re.sub(r"[^\d]", "", phone_or_label)
    if not digits:
        return phone_or_label
    for key in [digits, digits.lstrip("1"), "1" + digits.lstrip("1")]:
        if key in contacts:
            return contacts[key]
    return phone_or_label

# ═══════════════════════════════════════════════
# STEP 1: Parsing
# ═══════════════════════════════════════════════

def parse_imessage_file(
    file_path: str,
    config: ParsingConfig,
    contacts: Optional[Dict[str, str]] = None,
) -> List[List[Dict]]:
    """
    Parse a raw iMessage export into a list of conversations.

    Each conversation is a list of message dicts:
        { "speaker": str, "content": str, "timestamp": str,
          "is_reply": bool, "reply_to": int|None }

    If ``contacts`` is provided (phone→name dict), speaker labels that
    look like phone numbers are replaced with human-readable names.

    Conversations are split when the gap between messages exceeds
    config.max_gap_hours.  Messages are sorted by timestamp within
    each conversation (the raw export can reorder around replies).

    Reply threading is preserved: when a message is a reply to a
    specific earlier message, ``reply_to`` contains the 0-based index
    of the target in the (sorted, combined) conversation.
    """
    timestamp_pattern = r"([A-Z][a-z]{2} \d{1,2}, \d{4}\s+\d{1,2}:\d{2}:\d{2} [AP]M)"

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # ─── Phase 1 ──────────────────────────────────────────────
    # Scan for INDENTED blocks to build reply → parent mapping.
    #
    # In iMessage exports, when someone replies to a message:
    #   1. The parent message appears at the top level (unindented)
    #   2. The reply appears INDENTED (4 spaces) beneath the parent
    #   3. The reply ALSO appears later at the top level with
    #      "This message responded to an earlier message."
    #
    # We use (2) to figure out which parent each reply targets,
    # then only keep the top-level copies from (3).
    # ──────────────────────────────────────────────────────────

    reply_parent_map: Dict[Tuple[str, str], Dict] = {}
    # key = (reply_timestamp_str, reply_content[:60])
    # val = {"timestamp": ..., "speaker": ..., "content_prefix": ...}

    last_toplevel_msg: Optional[Dict] = None   # most recent unindented msg

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        is_indented = raw_line.startswith("    ")
        line = raw_line.strip()

        if not line:
            i += 1
            continue

        ts_match = re.match(timestamp_pattern, line)
        if not ts_match:
            i += 1
            continue

        if is_indented:
            # ── Indented timestamp → reply shown inline under parent ──
            reply_ts = datetime.strptime(ts_match.group(1), "%b %d, %Y %I:%M:%S %p")

            # Read sender
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break
            reply_sender = lines[i].strip()

            # Read content
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break
            reply_content = lines[i].strip()

            # Skip if it's a meta line (Tapbacks, system text)
            if (reply_content.startswith("Tapbacks:")
                    or reply_content.startswith("This message")):
                i += 1
                while i < len(lines) and lines[i].startswith("    "):
                    i += 1
                continue

            # Map this reply → the current parent
            if last_toplevel_msg:
                key = (
                    reply_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    reply_content[:60],
                )
                reply_parent_map[key] = {
                    "timestamp": last_toplevel_msg["timestamp"],
                    "speaker": last_toplevel_msg["speaker"],
                    "content_prefix": last_toplevel_msg["content"][:60],
                }

            # Skip remaining indented lines
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                i += 1
            continue

        else:
            # ── Top-level timestamp → track as potential reply parent ──
            msg_ts = datetime.strptime(ts_match.group(1), "%b %d, %Y %I:%M:%S %p")

            # Peek at sender + content (using temp index j)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                i += 1
                continue
            sender = lines[j].strip()

            j += 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                i += 1
                continue
            content = lines[j].strip()

            if not (content.startswith("This message")
                    or content.startswith("Tapbacks:")
                    or content.startswith("/Users/")
                    or content.startswith("Edited to")):
                resolved_speaker = "Me" if sender == "Me" else resolve_contact_name(sender, contacts or {})
                last_toplevel_msg = {
                    "timestamp": msg_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "speaker": resolved_speaker,
                    "content": content,
                }

            i += 1
            continue

    # ─── Phase 2 ──────────────────────────────────────────────
    # Parse only TOP-LEVEL (unindented) messages.
    # Indented copies are duplicates and are skipped entirely.
    # ──────────────────────────────────────────────────────────

    all_messages: List[Dict] = []

    i = 0
    while i < len(lines):
        raw_line = lines[i]

        # Skip all indented lines
        if raw_line.startswith("    ") or raw_line.startswith("\t"):
            i += 1
            continue

        line = raw_line.strip()
        ts_match = re.match(timestamp_pattern, line)

        if ts_match:
            timestamp_str = ts_match.group(1)
            timestamp = datetime.strptime(timestamp_str, "%b %d, %Y %I:%M:%S %p")

            i += 1
            if i >= len(lines):
                break
            sender = lines[i].strip()

            i += 1
            if i >= len(lines):
                break
            content = lines[i].strip()

            # ── Reply detection ──
            # Look ahead past Tapback lines / blanks for the reply marker.
            is_reply = False
            for look_ahead in range(1, 8):
                if i + look_ahead >= len(lines):
                    break
                ahead = lines[i + look_ahead].strip()
                if ahead == "This message responded to an earlier message.":
                    is_reply = True
                    i = i + look_ahead   # advance past the marker
                    break
                if re.match(timestamp_pattern, ahead):
                    break   # hit next message → stop looking
                # Continue past tapbacks and blanks
                if (ahead.startswith("Tapbacks:")
                        or ahead.startswith("Loved by")
                        or ahead.startswith("Liked by")
                        or ahead.startswith("Emphasized by")
                        or ahead.startswith("Laughed at by")
                        or ahead.startswith("Disliked by")
                        or ahead.startswith("Questioned by")
                        or ahead.startswith("Hearted by")
                        or ahead == ""):
                    continue
                break   # unknown line type → stop

            # Skip system messages / attachments
            if (content.startswith("This message")
                    or content.startswith("Tapbacks:")
                    or content.startswith("/Users/")
                    or content.startswith("Edited to")):
                i += 1
                continue

            # Look up parent if this is a reply
            reply_parent_info = None
            if is_reply:
                key = (
                    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    content[:60],
                )
                reply_parent_info = reply_parent_map.get(key)

            resolved_speaker = "Me" if sender == "Me" else resolve_contact_name(sender, contacts or {})
            msg = {
                "speaker": resolved_speaker,
                "content": content,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "is_reply": is_reply,
                "_reply_parent": reply_parent_info,   # resolved in Phase 5
            }
            all_messages.append(msg)

        i += 1

    # ─── Phase 3 ──────────────────────────────────────────────
    # Sort by timestamp, deduplicate, then split into conversations.
    # ──────────────────────────────────────────────────────────

    all_messages.sort(key=lambda m: m["timestamp"])

    # Deduplicate messages that appear twice in the export due to
    # iMessage sync artifacts.  The export often creates two entries
    # for the same physical message: one plain and one tagged as
    # "This message responded to an earlier message."  Both have
    # the same speaker, timestamp, and content.  We keep the reply-
    # flagged copy (if any) since it carries reply_parent metadata.
    deduped: List[Dict] = []
    for msg in all_messages:
        if deduped:
            prev = deduped[-1]
            if (prev["speaker"] == msg["speaker"]
                    and prev["timestamp"] == msg["timestamp"]
                    and prev["content"] == msg["content"]):
                # Duplicate detected — keep whichever has reply info
                if msg.get("is_reply") and not prev.get("is_reply"):
                    deduped[-1] = msg
                # else: keep prev (already in deduped)
                continue
        deduped.append(msg)

    if len(deduped) < len(all_messages):
        n_removed = len(all_messages) - len(deduped)
        print(f"  🧹 Removed {n_removed} duplicate messages (iMessage sync artifacts)")

    all_messages = deduped

    conversations: List[List[Dict]] = []
    current_conversation: List[Dict] = []

    for msg in all_messages:
        ts = datetime.strptime(msg["timestamp"], "%Y-%m-%d %H:%M:%S")
        if not current_conversation:
            current_conversation.append(msg)
        else:
            last_ts = datetime.strptime(
                current_conversation[-1]["timestamp"], "%Y-%m-%d %H:%M:%S"
            )
            if ts - last_ts > timedelta(hours=config.max_gap_hours):
                conversations.append(current_conversation)
                current_conversation = [msg]
            else:
                current_conversation.append(msg)

    if current_conversation:
        conversations.append(current_conversation)

    # ─── Phase 4 ──────────────────────────────────────────────
    # Combine consecutive messages (reply-aware).
    # ──────────────────────────────────────────────────────────

    if config.combine_consecutive:
        conversations = [
            _combine_consecutive(
                conv, config.combine_delimiter,
                max_gap_hours=config.max_combine_gap_hours,
            )
            for conv in conversations
        ]

    # ─── Phase 5 ──────────────────────────────────────────────
    # Resolve reply_to indices by matching content in the
    # (sorted, combined) conversation.
    # ──────────────────────────────────────────────────────────

    for conv in conversations:
        for idx, msg in enumerate(conv):
            parent_info = msg.pop("_reply_parent", None)
            msg["reply_to"] = None

            if parent_info and msg.get("is_reply"):
                prefix = parent_info["content_prefix"]
                match_len = min(40, len(prefix))
                if match_len > 5:
                    # Use `in` because the parent message may have been
                    # combined with earlier consecutive messages and the
                    # original content now sits in the middle of the turn.
                    needle = prefix[:match_len]
                    for j in range(idx - 1, -1, -1):
                        candidate = conv[j]
                        if (candidate["speaker"] == parent_info["speaker"]
                                and needle in candidate["content"]):
                            msg["reply_to"] = j
                            break

    return conversations


def _combine_consecutive(
    messages: List[Dict],
    delimiter: str,
    max_gap_hours: float = 0,
) -> List[Dict]:
    """Merge consecutive messages from the same speaker into one turn.

    Reply tags are preserved per-sub-message: when a reply message is
    merged into a group, its content is prefixed with a
    ``<reply_raw:IDX>`` marker (0-based conversation index) so that
    downstream formatting can resolve it to the correct ``<reply:N>``
    display tag after any history truncation.

    If *max_gap_hours* > 0, consecutive same-speaker messages separated
    by more than that many hours are treated as separate turns (prevents
    grouping a cluster of messages with a much-later follow-up).
    """
    if not messages:
        return messages

    combined = [messages[0].copy()]
    for msg in messages[1:]:
        prev = combined[-1]
        same_speaker = msg["speaker"] == prev["speaker"]

        # Time-gap check: don't combine if the gap exceeds the threshold.
        gap_too_large = False
        if same_speaker and max_gap_hours > 0:
            prev_ts = datetime.strptime(prev["timestamp"], "%Y-%m-%d %H:%M:%S")
            curr_ts = datetime.strptime(msg["timestamp"], "%Y-%m-%d %H:%M:%S")
            if (curr_ts - prev_ts) > timedelta(hours=max_gap_hours):
                gap_too_large = True

        if same_speaker and not gap_too_large:
            # Skip empty content (e.g. attachment-only messages in the
            # text export that have no textual representation).
            if msg["content"].strip():
                content_to_add = msg["content"]
                # Embed per-sub-message reply marker so it survives merging
                if msg.get("is_reply") and msg.get("reply_to") is not None:
                    content_to_add = f"<reply_raw:{msg['reply_to']}>" + content_to_add
                prev["content"] += delimiter + content_to_add
            prev["timestamp"] = msg["timestamp"]
        else:
            combined.append(msg.copy())

    return combined


# ═══════════════════════════════════════════════
# STEP 2: Goal / Intent Annotation
# ═══════════════════════════════════════════════

_RULE_BASED_TAXONOMY = [
    "emotional_support",
    "social_coordination",
    "information_exchange",
    "banter_bonding",
    "career_academic_advice",
    "persuasion_convincing",
    "conflict_resolution",
    "check_in",
    "collaboration",
    "recommendation",
]

def annotate_goal_rule_based(
    conversation_history: List[Dict],
    target_response: str,
) -> str:
    """
    Rule-based goal annotator (fallback when LLM goals are unavailable).

    Uses keyword and structural heuristics to classify the conversational goal.
    """
    taxonomy = _RULE_BASED_TAXONOMY
    # Flatten all text for keyword matching
    all_text = " ".join(
        msg["content"].lower() for msg in conversation_history
    ) + " " + target_response.lower()

    # ── Keyword-based scoring ──
    scores: Dict[str, float] = {g: 0.0 for g in taxonomy}

    # Emotional support signals
    emo_keywords = [
        "sorry", "sad", "stressed", "worried", "anxious", "tough", "hard",
        "struggling", "miss", "hurt", "crying", "depressed", "overwhelmed",
        "feel", "feeling", "vent", "dw", "don't worry", "it'll be ok",
        "you'll be fine", "hang in there", "proud of you", "love you",
        "❤️", "🫂", "💛", "😢", "😭",
    ]
    for kw in emo_keywords:
        if kw in all_text:
            scores["emotional_support"] += 1.0

    # Social coordination signals
    coord_keywords = [
        "meet", "hang", "dinner", "lunch", "coffee", "plan", "when",
        "where", "tomorrow", "tonight", "weekend", "saturday", "sunday",
        "friday", "time", "come", "pick up", "drive", "uber", "address",
        "location", "place", "let's go", "down to", "wanna", "schedule",
    ]
    for kw in coord_keywords:
        if kw in all_text:
            scores["social_coordination"] += 1.0

    # Information exchange signals
    info_keywords = [
        "how does", "what is", "explain", "tell me", "do you know",
        "link", "article", "paper", "read", "found", "interesting",
        "research", "study", "data", "results", "figure",
    ]
    for kw in info_keywords:
        if kw in all_text:
            scores["information_exchange"] += 1.0

    # Banter / bonding signals
    banter_keywords = [
        "lol", "lmao", "haha", "😂", "🤣", "💀", "bruh", "bro",
        "dude", "yo", "ngl", "icl", "fr", "lowkey", "highkey",
        "game", "movie", "show", "music", "song", "meme", "funny",
    ]
    for kw in banter_keywords:
        if kw in all_text:
            scores["banter_bonding"] += 1.0

    # Career / academic advice signals
    career_keywords = [
        "phd", "grad school", "industry", "job", "career", "research",
        "professor", "advisor", "application", "interview", "offer",
        "internship", "resume", "publication", "conference", "paper",
        "deadline", "submit", "arxiv", "review", "decision",
    ]
    for kw in career_keywords:
        if kw in all_text:
            scores["career_academic_advice"] += 1.0

    # Persuasion signals
    persuade_keywords = [
        "you should", "come on", "please", "just do it", "trust me",
        "i promise", "it'll be fun", "why not", "think about it",
    ]
    for kw in persuade_keywords:
        if kw in all_text:
            scores["persuasion_convincing"] += 1.0

    # Check-in signals
    checkin_keywords = [
        "how are you", "what's up", "how's it going", "how u been",
        "wyd", "what are you up to", "checking in", "hey", "hi",
        "sup", "how's everything",
    ]
    for kw in checkin_keywords:
        if kw in all_text:
            scores["check_in"] += 1.0

    # Collaboration signals
    collab_keywords = [
        "draft", "edit", "comments", "feedback", "review", "merge",
        "repo", "code", "bug", "fix", "implement", "section",
        "appendix", "figure", "table", "results", "benchmark",
        "let me know", "send", "version", "overleaf",
    ]
    for kw in collab_keywords:
        if kw in all_text:
            scores["collaboration"] += 1.0

    # Recommendation signals
    rec_keywords = [
        "recommend", "suggestion", "try", "check out", "have you seen",
        "have you tried", "you'd like", "restaurant", "place", "app",
    ]
    for kw in rec_keywords:
        if kw in all_text:
            scores["recommendation"] += 1.0

    # Conflict resolution signals
    conflict_keywords = [
        "sorry", "apologize", "my bad", "didn't mean", "misunderstood",
        "upset", "frustrated", "annoyed", "argue", "disagree",
    ]
    for kw in conflict_keywords:
        if kw in all_text:
            scores["conflict_resolution"] += 0.5  # lower weight — overlaps with emotional

    # Pick the highest-scoring goal, defaulting to banter_bonding
    best_goal = max(scores, key=lambda g: scores[g])
    if scores[best_goal] == 0:
        best_goal = "banter_bonding"

    return best_goal


# ═══════════════════════════════════════════════
# STEP 3: Training Example Generation
# ═══════════════════════════════════════════════

def _detect_other_speaker(conversations: List[List[Dict]]) -> str:
    """Find the most common non-'Me' speaker across all conversations."""
    speakers: Counter = Counter()
    for conv in conversations:
        for msg in conv:
            if msg["speaker"] != "Me":
                speakers[msg["speaker"]] += 1
    if speakers:
        return speakers.most_common(1)[0][0]
    return "Friend"


def _infer_relationship_description(
    conversations: List[List[Dict]],
    other_name: str,
) -> str:
    """
    Quick heuristic to describe the relationship based on conversation content.
    Replace with LLM-based inference for higher quality.
    """
    all_text = " ".join(
        msg["content"].lower()
        for conv in conversations
        for msg in conv
    )

    signals = []
    if any(kw in all_text for kw in ["phd", "research", "paper", "arxiv", "conference", "advisor"]):
        signals.append("academic colleague / research peer")
    if any(kw in all_text for kw in ["hang", "chill", "grab food", "party", "fun"]):
        signals.append("friend")
    if any(kw in all_text for kw in ["love you", "miss you", "babe", "❤️"]):
        signals.append("close friend or romantic partner")
    if any(kw in all_text for kw in ["class", "professor", "homework", "exam", "grade"]):
        signals.append("classmate")
    if any(kw in all_text for kw in ["campus", "dorm"]):
        signals.append("college peer")

    if not signals:
        signals.append("acquaintance")

    return "; ".join(signals)


def generate_training_examples(
    conversations: List[List[Dict]],
    gen_config: GenerationConfig,
    filter_config: FilterConfig,
    relationship_summary: Optional[str] = None,
    conversation_goals: Optional[List[Optional[str]]] = None,
    max_combine_gap_hours: float = 0,
) -> Tuple[List[Dict], Dict]:
    """
    Generate training examples from parsed conversations.

    Each example has:
        - instruction: system prompt (with optional goal + relationship context)
        - input:       conversation history formatted as text
        - output:      the target "Me" response
        - metadata:    goal label, speaker info, timestamps, etc.

    Args:
        relationship_summary: If provided, use this rich LLM-generated summary
                              instead of the rule-based heuristic.
        conversation_goals: If provided, a list of LLM-generated goals, one per
                            conversation. Used instead of the rule-based annotator.

    Returns:
        (examples_list, filter_stats_dict)
    """
    other_name = _detect_other_speaker(conversations)
    relationship_desc = relationship_summary or _infer_relationship_description(conversations, other_name)

    all_examples: List[Dict] = []
    filter_stats: Dict[str, int] = Counter()

    # Parse min_date cutoff once
    _min_date = None
    if filter_config.min_date:
        _min_date = filter_config.min_date  # "YYYY-MM-DD" string comparison works

    for conv_idx, conversation in enumerate(conversations):
        # ── Date filter: skip conversations with any message before cutoff ──
        if _min_date and conversation:
            earliest_ts = conversation[0].get("timestamp", "")[:10]
            if earliest_ts < _min_date:
                filter_stats["before_min_date"] += 1
                continue

        # Count only real text messages for the conversation-length filter.
        # Non-text messages (attachments, blocked strings, bare URLs) stay in
        # the conversation history so the flow is preserved, but they don't
        # inflate the effective message count.
        text_msg_count = sum(
            1 for m in conversation
            if not _is_non_text_message(m["content"], filter_config)
        )
        if text_msg_count < filter_config.min_conversation_messages:
            filter_stats["conversation_too_short"] += 1
            continue

        # Build prior conversation context (if conversation_buffer > 0)
        prior_convs = []
        if gen_config.conversation_buffer > 0:
            start = max(0, conv_idx - gen_config.conversation_buffer)
            prior_convs = conversations[start:conv_idx]

        # ── Group consecutive "Me" messages into single examples ──
        # Instead of one example per "Me" message, we collect all
        # consecutive "Me" messages (a "turn") and combine them with
        # <msg>.  The history ends at the last non-"Me" message.
        msg_idx = 0
        while msg_idx < len(conversation):
            message = conversation[msg_idx]

            if message["speaker"] != "Me" or not message["content"].strip():
                msg_idx += 1
                continue

            # Found the start of a "Me" turn — collect all consecutive
            first_me_idx = msg_idx
            me_parts: List[str] = []

            has_attachment = False
            while msg_idx < len(conversation) and conversation[msg_idx]["speaker"] == "Me":
                m = conversation[msg_idx]

                # Gap check: if this "Me" message is far from the previous
                # one, treat it as a separate turn (don't group).
                if msg_idx > first_me_idx and max_combine_gap_hours > 0:
                    prev_ts = datetime.strptime(
                        conversation[msg_idx - 1]["timestamp"], "%Y-%m-%d %H:%M:%S"
                    )
                    curr_ts = datetime.strptime(m["timestamp"], "%Y-%m-%d %H:%M:%S")
                    if (curr_ts - prev_ts) > timedelta(hours=max_combine_gap_hours):
                        break

                if m["content"].strip():
                    part = m["content"]
                    # Track if any sub-message is an attachment filename.
                    # _combine_consecutive may have already joined messages
                    # with <msg>, so check each sub-part individually.
                    if filter_config.filter_attachment_filenames:
                        for sub in part.split("<msg>"):
                            if sub.strip() and _is_attachment_filename(sub.strip()):
                                has_attachment = True
                                break
                    # Prepend reply token if this sub-message is a reply
                    if m.get("is_reply") and m.get("reply_to") is not None:
                        reply_display_idx = m["reply_to"] + 1
                        part = f"<reply:{reply_display_idx}>{part}"
                    # Resolve <reply_raw:N> markers embedded by
                    # _combine_consecutive (0-based → 1-based).
                    part = _REPLY_RAW_RE.sub(
                        lambda _m: f"<reply:{int(_m.group(1)) + 1}>", part
                    )
                    me_parts.append(part)
                msg_idx += 1

            # If any sub-message is an attachment, skip the entire turn
            # (the remaining text may reference the attachment content).
            # These messages still appear in conversation history for
            # other examples.
            if has_attachment:
                filter_stats["attachment_filename"] += 1
                continue

            if not me_parts:
                continue

            # Combined target: all consecutive "Me" messages joined
            target = "<msg>".join(p for p in me_parts if p.strip())

            # Clean up stray delimiters (e.g. from empty attachment placeholders)
            while "<msg><msg>" in target:
                target = target.replace("<msg><msg>", "<msg>")
            # Strip leading/trailing <msg> delimiters.  Use slice-based
            # removal — str.strip("<msg>") would strip individual chars
            # from the set {<,m,s,g,>} and corrupt <reply:N> tokens.
            while target.startswith("<msg>"):
                target = target[5:]
            while target.endswith("<msg>"):
                target = target[:-5]
            target = target.strip()
            if not target:
                continue
            history = conversation[:first_me_idx]
            timestamp = conversation[first_me_idx]["timestamp"]

            # Reply metadata for the first sub-message
            first_msg = conversation[first_me_idx]
            is_reply = first_msg.get("is_reply", False)
            reply_to_idx_raw = first_msg.get("reply_to")
            reply_to_idx = reply_to_idx_raw

            # ── Apply filters ──
            if not _passes_filters(history, target, filter_config, filter_stats):
                continue

            # ── Truncate history to max_input_turns (sliding window) ──
            # Instead of dropping examples with long history, keep them
            # and use only the most recent N turns as context.  Only
            # *text* messages count toward the limit; non-text messages
            # (GamePigeon, attachments, etc.) ride along for free so
            # they don't waste context slots.  When we truncate,
            # reply_to indices must be remapped so they point to the
            # correct position in the truncated window (or dropped if
            # the referenced message was trimmed away).
            truncation_offset = 0
            if filter_config.max_input_turns:
                # Count text messages from the end to find the cut point
                text_count = 0
                cut_idx = 0  # index into history where we start keeping
                for ri in range(len(history) - 1, -1, -1):
                    if not _is_non_text_message(history[ri]["content"], filter_config):
                        text_count += 1
                        if text_count >= filter_config.max_input_turns:
                            cut_idx = ri
                            break
                else:
                    # Fewer than max_input_turns text messages — keep all
                    cut_idx = 0

                if cut_idx > 0:
                    filter_stats["history_truncated"] += 1
                    truncation_offset = cut_idx
                    history = history[cut_idx:]

                # Remap <reply:N> tokens in target — they were built
                # with original indices so shift them by truncation_offset.
                _max_display = len(history)  # capture for closure
                def _remap_reply(match, _offset=truncation_offset, _max=_max_display):
                    old_num = int(match.group(1))
                    new_num = old_num - _offset
                    if new_num < 1 or new_num > _max:
                        return ""  # replied-to message outside visible window
                    return f"<reply:{new_num}>"
                target = _REPLY_TOKEN_RE.sub(_remap_reply, target)

                # Also remap the metadata reply_to_idx
                if reply_to_idx is not None:
                    reply_to_idx = reply_to_idx_raw - truncation_offset
                    if reply_to_idx < 0 or reply_to_idx >= len(history):
                        is_reply = False
                        reply_to_idx = None

            # General validation: strip reply tokens that reference
            # messages outside the visible history window (e.g. a sub-
            # message replying to another "Me" message in the same burst).
            _hist_len = len(history)
            def _strip_oob_reply(match, _max=_hist_len):
                n = int(match.group(1))
                if n < 1 or n > _max:
                    return ""
                return match.group(0)
            target = _REPLY_TOKEN_RE.sub(_strip_oob_reply, target)

            if reply_to_idx is not None and (reply_to_idx < 0 or reply_to_idx >= _hist_len):
                is_reply = False
                reply_to_idx = None

            # ── Annotate goal ──
            goal = None
            goal_source = None
            if gen_config.annotate_goals:
                # Prefer LLM-generated conversation-level goal
                if conversation_goals and conv_idx < len(conversation_goals) and conversation_goals[conv_idx]:
                    goal = conversation_goals[conv_idx]
                    goal_source = "llm_conversation_level"
                else:
                    goal = annotate_goal_rule_based(history, target)
                    goal_source = "rule_based"

            # ── Build system prompt ──
            system_prompt = gen_config.base_system_prompt.format(
                persona_name=gen_config.persona_name
            )
            if gen_config.annotate_goals and goal:
                system_prompt += gen_config.goal_prompt_template.format(goal=goal)
            if gen_config.inject_relationship_context:
                system_prompt += gen_config.relationship_prompt_template.format(
                    other_name=other_name,
                    relationship_description=relationship_desc,
                )

            # ── Format input text (with numbered messages) ──
            input_parts = []

            # Prior conversations
            if prior_convs:
                input_parts.append("Previous conversations:\n")
                for pc in prior_convs:
                    conv_text = "\n".join(
                        f"[{m['timestamp']}] {m['speaker']}: "
                        f"{_REPLY_RAW_RE.sub('', m['content'])}"
                        for m in pc
                    )
                    input_parts.append(f"{conv_text}\n---\n")

            # Current conversation history — numbered [1], [2], ...
            input_parts.append("Current conversation:\n")
            for h_idx, m in enumerate(history):
                msg_num = h_idx + 1  # 1-based
                # Include reply annotation for past messages
                # If history was truncated, remap reply_to to the new index
                reply_tag = ""
                if m.get("is_reply") and m.get("reply_to") is not None:
                    orig_reply_to = m["reply_to"]
                    remapped = orig_reply_to - truncation_offset
                    if remapped >= 0:
                        reply_tag = f"<reply:{remapped + 1}>"
                    # else: replied-to message was truncated away, drop tag
                content = f"{reply_tag}{m['content']}" if reply_tag else m["content"]
                # Resolve <reply_raw:N> markers from combined sub-messages
                _trunc = truncation_offset
                _hlen = len(history)
                def _resolve_raw(match, _off=_trunc, _mx=_hlen):
                    idx = int(match.group(1))
                    rem = idx - _off
                    if rem < 0 or rem >= _mx:
                        return ""
                    return f"<reply:{rem + 1}>"
                content = _REPLY_RAW_RE.sub(_resolve_raw, content)
                input_parts.append(
                    f"[{msg_num}] [{m['timestamp']}] {m['speaker']}: {content}\n"
                )

            # Current turn prompt
            current_num = len(history) + 1
            input_parts.append(f"[{current_num}] [{timestamp}] Me: ")
            input_text = "".join(input_parts)

            example = {
                "instruction": system_prompt,
                "input": input_text,
                "output": target,
                "timestamp": timestamp,
                "metadata": {
                    "goal": goal,
                    "goal_source": goal_source,
                    "other_speaker": other_name,
                    "relationship": relationship_desc,
                    "conversation_idx": conv_idx,
                    "message_idx": first_me_idx,
                    "num_history_turns": len(history),
                    "is_reply": is_reply,
                    "reply_to_msg_num": (reply_to_idx + 1) if reply_to_idx is not None else None,
                    "num_sub_messages": len(me_parts),
                },
            }
            all_examples.append(example)

    return all_examples, dict(filter_stats)


# ═══════════════════════════════════════════════
# STEP 4: Filtering
# ═══════════════════════════════════════════════

def _passes_filters(
    history: List[Dict],
    target: str,
    config: FilterConfig,
    stats: Counter,
) -> bool:
    """Return True if this example passes all filters."""

    # Blocked strings — only check the OUTPUT (target), not history.
    # The model should handle seeing these in context; it just shouldn't produce them.
    target_lower = target.lower()
    for bs in config.blocked_strings:
        if bs.lower() in target_lower:
            stats["blocked_string"] += 1
            return False

    # Min output words
    if len(target.split()) < config.min_output_words:
        stats["too_short_output"] += 1
        return False

    # Min input turns
    if len(history) < config.min_input_turns:
        stats["too_few_input_turns"] += 1
        return False
    # NOTE: max_input_turns is handled via truncation (not filtering)
    # in generate_training_examples — we keep the example and trim
    # history to the most recent N turns.

    # Emoji-only output
    if config.filter_emoji_only and _is_emoji_only(target):
        stats["emoji_only_output"] += 1
        return False

    # URLs in output
    if config.filter_urls_in_output and _contains_url(target):
        stats["url_in_output"] += 1
        return False

    # Attachment filenames (e.g. "IMG_9790.HEIC", "video.mov")
    if config.filter_attachment_filenames and _is_attachment_filename(target):
        stats["attachment_filename"] += 1
        return False

    # Duplicate consecutive messages (e.g. same text sent twice due to
    # iMessage sync issues).  Check <msg>-separated parts for back-to-back
    # identical content.
    if config.filter_duplicate_messages and "<msg>" in target:
        parts = [p.strip() for p in target.split("<msg>")]
        for i in range(1, len(parts)):
            if parts[i] and parts[i] == parts[i - 1]:
                stats["duplicate_message"] += 1
                return False

    # Group chats
    if config.filter_group_chats:
        speakers = {m["speaker"] for m in history if m["speaker"] != "Me"}
        if len(speakers) > 1:
            stats["group_chat"] += 1
            return False

    return True


def _is_emoji_only(text: str) -> bool:
    emoji_pattern = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+|"
        "[\U0001F1E6-\U0001F1FF]{2}|"
        "[\u200d\u2640-\u2642\u2600-\u2B55\u23cf\u23e9\u231a\u3030\ufe0f]"
        "+"
    )
    cleaned = emoji_pattern.sub("", text.strip())
    return len(cleaned) == 0


def _contains_url(text: str) -> bool:
    url_patterns = [
        "http://", "https://", "www.", ".com/", ".org/", ".edu/",
    ]
    return any(p in text.lower() for p in url_patterns)


# Common media / attachment file extensions the model can never produce.
_REPLY_TOKEN_RE = re.compile(r"<reply:(\d+)>")
_REPLY_RAW_RE = re.compile(r"<reply_raw:(\d+)>")

_ATTACHMENT_EXTS = (
    r"heic|jpg|jpe|jpeg|png|gif|bmp|tiff|webp|svg"
    r"|mov|mp4|avi|mkv|m4v"
    r"|mp3|aac|m4a|wav|caf|ogg"
    r"|pdf|docx?|xlsx?|pptx?|csv|txt|zip|rar"
    r"|vcf|pkpass"
)

_ATTACHMENT_RE = re.compile(
    # Allow spaces in filenames (e.g. "Jane Doe.vcf", "My Photo.heic")
    r"^(<reply:\d+>)?\s*.+\."
    rf"({_ATTACHMENT_EXTS})"
    # Handle double-extensions like "IMG_1449.PNG.jpeg" or truncated "IMG_1449.PNG.jpe"
    r"(\.\w+)?\s*$",
    re.IGNORECASE,
)

# iMessage exports format attachments as "TYPE filename.ext"
# e.g. "PNG image.png", "JPEG image.jpeg", "GIF image.gif"
_ATTACHMENT_PREFIXED_RE = re.compile(
    r"^(<reply:\d+>)?\s*"
    r"(?:PNG|JPEG|JPG|GIF|TIFF|BMP|WEBP|MOV|MP4|PDF|HEIC)\s+"
    r".+\."
    rf"({_ATTACHMENT_EXTS})"
    r"(\.\w+)?\s*$",
    re.IGNORECASE,
)

def _is_attachment_filename(text: str) -> bool:
    """Return True if the output is just a media/file attachment filename."""
    return bool(_ATTACHMENT_RE.match(text) or _ATTACHMENT_PREFIXED_RE.match(text))


def _is_non_text_message(content: str, filter_config: FilterConfig) -> bool:
    """Return True if a message is functionally non-text content.

    These are messages that the model can never see or produce meaningfully:
    attachments, GamePigeon, URLs-only, Find My, etc.  Used to strip them
    from conversation history *before* generating examples so they don't
    pollute the context or inflate conversation length counts.
    """
    text = content.strip()
    if not text:
        return True
    if _is_attachment_filename(text):
        return True
    for bs in filter_config.blocked_strings:
        if bs.lower() in text.lower():
            return True
    # A message that is ONLY a URL (no surrounding text) is non-text
    if _contains_url(text) and len(text.split()) <= 3:
        return True
    return False



# ═══════════════════════════════════════════════
# STEP 5: Statistics
# ═══════════════════════════════════════════════

def compute_dataset_stats(examples: List[Dict]) -> Dict:
    """Compute summary statistics for the generated dataset."""
    if not examples:
        return {"num_examples": 0}

    input_lengths = [len(ex["input"].split()) for ex in examples]
    output_lengths = [len(ex["output"].split()) for ex in examples]
    history_turns = [ex["metadata"]["num_history_turns"] for ex in examples]

    goal_sources = Counter(ex["metadata"].get("goal_source") for ex in examples)
    goals_preview = []
    for ex in examples[:5]:
        g = ex["metadata"].get("goal")
        if g:
            goals_preview.append(g[:80] + ("..." if len(g) > 80 else ""))

    # Timestamps
    timestamps = []
    for ex in examples:
        try:
            timestamps.append(datetime.strptime(ex["timestamp"], "%Y-%m-%d %H:%M:%S"))
        except (ValueError, TypeError):
            pass

    date_range = {}
    if timestamps:
        date_range = {
            "earliest": min(timestamps).strftime("%Y-%m-%d"),
            "latest": max(timestamps).strftime("%Y-%m-%d"),
        }

    import statistics
    def safe_stats(values):
        return {
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.mean(values), 1),
            "median": round(statistics.median(values), 1),
            "stdev": round(statistics.stdev(values), 1) if len(values) > 1 else 0,
        }

    return {
        "num_examples": len(examples),
        "date_range": date_range,
        "input_word_counts": safe_stats(input_lengths),
        "output_word_counts": safe_stats(output_lengths),
        "history_turn_counts": safe_stats(history_turns),
        "goal_sources": dict(goal_sources.most_common()),
        "goals_preview": goals_preview,
        "other_speaker": examples[0]["metadata"]["other_speaker"] if examples else None,
        "relationship": examples[0]["metadata"]["relationship"] if examples else None,
    }


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def _llm_config_to_annotator(llm_config: LLMConfig):
    """Convert centralized LLMConfig to an AnnotatorConfig for llm_annotator."""
    from IMPersona2.llm_annotator import AnnotatorConfig
    return AnnotatorConfig(
        model=llm_config.model,
        reasoning_effort=llm_config.reasoning_effort,
        max_concurrency=llm_config.max_concurrency,
    )


def conversation_id_from_path(path: str) -> str:
    """Extract a short conversation identifier from an input file path.

    Examples:
        "imessage_export_mac/Jane Doe.txt"  →  "Jane Doe"
        "imessage_export_mac/+15555550123.txt" →  "+15555550123"
        "chats/Mom.txt"                        →  "Mom"
    """
    return os.path.splitext(os.path.basename(path))[0]


def run_pipeline(
    config: PipelineConfig,
    *,
    input_file: str,
    output_file: str,
    llm_cache_file: str,
    contacts: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict], Dict, Dict, Optional[Dict]]:
    """
    Run the full pipeline for a SINGLE input file: parse → generate → filter → stats.

    Args:
        config:         Global pipeline configuration (persona, LLM, parsing, etc.).
        input_file:     Path to the raw iMessage export to process.
        output_file:    Where to write the bundled dataset JSON.
        llm_cache_file: Where to cache expensive LLM outputs.
        contacts:       Phone→name lookup (built from VCF).  Optional.

    Returns:
        (examples, filter_stats, dataset_stats, llm_usage_or_None)
    """
    # Sync persona_name from PersonaConfig → GenerationConfig
    config.generation.persona_name = config.persona.name

    conv_id = conversation_id_from_path(input_file)

    print(f"Pipeline config:")
    print(f"  Persona: {config.persona.full_name} ({', '.join(config.persona.nicknames)})")
    print(f"  LLM:     {config.llm.model} (reasoning: {config.llm.reasoning_effort}, concurrency: {config.llm.max_concurrency})")
    print()
    print(f"[1/6] Parsing {input_file} ...")
    conversations = parse_imessage_file(input_file, config.parsing, contacts=contacts)
    print(f"       → {len(conversations)} conversations found")

    # ── Create a shared LLM annotator (reused across all steps for cost tracking) ──
    annotator = None
    if config.use_llm_relationship or config.use_llm_goals or config.use_llm_cot:
        from IMPersona2.llm_annotator import LLMAnnotator
        annotator = LLMAnnotator(_llm_config_to_annotator(config.llm))

    # ── Load LLM cache (relationship summary + conversation goals) ──
    llm_cache = {}
    if os.path.exists(llm_cache_file):
        with open(llm_cache_file, "r", encoding="utf-8") as f:
            llm_cache = json.load(f)

    def _save_llm_cache():
        os.makedirs(os.path.dirname(llm_cache_file), exist_ok=True)
        with open(llm_cache_file, "w", encoding="utf-8") as f:
            json.dump(llm_cache, f, indent=2, ensure_ascii=False)

    # ── Load or generate relationship summary ──
    relationship_summary = None
    if config.generation.inject_relationship_context:
        cached_rel = llm_cache.get("relationship_summary")
        if cached_rel:
            relationship_summary = cached_rel
            print(f"[2/6] Loaded cached relationship summary ({len(relationship_summary)} chars)")
        elif config.use_llm_relationship:
            print(f"[2/6] Generating LLM-based relationship summary ...")
            try:
                other_name = _detect_other_speaker(conversations)
                persona = config.persona
                relationship_summary = annotator.analyze_relationship(
                    conversations, other_name,
                    persona_full_name=persona.full_name,
                    persona_nicknames=", ".join(persona.nicknames),
                    persona_bio=persona.bio,
                )
                llm_cache["relationship_summary"] = relationship_summary
                _save_llm_cache()
                print(f"       → Generated & cached ({len(relationship_summary)} chars)")
            except Exception as e:
                print(f"       ⚠ LLM relationship analysis failed: {e}")
                print(f"       → Falling back to rule-based heuristic")
        else:
            print(f"[2/6] Using rule-based relationship heuristic (LLM disabled)")

    # ── Load or generate conversation-level goals ──
    # Goals are cached by content hash so the cache survives conversation
    # re-indexing (e.g. when the gap-splitting settings change).
    conversation_goals = None
    if config.generation.annotate_goals:
        from IMPersona2.llm_annotator import LLMAnnotator as _LLMAnnotator

        goals_hash_cache: Dict[str, str] = llm_cache.get("conversation_goals_by_hash", {})

        if config.use_llm_goals:
            print(f"[3/6] Loading / generating conversation goals ({len(conversations)} conversations) ...")
            try:
                other_name = _detect_other_speaker(conversations)
                persona = config.persona
                conversation_goals = annotator.generate_conversation_goals_batch(
                    conversations, config.generation.persona_name, other_name,
                    persona_full_name=persona.full_name,
                    persona_nicknames=", ".join(persona.nicknames),
                    persona_bio=persona.bio,
                    cache=goals_hash_cache,
                )
                # Save updated hash cache (includes any newly generated goals)
                llm_cache["conversation_goals_by_hash"] = goals_hash_cache
                _save_llm_cache()
            except Exception as e:
                print(f"       ⚠ LLM goal generation failed: {e}")
                print(f"       → Falling back to rule-based annotator")
        else:
            # LLM disabled — try loading from hash cache directly
            all_cached = True
            conversation_goals = []
            for i, conv in enumerate(conversations):
                prior_convs = None
                if i > 0:
                    prior_convs = conversations[max(0, i - 1):i]
                h = _LLMAnnotator.goal_prompt_hash(conv, prior_convs)
                if h in goals_hash_cache:
                    conversation_goals.append(goals_hash_cache[h])
                else:
                    all_cached = False
                    break
            if all_cached and conversation_goals:
                print(f"[3/6] Loaded {len(conversation_goals)} goals from content-keyed cache (LLM disabled)")
            else:
                conversation_goals = None
                print(f"[3/6] Using rule-based goal annotator (LLM disabled, no cache)")

    print(f"[4/6] Generating training examples ...")
    print(f"       Goals: {'ON' if config.generation.annotate_goals else 'OFF'}")
    print(f"       Relationship context: {'ON' if config.generation.inject_relationship_context else 'OFF'}")
    if relationship_summary:
        print(f"       Relationship source: LLM-generated summary")
    if conversation_goals:
        print(f"       Goal source: LLM conversation-level goals")
    examples, filter_stats = generate_training_examples(
        conversations, config.generation, config.filtering,
        relationship_summary=relationship_summary,
        conversation_goals=conversation_goals,
        max_combine_gap_hours=config.parsing.max_combine_gap_hours,
    )
    print(f"       → {len(examples)} examples generated")

    # ── Generate chain-of-thought ──
    #
    # CoTs are generated SEQUENTIALLY within each conversation so that
    # later messages can see the prior monologues and avoid redundant
    # re-stating of context.  Different conversations are processed in
    # PARALLEL via ThreadPoolExecutor.
    #
    if config.use_llm_cot:
        import hashlib
        import threading
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Build a lookup of cached CoTs keyed by hash(input + output)
        cached_cots = llm_cache.get("chain_of_thought", {})

        def _cot_key(ex):
            raw = (ex["input"] + "||" + ex["output"]).encode()
            return hashlib.sha256(raw).hexdigest()[:16]

        # ── Group examples by conversation ──
        conv_groups: Dict[int, List[Tuple[int, dict, str]]] = defaultdict(list)
        num_cached = 0
        num_uncached = 0
        for i, ex in enumerate(examples):
            conv_idx = ex.get("metadata", {}).get("conversation_idx", -1)
            key = _cot_key(ex)
            conv_groups[conv_idx].append((i, ex, key))
            if key in cached_cots:
                num_cached += 1
            else:
                num_uncached += 1

        if num_uncached > 0:
            print(f"[5/6] Generating chain-of-thought ({num_uncached} new, {num_cached} cached, {len(conv_groups)} conversations) ...")
            try:
                persona_full = config.persona.full_name
                persona_nicks = ", ".join(config.persona.nicknames)

                lock = threading.Lock()
                generated_count = 0
                error_count = 0
                _CHECKPOINT_INTERVAL = 50  # flush cache to disk every N new CoTs

                def _checkpoint_cache():
                    """Flush cached CoTs to disk so progress survives crashes."""
                    llm_cache["chain_of_thought"] = cached_cots
                    _save_llm_cache()

                def _process_conversation(conv_idx: int, items: List[Tuple[int, dict, str]]):
                    """Process a single conversation's examples sequentially,
                    accumulating prior CoTs keyed by timestamp.

                    The timestamp→CoT map is passed to the LLM so that prior
                    monologues are shown inline next to their corresponding
                    "Me" messages in the conversation history.  Truncated
                    messages are handled automatically — if a "Me" message was
                    trimmed from the history, its timestamp simply won't
                    appear and the CoT won't be shown.
                    """
                    nonlocal generated_count, error_count

                    # Map timestamp → CoT for "Me" messages in this conversation.
                    # Timestamps are unique per message so this gives a reliable
                    # 1:1 association between CoTs and messages.
                    prior_cot_map: Dict[str, str] = {}

                    for global_idx, ex, cache_key in items:
                        ts = ex.get("timestamp", "")

                        if cache_key in cached_cots:
                            # Use cached CoT — still add to map for later messages
                            cot = cached_cots[cache_key]
                            meta = ex.get("metadata", {})
                            meta["chain_of_thought"] = cot
                            ex["output"] = f"<think>{cot}</think>{ex['output']}"
                            ex["metadata"] = meta
                            examples[global_idx] = ex
                            if not cot.startswith("[SKIP]") and not cot.startswith("[COT GENERATION FAILED"):
                                prior_cot_map[ts] = cot
                        else:
                            # Generate new CoT with inline prior context
                            history = annotator._parse_history_from_input(ex["input"])
                            other_speaker = ex.get("metadata", {}).get("other_speaker", "the other person")
                            conversation_context = annotator._extract_conversation_context(ex.get("instruction", ""))

                            # Build after-context: messages immediately following
                            # "Me"'s response in the original conversation.
                            # This lets the quality gate detect interrupted fragments
                            # (overlapping typing where the other person's message
                            # landed seconds after this response).
                            after_ctx = None
                            meta_conv = ex.get("metadata", {}).get("conversation_idx")
                            meta_msg = ex.get("metadata", {}).get("message_idx")
                            if meta_conv is not None and meta_msg is not None:
                                try:
                                    conv = conversations[meta_conv]
                                    # Skip past the "Me" turn
                                    after_start = meta_msg
                                    while after_start < len(conv) and conv[after_start]["speaker"] == "Me":
                                        after_start += 1
                                    # Take next 3 messages for context
                                    after_msgs = conv[after_start:after_start + 3]
                                    if after_msgs:
                                        after_ctx = [
                                            {"timestamp": m["timestamp"], "speaker": m["speaker"], "content": m["content"]}
                                            for m in after_msgs
                                        ]
                                except (IndexError, KeyError):
                                    pass

                            cot = annotator.generate_cot(
                                history, ex["output"],
                                other_speaker=other_speaker,
                                persona_full_name=persona_full,
                                persona_nicknames=persona_nicks,
                                conversation_context=conversation_context,
                                prior_cots=prior_cot_map if prior_cot_map else None,
                                after_context=after_ctx,
                            )

                            has_error = cot.startswith("[COT GENERATION FAILED")

                            meta = ex.get("metadata", {})
                            meta["chain_of_thought"] = cot
                            meta["cot_model"] = annotator.config.model
                            ex["output"] = f"<think>{cot}</think>{ex['output']}"
                            ex["metadata"] = meta
                            examples[global_idx] = ex

                            # Cache the result
                            if cot and not has_error:
                                with lock:
                                    cached_cots[cache_key] = cot

                            if not cot.startswith("[SKIP]") and not has_error:
                                prior_cot_map[ts] = cot

                            with lock:
                                generated_count += 1
                                if has_error:
                                    error_count += 1
                                pct = generated_count / num_uncached * 100
                                sys.stdout.write(
                                    f"\r  Generating CoT [{generated_count}/{num_uncached}] ({pct:.0f}%) "
                                    f"— errors: {error_count}"
                                )
                                sys.stdout.flush()

                                # ── Periodic checkpoint ──
                                if generated_count % _CHECKPOINT_INTERVAL == 0:
                                    _checkpoint_cache()

                # Process conversations in parallel (sequential within each)
                workers = min(config.llm.max_concurrency, len(conv_groups))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(_process_conversation, cidx, items): cidx
                        for cidx, items in conv_groups.items()
                    }
                    for future in as_completed(futures):
                        future.result()  # raise any exceptions

                print()  # newline after progress
                _checkpoint_cache()  # final save
                print(f"       → Generated & cached {generated_count} chain-of-thoughts")
            except Exception as e:
                # ── Save whatever we have so progress isn't lost ──
                print(f"\n       ⚠ CoT generation failed: {e}")
                print(f"       💾 Saving {len(cached_cots)} cached CoTs before exit...")
                llm_cache["chain_of_thought"] = cached_cots
                _save_llm_cache()
                import traceback; traceback.print_exc()
        else:
            # All cached — still need to apply them with prior-context ordering
            for conv_idx, items in conv_groups.items():
                for global_idx, ex, cache_key in items:
                    if cache_key in cached_cots:
                        cot = cached_cots[cache_key]
                        meta = ex.get("metadata", {})
                        meta["chain_of_thought"] = cot
                        ex["output"] = f"<think>{cot}</think>{ex['output']}"
                        ex["metadata"] = meta
                        examples[global_idx] = ex
            print(f"[5/6] All {len(examples)} chain-of-thoughts loaded from cache")

        # ── Quality gate: filter out examples flagged [SKIP] by the CoT LLM ──
        pre_skip_count = len(examples)
        skipped_reasons = []
        kept_examples = []
        for ex in examples:
            cot = ex.get("metadata", {}).get("chain_of_thought", "")
            if cot.startswith("[SKIP]"):
                reason = cot[len("[SKIP]"):].strip()
                skipped_reasons.append(reason)
                filter_stats["cot_quality_skip"] = filter_stats.get("cot_quality_skip", 0) + 1
            else:
                kept_examples.append(ex)
        if skipped_reasons:
            examples = kept_examples
            print(f"       → Quality gate: removed {len(skipped_reasons)} bad datapoints:")
            for reason in skipped_reasons:
                print(f"           • {reason}")
    else:
        print(f"[5/6] Chain-of-thought generation disabled")

    print(f"[6/6] Filter statistics & dataset statistics ...")
    if filter_stats:
        for reason, count in sorted(filter_stats.items(), key=lambda x: -x[1]):
            print(f"         {reason}: {count} removed")
    else:
        print(f"         (no examples filtered)")

    stats = compute_dataset_stats(examples)
    print(f"       → {stats['num_examples']} examples")
    if stats.get("date_range"):
        print(f"       → Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
    if stats.get("goal_sources"):
        print(f"       → Goal sources: {dict(stats['goal_sources'])}")
    if stats.get("goals_preview"):
        print(f"       → Sample goals:")
        for g in stats["goals_preview"][:3]:
            print(f"           \"{g}\"")

    # ── Compute LLM usage ──
    usage = None
    if annotator is not None:
        usage = annotator.get_usage()
        if usage["num_calls"] == 0:
            usage = None  # no actual calls made (all cached)

    # ── Bundle everything into a single output file ──
    dataset_bundle = {
        "metadata": {
            "conversation_id": conv_id,
            "model": config.llm.model,
            "input_file": input_file,
            "relationship_summary": relationship_summary,
            "conversation_goals": conversation_goals,
            "stats": stats,
            "filter_stats": filter_stats,
            "num_conversations": len(conversations),
            "num_examples": len(examples),
            "other_speaker": stats.get("other_speaker"),
            "llm_usage": usage,
        },
        "examples": examples,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset_bundle, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Dataset saved to {output_file}")

    # ── Print cost summary ──
    if usage:
        print(f"\n💰 LLM Cost Summary ({usage['model']}):")
        print(f"   API calls:     {usage['num_calls']:,}")
        print(f"   Input tokens:  {usage['input_tokens']:,}")
        print(f"   Output tokens: {usage['output_tokens']:,}")
        print(f"   Total tokens:  {usage['total_tokens']:,}")
        print(f"   Estimated cost: ${usage['total_cost_usd']:.2f}"
              f"  (input: ${usage['input_cost_usd']:.2f}"
              f" + output: ${usage['output_cost_usd']:.2f})")

    return examples, filter_stats, stats, usage


def run_all_pipelines(config: PipelineConfig):
    """
    Process all input files listed in config.input_files.

    Each file gets its own subfolder under config.output_dir, keyed by model:
        output_dir/<conversation_id>/<model>/dataset.json
        output_dir/<conversation_id>/<model>/llm_cache.json

    Already-processed conversations (where dataset.json exists) are skipped
    unless config.force_reprocess is True.

    Multiple contacts are processed in parallel (up to
    config.max_parallel_contacts at a time).  Each contact's full pipeline
    runs independently.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    input_files = config.input_files
    if not input_files:
        print("❌ No input files configured. Add files to PipelineConfig.input_files.")
        return

    # Build contacts lookup once for the entire run
    contacts = _build_contacts_lookup(config.contacts_file)
    if contacts:
        print(f"📇 Loaded {len(contacts)} contact entries from {config.contacts_file}")

    print(f"{'='*60}")
    print(f"Impersona-Env — Multi-Conversation Pipeline")
    print(f"{'='*60}")
    print(f"  Input files:        {len(input_files)}")
    print(f"  Output dir:         {config.output_dir}")
    print(f"  Force rerun:        {config.force_reprocess}")
    print(f"  Parallel contacts:  {config.max_parallel_contacts}")
    print()

    model_name = config.llm.model
    if config.dataset_version:
        model_name = f"{model_name}_{config.dataset_version}"

    # ── Separate files into skip vs. process ──
    to_process: list = []   # (index_1based, input_file, output_file, cache_file)
    skipped = 0

    for i, input_file in enumerate(input_files, 1):
        conv_id = conversation_id_from_path(input_file)
        conv_dir = os.path.join(config.output_dir, conv_id, model_name)
        output_file = os.path.join(conv_dir, "dataset.json")
        cache_file = os.path.join(conv_dir, "llm_cache.json")

        if os.path.exists(output_file) and not config.force_reprocess:
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    bundle = json.load(f)
                n = bundle.get("metadata", {}).get("num_examples", 0)
                speaker = bundle.get("metadata", {}).get("other_speaker", "?")
                print(f"⏭️  [{i}/{len(input_files)}] Skipping {conv_id} — already processed ({n} examples, speaker: {speaker})")
                skipped += 1
                continue
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupted file — reprocess

        if not os.path.exists(input_file):
            print(f"❌ [{i}/{len(input_files)}] Input file not found: {input_file} — skipping")
            continue

        to_process.append((i, input_file, output_file, cache_file))

    if not to_process:
        print(f"\n{'='*60}")
        print(f"✅ Done: 0 processed, {skipped} skipped (all up-to-date)")
        print(f"{'='*60}")
        return

    # ── Process contacts in parallel ──
    lock = threading.Lock()
    processed = 0
    errors = 0
    all_usages: List[Dict] = []

    def _run_one(item):
        nonlocal processed, errors
        idx, input_file, output_file, cache_file = item
        conv_id = conversation_id_from_path(input_file)
        with lock:
            print(f"\n{'='*60}")
            print(f"[{idx}/{len(input_files)}] Processing: {conv_id}")
            print(f"{'='*60}\n")
        try:
            _, _, _, usage = run_pipeline(
                config,
                input_file=input_file,
                output_file=output_file,
                llm_cache_file=cache_file,
                contacts=contacts,
            )
            with lock:
                processed += 1
                if usage and usage.get("num_calls", 0) > 0:
                    all_usages.append(usage)
        except Exception as e:
            with lock:
                errors += 1
                print(f"❌ [{idx}/{len(input_files)}] {conv_id} failed: {e}")
                import traceback; traceback.print_exc()

    workers = min(config.max_parallel_contacts, len(to_process))
    if workers <= 1:
        # Sequential — cleaner output
        for item in to_process:
            _run_one(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, item): item for item in to_process}
            for future in as_completed(futures):
                future.result()  # propagate unexpected errors

    print(f"\n{'='*60}")
    print(f"✅ Done: {processed} processed, {skipped} skipped, {errors} errors")

    # ── Aggregate cost across all processed contacts ──
    if all_usages:
        total_calls = sum(u["num_calls"] for u in all_usages)
        total_input = sum(u["input_tokens"] for u in all_usages)
        total_output = sum(u["output_tokens"] for u in all_usages)
        total_cost = sum(u["total_cost_usd"] for u in all_usages)
        print(f"\n💰 Total LLM Cost ({all_usages[0]['model']}):")
        print(f"   API calls:     {total_calls:,}")
        print(f"   Input tokens:  {total_input:,}")
        print(f"   Output tokens: {total_output:,}")
        print(f"   Total tokens:  {total_input + total_output:,}")
        print(f"   Estimated cost: ${total_cost:.2f}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Impersona-Env — Dataset Generation Pipeline"
    )
    parser.add_argument("--input_file", type=str, default=None,
                        help="Process a single file (overrides config.input_files)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess conversations even if output already exists")
    parser.add_argument("--no-goals", action="store_true",
                        help="Disable goal/intent annotation")
    parser.add_argument("--no-relationship", action="store_true",
                        help="Disable relationship context injection")
    parser.add_argument("--no-cot", action="store_true",
                        help="Disable chain-of-thought generation")
    parser.add_argument("--conversation-buffer", type=int, default=None,
                        help="Number of prior conversations as context")
    parser.add_argument("--max-gap-hours", type=float, default=None,
                        help="Max gap (hours) between messages for conversation splitting")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the annotation model (any model id; default: annotation in configs/models.yaml)")

    args = parser.parse_args()

    # Build config with CLI overrides
    config = PipelineConfig()

    if args.force:
        config.force_reprocess = True
    if args.no_goals:
        config.generation.annotate_goals = False
    if args.no_relationship:
        config.generation.inject_relationship_context = False
    if args.no_cot:
        config.use_llm_cot = False
    if args.conversation_buffer is not None:
        config.generation.conversation_buffer = args.conversation_buffer
    if args.max_gap_hours is not None:
        config.parsing.max_gap_hours = args.max_gap_hours
    if args.model:
        config.llm.model = args.model

    if args.input_file:
        # Single-file mode: process just this one file
        contacts = _build_contacts_lookup(config.contacts_file)
        conv_id = conversation_id_from_path(args.input_file)
        model_name = config.llm.model
        if config.dataset_version:
            model_name = f"{model_name}_{config.dataset_version}"
        conv_dir = os.path.join(config.output_dir, conv_id, model_name)
        output_file = os.path.join(conv_dir, "dataset.json")
        cache_file = os.path.join(conv_dir, "llm_cache.json")
        _, _, _, _ = run_pipeline(
            config,
            input_file=args.input_file,
            output_file=output_file,
            llm_cache_file=cache_file,
            contacts=contacts,
        )
    else:
        # Multi-file mode: process all configured input files
        run_all_pipelines(config)


if __name__ == "__main__":
    main()
