"""
Style judge — do the user model's messages read like the real person?

The judge sees a fixed random sample of the person's real messages (from the
SFT training data) as a style reference, then scores the simulated user's
visible messages in one eval conversation on a 1–7 scale.

Only the writing is graded. <think> reasoning is ignored, and content is left
to the coherence and fact-recall judges.

Usage:
    python -m IMPersona2.eval --eval-only eval_output/<run> --judges style
"""

import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from IMPersona2.model_config import model_for

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_FILE = REPO_ROOT / "training_data.json"
DEFAULT_REFERENCE_SIZE = 60
BURST_SEP = " ⏎ "

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_REPLY_TAG_RE = re.compile(r"<reply:\d+>\s*")

STYLE_SYSTEM_PROMPT = (
    "You are an expert in sociolinguistics and computer-mediated communication. "
    "You judge whether text messages were written by a specific person, based "
    "only on how they write. Be precise and cite specific messages."
)

STYLE_PROMPT = """\
You are judging whether a simulated user's text messages read as if they were
written by a specific real person, "{persona_name}".

REFERENCE — real texts {persona_name} sent to friends and family.
Each line is one burst of messages sent back-to-back, separated by "⏎".
{reference}

CONVERSATION — a simulated {persona_name} talking to an AI chatbot. Grade ONLY
the {persona_name} lines. The AI lines are context.
Scenario: {scenario_context}

{transcript}

WHAT TO GRADE — the writing, not the content:
- Message length, and how thoughts are split across back-to-back messages
- Capitalization, punctuation, and typo habits
- Abbreviations, slang, emoji, and filler words
- Recurring phrasings or verbal tics
- Tone: directness, humor, how feelings are expressed
- Register: how formal or casual they are

Adjust for the audience. {persona_name} is talking to an AI, not a friend, so
different topics and somewhat less intimacy are expected. Judge whether it
still sounds like {persona_name} typing.

Signs the writing does NOT read as {persona_name}:
- Polished, complete, grammatical sentences where they would text in fragments
- Assistant-like register: effusive thanks, balanced "on the one hand"
  framing, tidy summaries, or corporate phrasing
- Uniform message lengths, or one long message where they would send a burst
- Catchphrases from the reference repeated so often that it reads as parody

Do NOT reward copying specific content from the reference, and do NOT
penalize topics that differ from it.

SCALE
7 — Indistinguishable from the reference; a close friend would not notice.
6 — Clearly them, with one or two slightly-off messages.
5 — Mostly them, with some noticeably generic or too-polished stretches.
4 — Mixed: some of their habits, but often reads like a generic person.
3 — Occasional surface features (slang, lowercase), but mostly generic.
2 — Barely resembles them.
1 — Nothing like them (e.g. reads like an AI assistant).

Respond with ONLY a JSON object (no markdown, no extra text):
{{"score": <integer 1-7>,
  "matches": ["<habit that matches, quoting a message>", ...],
  "mismatches": ["<habit that doesn't, quoting a message>", ...],
  "reasoning": "<2-3 sentences>"}}
"""

PROMPT_SHA = hashlib.sha256((STYLE_SYSTEM_PROMPT + STYLE_PROMPT).encode()).hexdigest()[:10]


def _visible_messages(output: str) -> List[str]:
    """Training-example output → its visible messages (no <think>, no reply tags)."""
    text = _THINK_RE.sub("", output or "")
    return [_REPLY_TAG_RE.sub("", m).strip() for m in text.split("<msg>") if m.strip()]


def load_style_reference(
    path: Path = DEFAULT_REFERENCE_FILE,
    size: int = DEFAULT_REFERENCE_SIZE,
    seed: int = 0,
) -> List[List[str]]:
    """Sample `size` real message bursts from an alpaca-format SFT file.

    A fixed seed gives the same sample every time, so scores stay comparable
    across eval runs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Style reference not found: {path}. It is written by "
            f"`python run.py --step train`; or pass --style-reference."
        )
    examples = json.loads(path.read_text(encoding="utf-8"))
    bursts = [b for b in (_visible_messages(ex.get("output", "")) for ex in examples) if b]
    if not bursts:
        raise ValueError(f"No usable messages in {path}")
    return random.Random(seed).sample(bursts, min(size, len(bursts)))


def format_reference(bursts: List[List[str]]) -> str:
    return "\n".join(f"- {BURST_SEP.join(b)}" for b in bursts)


def format_transcript(conversation: Dict, persona_name: str, ai_chars: int = 400) -> str:
    """One line per burst: consecutive persona messages are joined with ⏎.

    AI replies are shown as the user model saw them (partner <think> stripped).
    """
    from IMPersona2.eval import strip_think_tokens

    lines: List[str] = []
    burst: List[str] = []
    for msg in conversation["messages"]:
        content = (msg.get("content") or "").strip()
        if msg["speaker"] in (persona_name, "Me"):
            content = _REPLY_TAG_RE.sub("", content).strip()
            if content:
                burst.append(content)
            continue
        if burst:
            lines.append(f"{persona_name}: {BURST_SEP.join(burst)}")
            burst = []
        content = " ".join(strip_think_tokens(content).split())
        if len(content) > ai_chars:
            content = content[:ai_chars] + " …"
        lines.append(f"AI: {content}")
    if burst:
        lines.append(f"{persona_name}: {BURST_SEP.join(burst)}")
    return "\n".join(lines)


def _validate(obj: Dict) -> None:
    score = obj["score"]
    if isinstance(score, str):
        score = int(score.strip())
    elif isinstance(score, float) and score.is_integer():
        score = int(score)
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 7:
        raise ValueError(f"score out of range: {obj['score']!r}")
    obj["score"] = score


def build_prompt(conversation: Dict, persona_name: str, reference: str) -> str:
    scenario = conversation["scenario"]
    return STYLE_PROMPT.format(
        persona_name=persona_name,
        reference=reference,
        scenario_context=scenario.get("impersona_context") or scenario.get("description", ""),
        transcript=format_transcript(conversation, persona_name),
    )


def _judge_single(conversation: Dict, client, persona_name: str, reference: str) -> Dict:
    from IMPersona2.eval import ask_json

    base = {"scenario_id": conversation["scenario"]["id"], "run_index": conversation.get("run_index")}
    try:
        obj = ask_json(client, STYLE_SYSTEM_PROMPT, build_prompt(conversation, persona_name, reference),
                       max_tokens=4096, validate=_validate)
    except RuntimeError as e:
        return {**base, "score": None, "error": str(e)}
    return {
        **base,
        "score": obj["score"],
        "matches": obj.get("matches", []),
        "mismatches": obj.get("mismatches", []),
        "reasoning": obj.get("reasoning", ""),
    }


def evaluate_style(
    conversations: List[Dict],
    persona_name: str,
    reference_file: Optional[Path] = None,
    reference_size: int = DEFAULT_REFERENCE_SIZE,
    seed: int = 0,
    max_workers: int = 20,
) -> Dict:
    """Score every conversation's style. Returns a summary dict for results.json."""
    from IMPersona2.eval import AIPartnerClient

    reference_file = Path(reference_file or DEFAULT_REFERENCE_FILE)
    bursts = load_style_reference(reference_file, reference_size, seed)
    reference = format_reference(bursts)
    model = model_for("style_judge")
    client = AIPartnerClient(
        model=model,
        base_url=model_for("style_judge", "base_url"),
        reasoning=model_for("style_judge", "reasoning"),
    )

    print(f"\n  Style judge ({model}, prompt {PROMPT_SHA}): {len(conversations)} conversations, "
          f"{len(bursts)} reference bursts (seed {seed})")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_judge_single, c, client, persona_name, reference) for c in conversations]
        for fut in as_completed(futures):
            r = fut.result()
            print(f"    {r['scenario_id']}: " + (f"{r['score']}/7" if r["score"] is not None else f"FAILED ({r['error']})"))
            results.append(r)

    results.sort(key=lambda r: (r["scenario_id"], r.get("run_index") or 0))
    scores = [r["score"] for r in results if r["score"] is not None]
    summary = {
        "judge_model": model,
        "prompt_sha": PROMPT_SHA,
        "reference_file": str(reference_file),
        "reference_size": len(bursts),
        "reference_seed": seed,
        "num_evaluated": len(results),
        "num_scored": len(scores),
        "average_score": round(sum(scores) / len(scores), 2) if scores else None,
        "per_conversation": results,
    }
    print(f"  Style: {summary['average_score']} / 7 ({len(scores)}/{len(results)} scored)")
    return summary
