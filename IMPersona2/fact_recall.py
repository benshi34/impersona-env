"""
Fact recall — does the user model recall and act on true facts about the person?

A four-stage pipeline. Stages 1–3 build a human-verified fact bank once;
stage 4 scores any eval run against it.

  1. propose  For each eval scenario, an LLM picks facts about the person they
              would plausibly draw on in that situation, phrased as questions.
              Facts go into one shared catalog: a scenario reuses an existing
              question when the same fact applies, so each fact is searched
              and reviewed only once.
  2. search   A search agent answers each question from the person's real
              message history (the memory-search index). It writes queries,
              reads the retrieved conversations, drops irrelevant ones, and
              refines for a few rounds. It returns an answer with verbatim
              evidence quotes, or NOT_FOUND.
  3. review   You verify each found fact in the terminal: accept, edit, or
              reject. Only verified facts are ever used for scoring.
  4. judge    For each eval conversation, the verified facts for its scenario
              are inserted into a judge prompt, which labels each one as used
              correctly, contradicted, missed, or not applicable.

The fact bank lives in .fact_bank/fact_bank.json (gitignored — it holds real
facts about you). Every stage saves as it goes and can be resumed.

Usage:
    python -m IMPersona2.fact_recall propose     # stage 1
    python -m IMPersona2.fact_recall search      # stage 2 (needs the memory index)
    python -m IMPersona2.fact_recall review      # stage 3 (interactive)
    python -m IMPersona2.fact_recall status
    python -m IMPersona2.eval --eval-only eval_output/<run> --judges facts   # stage 4
"""

import argparse
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from IMPersona2.experiment_config import default_persona_name
from IMPersona2.model_config import model_for

REPO_ROOT = Path(__file__).resolve().parent.parent
FACT_BANK_FILE = REPO_ROOT / ".fact_bank" / "fact_bank.json"
SCENARIOS_FILE = Path(__file__).resolve().parent / "eval_scenarios.json"
MEMORY_DIR = REPO_ROOT / ".memory_corpus"

KINDS = ("fact", "preference", "decision")
VERIFIED = ("accepted", "edited")
LABELS = ("used_correctly", "contradicted", "missed", "not_applicable")

_bank_lock = threading.Lock()


# ─────────────────────────────────────────────
# Fact bank storage
# ─────────────────────────────────────────────

def load_bank(path: Path = FACT_BANK_FILE) -> Dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": 1, "persona_name": default_persona_name(), "facts": {}, "scenarios": {}}


def save_bank(bank: Dict, path: Path = FACT_BANK_FILE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _referenced_ids(bank: Dict) -> List[str]:
    """Catalog ids used by at least one scenario, in catalog order."""
    used = {f["id"] for s in bank["scenarios"].values() for f in s["facts"]}
    return [fid for fid in bank["facts"] if fid in used]


def verified_facts(bank: Dict, scenario_id: str) -> List[Dict]:
    """Human-verified facts for one scenario, ready for the judge prompt."""
    out = []
    for ref in bank["scenarios"].get(scenario_id, {}).get("facts", []):
        fact = bank["facts"].get(ref["id"])
        review = (fact or {}).get("review") or {}
        if review.get("status") in VERIFIED:
            out.append({"id": fact["id"], "kind": fact["kind"],
                        "question": fact["question"], "answer": review["answer"]})
    return out


def _load_scenarios(ids: Optional[List[str]] = None, category: Optional[str] = None) -> List[Dict]:
    scenarios = json.loads(SCENARIOS_FILE.read_text(encoding="utf-8"))
    if ids:
        scenarios = [s for s in scenarios if s["id"] in ids]
    if category:
        scenarios = [s for s in scenarios if s["category"] == category]
    return scenarios


def _client(task: str):
    from IMPersona2.eval import AIPartnerClient
    return AIPartnerClient(
        model=model_for(task),
        base_url=model_for(task, "base_url"),
        reasoning=model_for(task, "reasoning"),
    )


def _norm(text: str) -> str:
    """Normalize for quote matching: typography, <msg> separators, spacing, case."""
    text = (text or "").replace("<msg>", " ")
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "…": "..."}))
    return re.sub(r"\s+", " ", text).strip().lower()


# ─────────────────────────────────────────────
# Stage 1 — propose facts per scenario
# ─────────────────────────────────────────────

PROPOSE_SYSTEM = (
    "You design evaluations for a model that simulates one specific person. "
    "You never guess facts about the person; you only ask questions."
)

PROPOSE_PROMPT = """\
A model has been trained on a real person's text messages to simulate them.
The person is {persona_name}. The model will be placed in the scenario below,
chatting with an AI assistant. We want to test whether it recalls and acts on
true facts about {persona_name}'s life, instead of inventing generic details.

SCENARIO: {name}
{description}
Context given to the simulated {persona_name}: {context}

Choose about {n} facts about {persona_name} that the REAL {persona_name} would likely
draw on in this conversation. Each fact is a question that will be answered
later by searching their messages. Do NOT guess answers.

FACT CATALOG — questions already chosen for other scenarios:
{catalog}

Reuse catalog entries by id whenever the same fact applies, even if you would
phrase it differently. Only add a new question for a fact not already in the
catalog.

Good questions:
- Are specific to this person: their job or school, where they live, the
  people in their life, ongoing plans, past events, habits, tastes, and how
  they tend to make decisions.
- Would visibly change what they say in THIS scenario if the model knew the
  answer.
- Are plausibly answerable from someone's personal text messages.
- Are self-contained: they make sense without seeing this scenario.
- Cover a mix of kinds:
    "fact"       — biographical or situational ("What city does {persona_name} live in?")
    "preference" — tastes and dislikes ("What food does {persona_name} usually want?")
    "decision"   — a choice they made, or how they tend to choose
                   ("Has {persona_name} decided whether to apply to PhD programs?")

Bad questions: generic ones that apply to anyone, questions only answerable
by guessing, and near-duplicates of each other or of the catalog.

Respond with ONLY a JSON object:
{{"reuse": [{{"id": "F12", "why_relevant": "<one sentence>"}}],
  "new": [{{"question": "...", "kind": "fact|preference|decision",
           "why_relevant": "<one sentence: how it would change their messages here>"}}]}}
"""


def _validate_proposal(obj: Dict) -> None:
    if not isinstance(obj.get("reuse", []), list) or not isinstance(obj.get("new", []), list):
        raise ValueError("'reuse' and 'new' must be lists")
    if not obj.get("reuse") and not obj.get("new"):
        raise ValueError("no facts proposed")


def _format_catalog(bank: Dict) -> str:
    if not bank["facts"]:
        return "(empty — this is the first scenario)"
    return "\n".join(f"[{f['id']}] ({f['kind']}) {f['question']}" for f in bank["facts"].values())


def _next_fact_id(bank: Dict) -> str:
    nums = [int(fid[1:]) for fid in bank["facts"] if fid[1:].isdigit()]
    return f"F{max(nums, default=0) + 1}"


def _apply_proposal(bank: Dict, scenario: Dict, obj: Dict, n: int) -> List[Dict]:
    """Merge one scenario's proposal into the catalog. Returns its fact refs."""
    by_question = {_norm(f["question"]): fid for fid, f in bank["facts"].items()}
    refs: List[Dict] = []
    chosen = set()

    def add(fid: str, why: str) -> None:
        if fid not in chosen and len(refs) < n:
            chosen.add(fid)
            refs.append({"id": fid, "why_relevant": why})

    for item in obj.get("reuse", []):
        if not isinstance(item, dict):
            item = {"id": item}
        fid = str(item.get("id", "")).strip().upper()
        if fid in bank["facts"]:
            add(fid, str(item.get("why_relevant", "")))

    for item in obj.get("new", []):
        if not isinstance(item, dict):
            item = {"question": item}
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        why = str(item.get("why_relevant", ""))
        existing = by_question.get(_norm(question))
        if existing:
            add(existing, why)
            continue
        if len(refs) >= n:
            break
        fid = _next_fact_id(bank)
        kind = item.get("kind") if item.get("kind") in KINDS else "fact"
        bank["facts"][fid] = {"id": fid, "question": question, "kind": kind,
                              "search": None, "review": None}
        by_question[_norm(question)] = fid
        add(fid, why)

    if refs:
        bank["scenarios"][scenario["id"]] = {"name": scenario["name"], "facts": refs}
    return refs


def propose(scenarios: List[Dict], n: int = 6, force: bool = False,
            bank_path: Path = FACT_BANK_FILE) -> None:
    """Stage 1. Runs sequentially so every scenario sees the up-to-date catalog."""
    from IMPersona2.eval import ask_json

    bank = load_bank(bank_path)
    persona = bank.get("persona_name") or default_persona_name()
    todo = [s for s in scenarios if force or s["id"] not in bank["scenarios"]]
    print(f"  Proposing facts for {len(todo)} scenario(s) with {model_for('fact_propose')} "
          f"({len(scenarios) - len(todo)} already done)")
    client = _client("fact_propose")

    for i, s in enumerate(todo, 1):
        before = len(bank["facts"])
        prompt = PROPOSE_PROMPT.format(
            persona_name=persona, name=s["name"], description=s.get("description", ""),
            context=s.get("impersona_context", ""), n=n, catalog=_format_catalog(bank),
        )
        try:
            obj = ask_json(client, PROPOSE_SYSTEM, prompt, max_tokens=4096, validate=_validate_proposal)
        except RuntimeError as e:
            print(f"    ✗ [{i}/{len(todo)}] {s['id']}: {e}")
            continue
        refs = _apply_proposal(bank, s, obj, n)
        if not refs:
            print(f"    ✗ [{i}/{len(todo)}] {s['id']}: no usable facts in the reply; will retry next run")
            continue
        save_bank(bank, bank_path)
        print(f"    ✓ [{i}/{len(todo)}] {s['id']}: {len(refs)} facts "
              f"({len(bank['facts']) - before} new; catalog: {len(bank['facts'])})")


# ─────────────────────────────────────────────
# Stage 2 — search agent over the message history
# ─────────────────────────────────────────────

SEARCH_SYSTEM = (
    "You are a careful research assistant. You answer a question about a person "
    "using ONLY excerpts from their own text messages. You never guess, and you "
    "quote evidence verbatim."
)

SEARCH_PROMPT = """\
QUESTION about {persona_name}: {question}

You are answering this by searching {persona_name}'s personal text-message history
with a semantic search tool. In the excerpts, "Me" is {persona_name}; other speakers
are the people they were texting. "<msg>" separates messages sent back-to-back.
Each excerpt's header gives its date and contact.

Queries tried so far: {queries}

Excerpts you are keeping:
{excerpts}

Search round {round} of {max_rounds}.{note}

Choose ONE action:
- "search": run 1–3 NEW queries, different from those already tried. Describe
  the situation in natural language (e.g. "{persona_name} talking about starting a
  new job") rather than listing keywords. Put the ids of the excerpts that are
  relevant in "keep"; the rest are dropped.
- "answer": answer ONLY from the excerpts, supporting every claim with a quote
  copied verbatim from an excerpt. Write one or two sentences, as in a fact
  sheet. If the answer changed over time, give the most recent state and its
  date.
- "not_found": the excerpts don't answer the question and more searching is
  unlikely to help.

Respond with ONLY a JSON object, in one of these forms:
{{"action": "search", "queries": ["..."], "keep": ["E2"]}}
{{"action": "answer", "answer": "...", "confidence": "high|medium|low",
  "evidence": [{{"excerpt": "E2", "quote": "<verbatim text>"}}]}}
{{"action": "not_found", "reason": "..."}}
"""


def _excerpt_id(value) -> str:
    eid = str(value).strip().upper()
    return f"E{eid}" if eid.isdigit() else eid


def _validate_search_step(obj: Dict) -> None:
    action = obj.get("action")
    if action == "search":
        queries = obj.get("queries")
        if not isinstance(queries, list) or not any(isinstance(q, str) and q.strip() for q in queries):
            raise ValueError("search action needs a non-empty 'queries' list")
    elif action == "answer":
        if not str(obj.get("answer", "")).strip() or not isinstance(obj.get("evidence", []), list):
            raise ValueError("answer action needs 'answer' and an 'evidence' list")
    elif action != "not_found":
        raise ValueError(f"unknown action {action!r}")


def _format_excerpts(pool: Dict[str, Dict]) -> str:
    if not pool:
        return "(none yet)"
    return "\n\n".join(
        f"[{eid} · {r['date_start']}..{r['date_end']} · {r['contact']}]\n{r['text']}"
        for eid, r in pool.items()
    )


def search_fact(fact: Dict, store, client, persona_name: str,
                max_rounds: int = 3, k: int = 4) -> Dict:
    """Run the search agent for one catalog fact. Returns its `search` record."""
    from IMPersona2.eval import ask_json

    max_rounds = max(2, max_rounds)  # round 1 must be free to search
    queries: List[str] = []
    pool: Dict[str, Dict] = {}
    seen = set()
    next_id = 0
    record = {"model": model_for("fact_search"), "searched_at": datetime.now().isoformat(timespec="seconds")}

    def not_found(reason: str) -> Dict:
        return {**record, "status": "not_found", "reason": reason, "queries": queries}

    for rnd in range(1, max_rounds + 1):
        final = rnd == max_rounds
        if not pool and rnd == 1:
            note = " Nothing has been retrieved yet, so you must search."
        elif final:
            note = " This is the FINAL round: choose answer or not_found."
        else:
            note = ""
        prompt = SEARCH_PROMPT.format(
            persona_name=persona_name, question=fact["question"],
            queries=json.dumps(queries, ensure_ascii=False) if queries else "(none)",
            excerpts=_format_excerpts(pool), round=rnd, max_rounds=max_rounds, note=note,
        )
        step = ask_json(client, SEARCH_SYSTEM, prompt, max_tokens=4096, validate=_validate_search_step)
        action = step["action"]

        if action == "not_found":
            return not_found(str(step.get("reason", "")))

        if action == "answer":
            if not pool:
                if final:
                    return not_found("answered without retrieving any evidence")
                continue
            evidence = []
            for ev in step.get("evidence", []):
                if not isinstance(ev, dict):
                    continue
                eid = _excerpt_id(ev.get("excerpt", ""))
                src = pool.get(eid)
                quote = str(ev.get("quote", "")).strip()
                evidence.append({
                    "excerpt": eid,
                    "quote": quote,
                    "contact": src["contact"] if src else None,
                    "date": src["date_start"] if src else None,
                    # Catches paraphrased or invented quotes.
                    "quote_verified": bool(src and quote and _norm(quote) in _norm(src["text"])),
                })
            return {**record, "status": "found", "answer": str(step["answer"]).strip(),
                    "confidence": step.get("confidence", "low"), "evidence": evidence,
                    "all_quotes_verified": bool(evidence) and all(e["quote_verified"] for e in evidence),
                    "queries": queries}

        # action == "search"
        if final:
            return not_found("ran out of search rounds")
        keep = step.get("keep")
        if isinstance(keep, list):
            keep_ids = {_excerpt_id(x) for x in keep}
            pool = {eid: r for eid, r in pool.items() if eid in keep_ids}
        tried = {q.lower() for q in queries}
        added = 0
        for q in step["queries"]:
            if added == 3:
                break
            if not isinstance(q, str) or not q.strip() or q.strip().lower() in tried:
                continue
            q = q.strip()
            queries.append(q)
            tried.add(q.lower())
            added += 1
            for r in store.search(q, k=k):
                key = (r["contact"], r["date_start"], r["text"][:200])
                if key in seen:
                    continue
                seen.add(key)
                next_id += 1
                pool[f"E{next_id}"] = r

    return not_found("ran out of search rounds")


def search(scenario_ids: Optional[List[str]] = None, force: bool = False, max_rounds: int = 3,
           max_workers: int = 8, bank_path: Path = FACT_BANK_FILE) -> None:
    """Stage 2. Searches every catalog fact used by a scenario (optionally filtered)."""
    from IMPersona2.memory_search import MemoryStore

    bank = load_bank(bank_path)
    if not bank["facts"]:
        print("  No facts yet — run `python -m IMPersona2.fact_recall propose` first.")
        return
    try:
        store = MemoryStore(MEMORY_DIR / "conversations.json", MEMORY_DIR)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  Memory index unavailable: {e}\n  Build it with `python run.py --step memory`.")
        return

    ids = _referenced_ids(bank)
    if scenario_ids:
        wanted = {f["id"] for sid in scenario_ids for f in bank["scenarios"].get(sid, {}).get("facts", [])}
        ids = [fid for fid in ids if fid in wanted]
    todo = [bank["facts"][fid] for fid in ids
            if force or not bank["facts"][fid].get("search") or bank["facts"][fid]["search"]["status"] == "error"]

    persona = bank.get("persona_name") or default_persona_name()
    client = _client("fact_search")
    print(f"  Searching {len(todo)} fact(s) with {model_for('fact_search')} "
          f"(up to {max_rounds} rounds each)")

    def work(fact):
        try:
            return fact, search_fact(fact, store, client, persona, max_rounds=max_rounds)
        except Exception as e:
            return fact, {"status": "error", "reason": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for fut in as_completed([pool.submit(work, f) for f in todo]):
            fact, rec = fut.result()
            with _bank_lock:
                fact["search"] = rec
                fact["review"] = None  # a new search result needs a fresh review
                save_bank(bank, bank_path)
            mark = {"found": "✓", "not_found": "–", "error": "✗"}[rec["status"]]
            detail = rec.get("answer") or rec.get("reason", "")
            warn = "" if rec["status"] != "found" or rec["all_quotes_verified"] else "  (unverified quote)"
            print(f"    {mark} {fact['id']} {fact['question'][:60]} → {detail[:80]}{warn}")


# ─────────────────────────────────────────────
# Stage 3 — human review
# ─────────────────────────────────────────────

def _choose(allowed: str) -> Optional[str]:
    """Read one of the allowed single-letter choices; None on EOF/Ctrl-C."""
    while True:
        try:
            choice = input(f"  [{'/'.join(allowed)}] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if choice and choice in allowed:
            return choice


def _ask_answer(default: str = "") -> Optional[str]:
    """Read a corrected answer. Empty input keeps `default`; None if still empty."""
    try:
        text = input("  Correct answer (blank = keep shown answer): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return text or default or None


def review(include_not_found: bool = False, bank_path: Path = FACT_BANK_FILE) -> None:
    bank = load_bank(bank_path)
    used_by: Dict[str, List[str]] = {}
    for s in bank["scenarios"].values():
        for ref in s["facts"]:
            used_by.setdefault(ref["id"], []).append(s["name"])

    pending = []
    for fid in _referenced_ids(bank):
        fact = bank["facts"][fid]
        status = (fact.get("search") or {}).get("status")
        if (fact.get("review") or {}).get("status"):
            continue
        if status == "found" or (include_not_found and status == "not_found"):
            pending.append(fact)

    print(f"\n  {len(pending)} fact(s) to review. Only accepted or edited facts are used for scoring.")
    print("  a = accept · e = edit the answer · r = reject · s = skip for now · q = quit (progress is saved)\n")

    for n, fact in enumerate(pending, 1):
        s = fact["search"]
        names = used_by.get(fact["id"], [])
        print("─" * 72)
        print(f"  [{n}/{len(pending)}] {fact['id']} · {fact['kind']} · used by {len(names)} scenario(s): "
              f"{', '.join(names[:3])}{' …' if len(names) > 3 else ''}")
        print(f"  Q: {fact['question']}")
        if s["status"] == "found":
            print(f"  A: {s['answer']}   (confidence: {s.get('confidence')})")
            for ev in s.get("evidence", []):
                flag = "" if ev.get("quote_verified") else "   ⚠ not found verbatim in the excerpt"
                quote = " / ".join(line.strip() for line in ev["quote"].splitlines() if line.strip())
                print(f"     “{quote}” — {ev.get('contact')}, {ev.get('date')}{flag}")
            allowed = "aersq"
        else:
            print(f"  Search found nothing ({s.get('reason', '')}). Queries: {', '.join(s.get('queries', []))}")
            allowed = "ersq"

        choice = _choose(allowed)
        if choice in (None, "q"):
            break
        if choice == "s":
            continue
        if choice == "e":
            answer = _ask_answer(s.get("answer", "") if s["status"] == "found" else "")
            if answer is None:
                print("  No answer given — skipped.")
                continue
            fact["review"] = {"status": "edited", "answer": answer}
        elif choice == "a":
            fact["review"] = {"status": "accepted", "answer": s["answer"]}
        else:
            fact["review"] = {"status": "rejected", "answer": None}
        fact["review"]["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
        save_bank(bank, bank_path)

    print_status(bank_path)


def print_status(bank_path: Path = FACT_BANK_FILE) -> None:
    bank = load_bank(bank_path)
    facts = [bank["facts"][fid] for fid in _referenced_ids(bank)]
    search_status = [(f.get("search") or {}).get("status") for f in facts]
    review_status = [(f.get("review") or {}).get("status") for f in facts]
    found = search_status.count("found")
    verified = sum(r in VERIFIED for r in review_status)
    rejected = review_status.count("rejected")
    with_facts = sum(1 for sid in bank["scenarios"] if verified_facts(bank, sid))
    print(f"\n  Fact bank: {bank_path}")
    print(f"    scenarios: {len(bank['scenarios'])} ({with_facts} with verified facts) · unique facts: {len(facts)}")
    print(f"    search: {found} found · {search_status.count('not_found')} not found · "
          f"{search_status.count('error')} errors · {search_status.count(None)} not searched")
    print(f"    review: {verified} verified · {rejected} rejected · "
          f"{sum(1 for f, r in zip(search_status, review_status) if f == 'found' and not r)} found but unreviewed")


# ─────────────────────────────────────────────
# Stage 4 — judge eval conversations
# ─────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert evaluator. You check whether a simulated person's "
    "conversation is consistent with verified facts about the real person. "
    "Be precise and cite specific messages."
)

JUDGE_PROMPT = """\
A model simulating a real person, {persona_name}, chatted with an AI assistant in
the scenario below. You have VERIFIED FACTS about the real {persona_name}. Check
whether the simulation recalls and acts on them.

SCENARIO: {scenario_context}

VERIFIED FACTS (ground truth):
{facts}

CONVERSATION. {persona_name}'s private thoughts appear as [thought: ...]. Recalling
a fact in a thought counts, as long as it is consistent with the truth.
{transcript}

Label EACH fact with exactly one of:
- "used_correctly"  The fact comes up in a message or thought and is consistent
                    with the truth, or {persona_name}'s choices and stated
                    preferences clearly match it.
- "contradicted"    {persona_name} says, thinks, or does something inconsistent
                    with it: a wrong job, city, or person, the opposite
                    preference, or a decision that conflicts with it.
- "missed"          The conversation reached a point where the real {persona_name}
                    would naturally have drawn on this fact, but the simulation
                    stayed vague or filled in generic or invented details.
- "not_applicable"  The conversation never went anywhere the fact would matter.

Be strict about "contradicted": invented details that conflict with a verified
fact count, even if mentioned casually. Mark "missed" only when the absence is
noticeable at a specific moment, not merely because the fact never came up.

Respond with ONLY a JSON object:
{{"labels": [{{"fact_id": "F1", "label": "used_correctly|contradicted|missed|not_applicable",
              "evidence": "<message or thought it hinges on, or empty>",
              "reason": "<one sentence>"}}]}}
Return exactly one label for each of the {num_facts} facts.
"""

PROMPT_SHA = hashlib.sha256((JUDGE_SYSTEM + JUDGE_PROMPT).encode()).hexdigest()[:10]


def format_facts(facts: List[Dict]) -> str:
    return "\n".join(f"- [{f['id']}] ({f['kind']}) {f['question']} → {f['answer']}" for f in facts)


def format_transcript(conversation: Dict, persona_name: str, ai_chars: int = 600) -> str:
    """AI replies are shown as the user model saw them (partner <think> stripped)."""
    from IMPersona2.eval import strip_think_tokens

    lines = []
    for i, msg in enumerate(conversation["messages"]):
        content = msg.get("content") or ""
        if msg["speaker"] in (persona_name, "Me"):
            lines.append(f"[{i}] {persona_name}: {content}")
            if msg.get("chain_of_thought"):
                lines.append(f"     [thought: {msg['chain_of_thought']}]")
        else:
            content = " ".join(strip_think_tokens(content).split())
            lines.append(f"[{i}] AI: {content if len(content) <= ai_chars else content[:ai_chars] + ' …'}")
    return "\n".join(lines)


def build_judge_prompt(conversation: Dict, facts: List[Dict], persona_name: str) -> str:
    scenario = conversation["scenario"]
    return JUDGE_PROMPT.format(
        persona_name=persona_name,
        scenario_context=scenario.get("impersona_context") or scenario.get("description", ""),
        facts=format_facts(facts),
        transcript=format_transcript(conversation, persona_name),
        num_facts=len(facts),
    )


def _judge_single(conversation: Dict, facts: List[Dict], client, persona_name: str) -> Dict:
    from IMPersona2.eval import ask_json

    base = {"scenario_id": conversation["scenario"]["id"], "run_index": conversation.get("run_index")}
    wanted = {f["id"] for f in facts}

    def validate(obj: Dict) -> None:
        labels = {}
        for item in obj.get("labels", []):
            fid = str(item.get("fact_id", "")).strip().upper()
            if fid in wanted and item.get("label") in LABELS and fid not in labels:
                labels[fid] = {"fact_id": fid, "label": item["label"],
                               "evidence": str(item.get("evidence", "")), "reason": str(item.get("reason", ""))}
        missing = wanted - labels.keys()
        if missing:
            raise ValueError(f"missing labels for {sorted(missing)}")
        obj["labels"] = [labels[f["id"]] for f in facts]

    try:
        obj = ask_json(client, JUDGE_SYSTEM, build_judge_prompt(conversation, facts, persona_name),
                       max_tokens=8192, validate=validate)
    except RuntimeError as e:
        return {**base, "labels": [], "error": str(e)}
    return {**base, "labels": obj["labels"]}


def _rates(counts: Dict[str, int]) -> Dict:
    applicable = counts["used_correctly"] + counts["contradicted"] + counts["missed"]
    total = applicable + counts["not_applicable"]
    pct = lambda x, d: round(100 * x / d, 1) if d else None  # noqa: E731
    return {
        "counts": counts,
        "num_labels": total,
        "num_applicable": applicable,
        # Share of applicable facts the simulation got right — the headline number.
        "accuracy_pct": pct(counts["used_correctly"], applicable),
        "contradiction_pct": pct(counts["contradicted"], applicable),
        "applicable_pct": pct(applicable, total),
    }


def evaluate_fact_recall(conversations: List[Dict], persona_name: str,
                         bank_path: Path = FACT_BANK_FILE, max_workers: int = 20) -> Dict:
    """Score every conversation against the verified fact bank."""
    bank = load_bank(bank_path)
    jobs = [(c, verified_facts(bank, c["scenario"]["id"])) for c in conversations]
    jobs = [(c, facts) for c, facts in jobs if facts]
    if not jobs:
        raise ValueError(
            f"No verified facts in {bank_path} for these scenarios. Build the bank first: "
            f"python -m IMPersona2.fact_recall propose / search / review"
        )

    model = model_for("fact_judge")
    client = _client("fact_judge")
    print(f"\n  Fact-recall judge ({model}, prompt {PROMPT_SHA}): {len(jobs)} of "
          f"{len(conversations)} conversations have verified facts")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_judge_single, c, facts, client, persona_name) for c, facts in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            if r.get("error"):
                print(f"    ✗ {r['scenario_id']}: {r['error']}")
            else:
                c = {k: sum(1 for l in r["labels"] if l["label"] == k) for k in LABELS}
                print(f"    ✓ {r['scenario_id']}: " + " ".join(f"{k}={v}" for k, v in c.items() if v))
            results.append(r)

    ok = [r for r in results if not r.get("error")]
    kind_of = {fid: f["kind"] for fid, f in bank["facts"].items()}
    totals = {k: 0 for k in LABELS}
    by_kind: Dict[str, Dict[str, int]] = {}
    for r in ok:
        for l in r["labels"]:
            totals[l["label"]] += 1
            by_kind.setdefault(kind_of.get(l["fact_id"], "fact"), {k: 0 for k in LABELS})[l["label"]] += 1

    summary = {
        "judge_model": model,
        "prompt_sha": PROMPT_SHA,
        "fact_bank": str(bank_path),
        "num_conversations": len(conversations),
        "num_evaluated": len(ok),
        "num_failed": len(results) - len(ok),
        "num_without_verified_facts": len(conversations) - len(jobs),
        **_rates(totals),
        "by_kind": {k: _rates(v) for k, v in sorted(by_kind.items())},
        "per_conversation": sorted(results, key=lambda r: (r["scenario_id"], r.get("run_index") or 0)),
    }
    print(f"  Fact recall: {summary['accuracy_pct']}% used correctly · "
          f"{summary['contradiction_pct']}% contradicted "
          f"(of {summary['num_applicable']} applicable labels)")
    return summary


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build and inspect the fact-recall fact bank.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("propose", help="Stage 1: choose facts for each eval scenario")
    p.add_argument("--scenarios", type=str, default="", help="Comma-separated scenario ids (default: all)")
    p.add_argument("--category", type=str, default=None, help="Only scenarios in this category")
    p.add_argument("--facts-per-scenario", type=int, default=6)
    p.add_argument("--force", action="store_true", help="Redo scenarios that already have facts")

    s = sub.add_parser("search", help="Stage 2: search your message history for each fact")
    s.add_argument("--scenarios", type=str, default="", help="Only facts used by these scenario ids")
    s.add_argument("--max-rounds", type=int, default=3)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--force", action="store_true", help="Re-search facts that were already searched")

    r = sub.add_parser("review", help="Stage 3: verify facts interactively")
    r.add_argument("--include-not-found", action="store_true",
                   help="Also show facts the search couldn't find, so you can write the answer yourself")

    sub.add_parser("status", help="Show fact bank progress")

    args = parser.parse_args()
    ids = [x.strip() for x in getattr(args, "scenarios", "").split(",") if x.strip()] or None

    if args.cmd == "propose":
        propose(_load_scenarios(ids, args.category), n=args.facts_per_scenario, force=args.force)
        print_status()
    elif args.cmd == "search":
        search(ids, force=args.force, max_rounds=args.max_rounds, max_workers=args.workers)
        print_status()
    elif args.cmd == "review":
        review(include_not_found=args.include_not_found)
    else:
        print_status()


if __name__ == "__main__":
    main()
