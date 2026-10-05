#!/usr/bin/env python3
"""
Post-process a DPO dataset to filter out bad training examples.

Uses an LLM judge (`dpo_filter` in configs/models.yaml) to evaluate
each DPO datapoint holistically and filter out low-quality examples.

Filter criteria:
  1. **Dead conversation** — the conversation has gone stale, the IMPersona
     has run out of things to talk about, or the natural conclusion was
     already reached.
  2. **Nonsensical / weird messages** — the assistant or IMPersona produced
     incoherent, hallucinated, or clearly broken output (e.g. repeated
     text, gibberish, hallucinated events that didn't happen).

The judge sees the FULL DPO datapoint: conversation context, the user's
input, the persona's internal reaction (CoT), the persona's visible
reply, the chosen/rejected pair, and the observable guideline.

Usage:
    # Filter a single DPO dataset (judge: dpo_filter in configs/models.yaml):
    python -m IMPersona2.filter_dpo_dataset \
        --input eval_output/run_.../dpo_dataset.json

    # Dry run — show classifications without writing:
    python -m IMPersona2.filter_dpo_dataset \
        --input eval_output/run_.../dpo_dataset.json \
        --dry-run

    # Use a different judge model:
    python -m IMPersona2.filter_dpo_dataset \
        --input eval_output/run_.../dpo_dataset.json \
        --judge-model <model-id>

    # Custom output path:
    python -m IMPersona2.filter_dpo_dataset \
        --input eval_output/run_.../dpo_dataset.json \
        --output eval_output/run_.../dpo_dataset_filtered.json
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from IMPersona2.eval import AIPartnerClient, C
from IMPersona2.model_config import model_for


# ─────────────────────────────────────────────
# DPO Quality Judge
# ─────────────────────────────────────────────

FILTER_SYSTEM_PROMPT = """\
You are an expert data-quality judge for DPO (Direct Preference \
Optimization) training datasets.

You will be given a COMPLETE DPO datapoint from a conversation between \
a user (the "IMPersona" — a simulated person) and an AI assistant. The \
conversations were generated in a simulation where the IMPersona and AI \
assistant chat for up to 50 turns.

Your job: determine whether this datapoint should be **KEPT** for \
training or **REMOVED** as low-quality.

REMOVE the datapoint if ANY of the following apply:

**REASON 1 — Dead / stale conversation:**
The conversation has gone dead at the point of this datapoint. Signs:
- The IMPersona has clearly run out of meaningful things to say and is \
filling time (very short filler like "yea", "ok", "true" with no \
substance, random topic-hopping with no depth, gibberish).
- The natural conclusion was already reached (goodbyes were exchanged, \
e.g. "take care", "sounds good", "bye") and the conversation \
continues artificially with off-topic or incoherent messages.
- The conversation is going in circles with no new information.
IMPORTANT: Brief messages like "yea" or "ok" are NOT automatically \
dead — look at what the persona says NEXT (their visible reply) and \
their internal chain-of-thought. If the persona follows up with \
something substantive, a correction, or emotional engagement, the \
conversation is still alive even if the input message itself is brief. \
HOWEVER, if goodbyes were clearly exchanged and the next messages are \
off-topic gibberish or random topic-hopping (e.g. suddenly asking \
about an unrelated game), that IS dead regardless of what the persona \
says next.

**REASON 2 — Nonsensical / broken content:**
The assistant, IMPersona, or the rewrite produced clearly broken, \
incoherent, or hallucinated output. Signs:
- CRITICAL: The chosen (rewritten) response references events, \
problems, or situations that NEVER ACTUALLY HAPPENED in the \
conversation context above. For example, if the chosen response \
apologizes for "glitching" or "repeating text" but the actual \
assistant messages in the conversation are normal, that is a \
hallucination and the datapoint must be removed. Cross-check the \
chosen response's claims against what actually appears in the \
CONVERSATION CONTEXT section.
- The assistant's response contains repeated text, gibberish, or is \
clearly malformed.
- The IMPersona's messages are incoherent or clearly off-topic in a \
way that makes the DPO pair meaningless.
- The chosen and rejected responses are essentially identical (no \
meaningful improvement was made).

KEEP the datapoint if:
- The conversation is still meaningfully engaged (even if brief).
- The chosen response is a genuine improvement over the rejected one.
- The persona's chain-of-thought shows real engagement or a valid \
concern about the assistant's response.
- The datapoint captures a real conversational failure worth training on.

Output EXACTLY one of these formats:
KEEP: <brief reason>
REMOVE-DEAD: <brief reason why conversation is dead/stale>
REMOVE-NONSENSE: <brief reason why content is broken/hallucinated>

Be CONSERVATIVE — only remove when there is clear evidence of a \
quality problem. When in doubt, KEEP."""

FILTER_USER_TEMPLATE = """\
SCENARIO: {scenario_name}
TURN INDEX: {turn_index} (out of up to 50)

═══ CONVERSATION CONTEXT ═══
{conversation_text}

═══ LAST USER MESSAGE (input to the assistant) ═══
{input_text}

═══ PERSONA'S INTERNAL REACTION (private chain-of-thought) ═══
{persona_cot}

═══ WHAT THE PERSONA ACTUALLY REPLIED ═══
{persona_reply}

═══ DISAGREEMENT JUDGE REASON ═══
{judge_reason}

═══ OBSERVABLE GUIDELINE (given to rewrite model) ═══
{guideline}

═══ CHOSEN RESPONSE (rewrite — should be better) ═══
{chosen}

═══ REJECTED RESPONSE (original — should be worse) ═══
{rejected}

Should this DPO datapoint be KEPT or REMOVED?"""


def format_conversation_for_judge(
    chat_messages_before: List[Dict[str, str]],
    strip_think: bool = True,
) -> str:
    """Format the conversation history into a readable block for the judge."""
    lines = []
    for msg in chat_messages_before:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Strip <think>...</think> blocks from assistant messages
        if strip_think and role == "assistant":
            content = re.sub(
                r"<think>.*?</think>", "", content, flags=re.DOTALL
            ).strip()

        label = "USER" if role == "user" else "AI ASSISTANT"
        lines.append(f"[{label}]: {content}")

    return "\n\n".join(lines)


def classify_datapoint(
    judge: AIPartnerClient,
    example: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Use the LLM judge to evaluate a full DPO datapoint.

    Returns (should_remove, reason, removal_type).
    removal_type is one of: "", "dead", "nonsense".
    """
    chat_before = example.get("_chat_messages_before", [])
    input_text = example.get("input", "")
    scenario_name = example.get("_scenario_name", "Unknown")
    turn_index = example.get("_turn_index", "?")

    conversation_text = format_conversation_for_judge(chat_before)

    user_prompt = FILTER_USER_TEMPLATE.format(
        scenario_name=scenario_name,
        turn_index=turn_index,
        conversation_text=conversation_text or "(no prior context)",
        input_text=input_text,
        persona_cot=example.get("_persona_chain_of_thought", "(not available)"),
        persona_reply=example.get("_persona_visible_response", "(not available)"),
        judge_reason=example.get("_judge_reason", "(not available)"),
        guideline=example.get("_observable_guideline", "(not available)"),
        chosen=example.get("chosen", ""),
        rejected=example.get("rejected", ""),
    )

    response = judge.generate(
        system_prompt=FILTER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=1024,
    )

    response = response.strip()

    # Parse the response
    first_line = response.split("\n")[0].strip()
    upper = first_line.upper()

    if upper.startswith("REMOVE-DEAD:"):
        reason = first_line[len("REMOVE-DEAD:"):].strip()
        return (True, reason, "dead")
    elif upper.startswith("REMOVE-NONSENSE:"):
        reason = first_line[len("REMOVE-NONSENSE:"):].strip()
        return (True, reason, "nonsense")
    elif upper.startswith("REMOVE:"):
        reason = first_line[len("REMOVE:"):].strip()
        return (True, reason, "other")
    elif upper.startswith("KEEP:"):
        reason = first_line[len("KEEP:"):].strip()
        return (False, reason, "")
    else:
        # Fallback heuristic
        if "REMOVE" in upper:
            return (True, response, "other")
        return (False, f"(unparsed) {response}", "")


def run_quality_filter(
    examples: List[Dict],
    judge: AIPartnerClient,
    max_workers: int = 8,
    verbose: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict[str, Any]]]:
    """
    Filter examples using the comprehensive quality judge.

    Returns (kept_examples, removed_examples, classifications).
    """
    classifications: List[Dict[str, Any]] = []
    kept: List[Dict] = []
    removed: List[Dict] = []

    def _classify(idx: int, example: Dict) -> Tuple[int, bool, str, str]:
        should_remove, reason, removal_type = classify_datapoint(judge, example)
        return (idx, should_remove, reason, removal_type)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_classify, i, ex): i
            for i, ex in enumerate(examples)
        }

        for future in as_completed(futures):
            idx, should_remove, reason, removal_type = future.result()
            ex = examples[idx]

            scenario = ex.get("_scenario_name", "?")
            turn = ex.get("_turn_index", "?")

            classifications.append({
                "index": idx,
                "scenario": scenario,
                "turn_index": turn,
                "should_remove": should_remove,
                "removal_type": removal_type,
                "reason": reason,
            })

            if should_remove:
                removed.append(ex)
                type_label = removal_type.upper() if removal_type else "REMOVE"
                if verbose:
                    print(
                        f"  {C.RED}✗ {type_label}{C.RESET} [{idx}] {scenario} "
                        f"turn {turn}: {reason[:80]}"
                    )
            else:
                kept.append(ex)
                if verbose:
                    print(
                        f"  {C.GREEN}✓ KEEP{C.RESET} [{idx}] {scenario} "
                        f"turn {turn}: {reason[:80]}"
                    )

    # Sort by original index to maintain order
    kept.sort(key=lambda x: examples.index(x))
    removed.sort(key=lambda x: examples.index(x))
    classifications.sort(key=lambda x: x["index"])

    return kept, removed, classifications


def generate_filter_viewer_html(
    all_examples: List[Dict],
    classifications: List[Dict[str, Any]],
    input_file: str,
    judge_model: str,
    judge_reasoning: Optional[str],
    run_stats: Optional[Dict[str, Dict]] = None,
) -> str:
    """Generate a self-contained HTML viewer showing kept vs filtered DPO examples.

    Args:
        run_stats: Optional dict mapping scenario_id → {"runs": int, "dpo": int, "name": str}
                   with total conversation counts per scenario (for yield chart).
    """
    import html as html_mod
    from collections import Counter

    # Build a lookup: index → classification
    cls_by_idx = {c["index"]: c for c in classifications}

    # Annotate each example with its classification
    annotated = []
    for i, ex in enumerate(all_examples):
        cls = cls_by_idx.get(i, {})
        should_remove = cls.get("should_remove", cls.get("is_dead", False))
        removal_type = cls.get("removal_type", "dead" if should_remove else "")
        annotated.append({
            **ex,
            "_filter_status": "REMOVED" if should_remove else "KEPT",
            "_filter_removal_type": removal_type,
            "_filter_reason": cls.get("reason", ""),
            "_filter_index": i,
        })

    scenarios = Counter(ex.get("_scenario_name", "unknown") for ex in all_examples)
    total = len(all_examples)
    kept = sum(1 for c in classifications if not c.get("is_dead"))
    removed = total - kept

    run_label = (all_examples[0].get("_run_label") or "unknown") if all_examples else "unknown"
    run_id = (all_examples[0].get("_run_id") or "") if all_examples else ""

    annotated_json = json.dumps(annotated, ensure_ascii=False)
    # Escape </script> inside JSON data so it doesn't break the HTML <script> block.
    # The standard trick is to split the closing tag: "<\/script>".
    annotated_json = annotated_json.replace("</script>", r"<\/script>")
    annotated_json = annotated_json.replace("</style>", r"<\/style>")
    scenarios_json = json.dumps(dict(scenarios.most_common()), ensure_ascii=False)
    run_stats_json = json.dumps(run_stats or {}, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DPO Filter Viewer — {html_mod.escape(run_label)}</title>
<style>
:root {{
  --bg: #0d1117; --surface: #161b22; --surface2: #21262d;
  --border: #30363d; --text: #e6edf3; --muted: #8b949e;
  --accent: #58a6ff; --green: #3fb950; --red: #f85149;
  --purple: #bc8cff; --yellow: #d29922; --orange: #db6d28;
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
.stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
               gap:12px; margin-bottom:24px; }}
.stat-card {{ background:var(--surface); border:1px solid var(--border);
             border-radius:8px; padding:14px; }}
.stat-card .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.5px; }}
.stat-card .value {{ font-size:22px; font-weight:600; margin-top:2px; }}
.stat-card .value.green {{ color:var(--green); }}
.stat-card .value.red {{ color:var(--red); }}

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
  cursor:pointer; font-size:13px; transition:all .2s; user-select:none;
}}
.toggle-btn.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.toggle-btn.green.active {{ background:var(--green); border-color:var(--green); }}
.toggle-btn.red.active {{ background:var(--red); border-color:var(--red); }}

/* Example cards */
.example-list {{ display:flex; flex-direction:column; gap:20px; }}
.example-card {{ background:var(--surface); border:2px solid var(--border);
                 border-radius:8px; overflow:hidden; transition: border-color .2s; }}
.example-card:hover {{ border-color:var(--accent); }}
.example-card.dead {{ border-color: var(--red); opacity:0.85; }}
.example-card.dead:hover {{ opacity:1; }}
.example-card.alive {{ border-color: var(--green); }}

.card-header {{ display:flex; justify-content:space-between; align-items:center;
                padding:12px 16px; border-bottom:1px solid var(--border); }}
.card-header.alive {{ background:#1a2a1a; }}
.card-header.dead {{ background:#2a1a1a; }}
.card-header .idx {{ font-weight:600; }}
.card-header .idx.alive {{ color:var(--green); }}
.card-header .idx.dead {{ color:var(--red); }}
.card-header .scenario {{ color:var(--muted); font-size:13px; }}

/* Status badge */
.status-badge {{ display:inline-block; padding:2px 10px; border-radius:12px;
                 font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.5px; }}
.status-badge.alive {{ background:#1a2a1a; color:var(--green); border:1px solid #2d4a2d; }}
.status-badge.dead {{ background:#2a1a1a; color:var(--red); border:1px solid #4a2d2d; }}

.card-body {{ padding:16px; }}
.section-label {{ font-size:11px; text-transform:uppercase; letter-spacing:.8px;
                  color:var(--muted); margin-bottom:6px; margin-top:16px; }}
.section-label:first-child {{ margin-top:0; }}

/* Filter reason */
.filter-reason {{ padding:10px 14px; border-radius:6px; font-size:13px;
                   white-space:pre-wrap; word-break:break-word; }}
.filter-reason.alive {{ background:#1a2a1a; border:1px solid #2d4a2d; color:var(--green);
                         border-left:3px solid var(--green); }}
.filter-reason.dead {{ background:#2a1a1a; border:1px solid #4a2d2d; color:var(--red);
                        border-left:3px solid var(--red); }}

/* Conversation context */
.context-box {{ background:var(--bg); padding:12px 14px; border-radius:6px;
                font-size:13px; white-space:pre-wrap; word-break:break-word;
                max-height:300px; overflow-y:auto; line-height:1.5; }}
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

/* Scenario Distribution */
.distribution {{ background:var(--surface); border:1px solid var(--border);
                 border-radius:8px; padding:20px; margin-bottom:24px; }}
.distribution h2 {{ font-size:16px; margin-bottom:16px; color:var(--accent); }}
.dist-row {{ display:flex; align-items:center; margin-bottom:8px; gap:10px; }}
.dist-label {{ width:220px; font-size:12px; color:var(--text); text-align:right;
               white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex-shrink:0; }}
.dist-bar-bg {{ flex:1; height:22px; background:var(--surface2); border-radius:4px;
                overflow:hidden; display:flex; position:relative; }}
.dist-bar-kept {{ background:var(--green); height:100%; transition:width .3s; }}
.dist-bar-removed {{ background:var(--red); height:100%; opacity:0.7; transition:width .3s; }}
.dist-count {{ width:80px; font-size:12px; color:var(--muted); flex-shrink:0; }}
.dist-legend {{ display:flex; gap:20px; margin-top:12px; font-size:12px; color:var(--muted); }}
.dist-legend-item {{ display:flex; align-items:center; gap:6px; }}
.dist-legend-swatch {{ width:14px; height:14px; border-radius:3px; }}
.dist-tabs {{ display:flex; gap:8px; margin-bottom:16px; }}
.dist-tab {{
  background:var(--surface2); border:1px solid var(--border);
  color:var(--text); padding:5px 14px; border-radius:6px;
  cursor:pointer; font-size:12px; transition:all .2s; user-select:none;
}}
.dist-tab.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}

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
  <h1>DPO Filter <span>Viewer</span></h1>
  <div class="meta">
    {html_mod.escape(run_label)} &middot; {html_mod.escape(run_id)}
    &middot; judge: {html_mod.escape(judge_model)} ({html_mod.escape(judge_reasoning or 'default')})
  </div>
</div>

<div class="stats-grid" id="stats-grid"></div>

<div class="distribution" id="scenario-distribution">
  <h2>📊 Scenario Distribution</h2>
  <div class="dist-tabs">
    <span class="dist-tab active" onclick="setDistView('scenario')">By Scenario</span>
    <span class="dist-tab" onclick="setDistView('run')">By Run</span>
    <span class="dist-tab" onclick="setDistView('yield')">Yield (DPO / Runs)</span>
  </div>
  <div id="dist-chart"></div>
  <div id="dist-legend" class="dist-legend">
    <div class="dist-legend-item"><div class="dist-legend-swatch" style="background:var(--green)"></div> Kept</div>
    <div class="dist-legend-item"><div class="dist-legend-swatch" style="background:var(--red);opacity:0.7"></div> Removed</div>
  </div>
</div>

<div class="controls">
  <select id="filter-status">
    <option value="">All ({total})</option>
    <option value="KEPT">✓ Kept ({kept})</option>
    <option value="REMOVED">✗ Removed ({removed})</option>
  </select>
  <select id="filter-scenario"><option value="">All scenarios</option></select>
  <input type="text" id="search-input" placeholder="Search conversation, chosen, rejected, filter reason…">
  <button class="toggle-btn active" id="toggle-context" onclick="toggleContext()">Context</button>
  <button class="toggle-btn" id="toggle-think" onclick="toggleThink()">&lt;think&gt;</button>
  <button class="toggle-btn active" id="toggle-cot" onclick="toggleCot()">CoT</button>
  <button class="toggle-btn active" id="toggle-pairs" onclick="togglePairs()">Pairs</button>
  <button class="toggle-btn active" id="toggle-meta" onclick="toggleMeta()">Metadata</button>
</div>

<div class="example-list" id="example-list"></div>
<div class="pagination" id="pagination"></div>

</div>

<script>
const ALL = {annotated_json};
const SCENARIOS = {scenarios_json};
const RUN_STATS = {run_stats_json};
const PER_PAGE = 20;
let page = 0;
let showContext = true, showCot = true, showPairs = true, showMeta = true, showThink = false;

function escapeHtml(t) {{ const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }}

function stripThink(text) {{
  if (!text) return text;
  return text.replace(/<think>[\\s\\S]*?<\\/think>/g, '').trim();
}}

function fmtChat(msgs) {{
  if (!msgs || !msgs.length) return '<span style="color:var(--muted)">(no prior context)</span>';
  return msgs.map(m => {{
    const role = m.role || '?';
    const content = (role === 'assistant' && !showThink) ? stripThink(m.content) : m.content;
    if (role === 'user') {{
      return '<div style="margin-bottom:8px"><span style="color:var(--green);font-weight:600">USER:</span> ' + escapeHtml(content) + '</div>';
    }} else {{
      return '<div style="margin-bottom:8px"><span style="color:var(--accent);font-weight:600">ASSISTANT:</span> ' + escapeHtml(content) + '</div>';
    }}
  }}).join('');
}}

function filtered() {{
  let list = ALL;
  const st = document.getElementById('filter-status').value;
  if (st) list = list.filter(e => e._filter_status === st);
  const sc = document.getElementById('filter-scenario').value;
  if (sc) list = list.filter(e => e._scenario_name === sc);
  const q = document.getElementById('search-input').value.toLowerCase();
  if (q) list = list.filter(e => {{
    const blob = [
      e.chosen||'', e.rejected||'', e._persona_chain_of_thought||'',
      e._observable_guideline||'', e._judge_reason||'', e.input||'',
      e.instruction||'', e._persona_visible_response||'',
      e._filter_reason||'', e._filter_status||'',
    ].join(' ').toLowerCase();
    return blob.includes(q);
  }});
  return list;
}}

function renderStats() {{
  const g = document.getElementById('stats-grid');
  const kept = ALL.filter(e => e._filter_status === 'KEPT').length;
  const removed = ALL.filter(e => e._filter_status === 'REMOVED').length;
  const deadCount = ALL.filter(e => e._filter_removal_type === 'dead').length;
  const nonsenseCount = ALL.filter(e => e._filter_removal_type === 'nonsense').length;
  const otherCount = removed - deadCount - nonsenseCount;
  const scenarios = new Set(ALL.map(e => e._scenario_id));
  const rate = ALL.length ? Math.round(removed/ALL.length*100) : 0;
  const items = [
    {{l:'Total Examples', v:ALL.length, c:''}},
    {{l:'Kept', v:kept, c:'green'}},
    {{l:'Removed', v:removed, c:'red'}},
    {{l:'Filter Rate', v:rate+'%', c:''}},
    {{l:'Scenarios', v:scenarios.size, c:''}},
  ];
  if (deadCount > 0) items.push({{l:'Dead Convo', v:deadCount, c:'red'}});
  if (nonsenseCount > 0) items.push({{l:'Nonsense', v:nonsenseCount, c:'red'}});
  if (otherCount > 0) items.push({{l:'Other', v:otherCount, c:'red'}});
  g.innerHTML = items.map(c => `<div class="stat-card"><div class="label">${{c.l}}</div><div class="value ${{c.c}}">${{c.v}}</div></div>`).join('');
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

  el.innerHTML = slice.map((ex, i) => {{
    const isRemoved = ex._filter_status === 'REMOVED';
    const statusCls = isRemoved ? 'dead' : 'alive';
    const removalType = ex._filter_removal_type || '';
    const typeLabel = removalType === 'dead' ? 'DEAD CONVO'
                    : removalType === 'nonsense' ? 'NONSENSE'
                    : removalType ? removalType.toUpperCase() : 'REMOVED';
    const statusLabel = isRemoved ? ('✗ ' + typeLabel) : '✓ KEPT';
    const statusIcon = isRemoved ? (removalType === 'nonsense' ? '🤪' : '💀') : '✅';

    return `
    <div class="example-card ${{statusCls}}">
      <div class="card-header ${{statusCls}}">
        <div style="display:flex;align-items:center;gap:10px;">
          <span class="idx ${{statusCls}}">#${{ex._filter_index+1}}</span>
          <span class="status-badge ${{statusCls}}">${{statusIcon}} ${{statusLabel}}</span>
        </div>
        <span class="scenario">${{escapeHtml(ex._scenario_name||'')}} &middot; turn ${{ex._turn_index}}</span>
      </div>
      <div class="card-body">
        <div class="section-label">${{isRemoved ? (removalType === 'nonsense' ? '🤪' : '💀') : '🔍'}} Filter Verdict</div>
        <div class="filter-reason ${{statusCls}}">${{escapeHtml(ex._filter_reason)}}</div>

        ${{showContext ? `
          <div class="section-label">Conversation Context</div>
          <div class="context-box">${{fmtChat(ex._chat_messages_before)}}</div>

          <div class="section-label">Last User Message</div>
          <div class="context-box"><span style="color:var(--green);font-weight:600">USER:</span> ${{escapeHtml(ex.input)}}</div>
        ` : ''}}

        ${{showCot && ex._persona_chain_of_thought ? `
          <div class="section-label">💭 Persona's Internal Reaction</div>
          <div class="cot-box">${{escapeHtml(ex._persona_chain_of_thought)}}</div>` : ''}}

        ${{ex._observable_guideline ? `
          <div class="section-label">🔍 Observable Guideline</div>
          <div style="background:#1a1a2a;border:1px solid #2d2d4a;padding:12px 14px;border-radius:6px;font-size:13px;color:#7eb8da;border-left:3px solid var(--accent);">${{escapeHtml(ex._observable_guideline)}}</div>` : ''}}

        ${{ex._persona_visible_response ? `
          <div class="section-label">👤 What the persona replied</div>
          <div style="background:var(--surface2);padding:10px 14px;border-radius:6px;font-size:13px;color:var(--green);border-left:3px solid var(--green);">${{escapeHtml(ex._persona_visible_response).replace(/&lt;msg&gt;/g, '<br>')}}</div>` : ''}}

        ${{ex._judge_reason ? `
          <div class="section-label">⚖️ DPO Judge Reason</div>
          <div class="judge-box">${{escapeHtml(ex._judge_reason)}}</div>` : ''}}

        ${{showPairs ? `
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
          </div>` : ''}}

        ${{showMeta ? `
          <div class="section-label">Metadata</div>
          <div class="metadata-grid">
            <div class="metadata-item"><strong>Scenario:</strong> ${{escapeHtml(ex._scenario_id||'')}}</div>
            <div class="metadata-item"><strong>Turn:</strong> ${{ex._turn_index}}</div>
            <div class="metadata-item"><strong>Rewrite model:</strong> ${{escapeHtml(ex._rewrite_model||'')}}</div>
            <div class="metadata-item"><strong>Run:</strong> ${{escapeHtml(ex._run_label||'')}}</div>
            <div class="metadata-item"><strong>Msgs before:</strong> ${{(ex._chat_messages_before||[]).length}}</div>
          </div>` : ''}}
      </div>
    </div>`;
  }}).join('');

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

function toggleContext() {{ showContext=!showContext; document.getElementById('toggle-context').classList.toggle('active'); renderExamples(); }}
function toggleThink() {{ showThink=!showThink; document.getElementById('toggle-think').classList.toggle('active'); renderExamples(); }}
function toggleCot() {{ showCot=!showCot; document.getElementById('toggle-cot').classList.toggle('active'); renderExamples(); }}
function togglePairs() {{ showPairs=!showPairs; document.getElementById('toggle-pairs').classList.toggle('active'); renderExamples(); }}
function toggleMeta() {{ showMeta=!showMeta; document.getElementById('toggle-meta').classList.toggle('active'); renderExamples(); }}

document.getElementById('filter-status').addEventListener('change', () => {{ page=0; renderExamples(); }});
document.getElementById('filter-scenario').addEventListener('change', () => {{ page=0; renderExamples(); }});
document.getElementById('search-input').addEventListener('input', () => {{ page=0; renderExamples(); }});

let distView = 'scenario';

function setDistView(view) {{
  distView = view;
  document.querySelectorAll('.dist-tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  renderDistribution();
}}

function renderDistribution() {{
  const chart = document.getElementById('dist-chart');
  const legend = document.getElementById('dist-legend');

  if (distView === 'yield') {{
    renderYieldChart(chart);
    legend.innerHTML = `
      <div class="dist-legend-item"><div class="dist-legend-swatch" style="background:var(--accent)"></div> DPO datapoints</div>
      <div class="dist-legend-item"><div class="dist-legend-swatch" style="background:var(--surface2);border:1px solid var(--border)"></div> Runs (no disagreement)</div>
    `;
    return;
  }}

  // Restore default legend
  legend.innerHTML = `
    <div class="dist-legend-item"><div class="dist-legend-swatch" style="background:var(--green)"></div> Kept</div>
    <div class="dist-legend-item"><div class="dist-legend-swatch" style="background:var(--red);opacity:0.7"></div> Removed</div>
  `;

  // Group by scenario or run
  const groups = {{}};
  ALL.forEach(ex => {{
    const key = distView === 'scenario'
      ? (ex._scenario_name || 'unknown')
      : (ex._run_label || ex._run_id || 'unknown');
    if (!groups[key]) groups[key] = {{kept:0, removed:0}};
    if (ex._filter_status === 'REMOVED') groups[key].removed++;
    else groups[key].kept++;
  }});
  // Sort by total count descending
  const sorted = Object.entries(groups).sort((a,b) =>
    (b[1].kept + b[1].removed) - (a[1].kept + a[1].removed));
  const maxTotal = sorted.length ? (sorted[0][1].kept + sorted[0][1].removed) : 1;

  chart.innerHTML = sorted.map(([name, counts]) => {{
    const total = counts.kept + counts.removed;
    const keptPct = (counts.kept / maxTotal * 100).toFixed(1);
    const removedPct = (counts.removed / maxTotal * 100).toFixed(1);
    const label = distView === 'run' ? name.replace(/^run_impersona_/, '').substring(0, 40) : name;
    return `<div class="dist-row">
      <div class="dist-label" title="${{escapeHtml(name)}}">${{escapeHtml(label)}}</div>
      <div class="dist-bar-bg">
        <div class="dist-bar-kept" style="width:${{keptPct}}%" title="Kept: ${{counts.kept}}"></div>
        <div class="dist-bar-removed" style="width:${{removedPct}}%" title="Removed: ${{counts.removed}}"></div>
      </div>
      <div class="dist-count">${{counts.kept}}✓ ${{counts.removed}}✗ (${{total}})</div>
    </div>`;
  }}).join('');
}}

function renderYieldChart(chart) {{
  if (!Object.keys(RUN_STATS).length) {{
    chart.innerHTML = '<div style="color:var(--muted);padding:20px;text-align:center;">No run stats available. Pass run_stats to generate_filter_viewer_html().</div>';
    return;
  }}
  // Build rows: scenario → runs, dpo, rate
  const rows = Object.entries(RUN_STATS).map(([sid, s]) => ({{
    id: sid, name: s.name || sid, runs: s.runs || 0, dpo: s.dpo || 0,
    rate: s.runs > 0 ? (s.dpo / s.runs) : 0,
  }}));
  rows.sort((a,b) => b.rate - a.rate);
  const maxRuns = Math.max(...rows.map(r => r.runs));
  const maxDpo = Math.max(...rows.map(r => r.dpo));
  const maxVal = Math.max(maxRuns, maxDpo, 1);
  const totalRuns = rows.reduce((s,r) => s + r.runs, 0);
  const totalDpo = rows.reduce((s,r) => s + r.dpo, 0);

  let html = `<div style="color:var(--muted);font-size:13px;margin-bottom:12px;">
    Total: ${{totalRuns}} conversation runs → ${{totalDpo}} DPO datapoints
    (overall yield: ${{(totalDpo/totalRuns).toFixed(2)}} DPO/run)
  </div>`;

  html += rows.map(r => {{
    const dpoPct = (r.dpo / maxVal * 100).toFixed(1);
    const emptyPct = ((r.runs - r.dpo) / maxVal * 100).toFixed(1);
    const emptyW = r.runs > r.dpo ? emptyPct : '0';
    return `<div class="dist-row">
      <div class="dist-label" title="${{escapeHtml(r.id)}}">${{escapeHtml(r.name)}}</div>
      <div class="dist-bar-bg">
        <div class="dist-bar-kept" style="width:${{dpoPct}}%;background:var(--accent)" title="DPO: ${{r.dpo}}"></div>
        <div style="width:${{emptyW}}%;height:100%;background:var(--surface2);border-right:1px solid var(--border)" title="Runs w/o DPO: ${{Math.max(0, r.runs - r.dpo)}}"></div>
      </div>
      <div class="dist-count">${{r.dpo}}/${{r.runs}} (${{r.rate.toFixed(2)}})</div>
    </div>`;
  }}).join('');

  chart.innerHTML = html;
}}

renderStats();
renderScenarioFilter();
renderDistribution();
renderExamples();
</script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process a DPO dataset to filter out bad training examples."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the DPO dataset JSON file to filter.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for the filtered dataset. "
            "Default: <input_stem>_filtered.json"
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=model_for("dpo_filter"),
        help="Model to use as the LLM judge (default: dpo_filter in configs/models.yaml)",
    )
    parser.add_argument(
        "--judge-reasoning",
        type=str,
        default=model_for("dpo_filter", "reasoning"),
        help="Reasoning effort for the judge model (default: dpo_filter in configs/models.yaml)",
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=model_for("dpo_filter", "base_url"),
        help=(
            "OpenAI-compatible endpoint for the judge model, e.g. https://openrouter.ai/api/v1 "
            "(default: dpo_filter in configs/models.yaml)"
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max parallel judge calls (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show classifications without writing any files",
    )
    parser.add_argument(
        "--save-classifications",
        action="store_true",
        help="Save the full classification results to a JSON file",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    # Resolve output path
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        stem, ext = os.path.splitext(input_path)
        output_path = f"{stem}_filtered{ext}"

    print(f"\n{'═' * 66}")
    print(f"  IMPersona DPO Dataset Filter")
    print(f"{'═' * 66}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Judge:  {args.judge_model} (reasoning={args.judge_reasoning})")

    # ── Load dataset ──
    print(f"\n📥 Loading dataset...")
    with open(input_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if not examples:
        print("❌ No examples in dataset.")
        sys.exit(1)

    print(f"  Loaded {len(examples)} examples")

    # ── Initialize judge ──
    judge = AIPartnerClient(
        model=args.judge_model,
        reasoning=args.judge_reasoning,
        base_url=args.judge_base_url or None,
    )

    # ── Run quality filter ──
    print(f"\n🔍 Running DPO quality filter...")
    print(f"  (parallel workers: {args.max_workers})\n")

    kept, removed, classifications = run_quality_filter(
        examples=examples,
        judge=judge,
        max_workers=args.max_workers,
        verbose=True,
    )

    # ── Summary ──
    print(f"\n{'─' * 66}")
    print(f"  📊 Filter Results")
    print(f"{'─' * 66}")
    # Count by removal type
    dead_count = sum(1 for c in classifications if c.get("should_remove") and c.get("removal_type") == "dead")
    nonsense_count = sum(1 for c in classifications if c.get("should_remove") and c.get("removal_type") == "nonsense")
    other_count = len(removed) - dead_count - nonsense_count

    print(f"  Total examples:    {len(examples)}")
    print(f"  Kept:              {C.GREEN}{len(kept)}{C.RESET}")
    print(f"  Removed:           {C.RED}{len(removed)}{C.RESET}")
    if dead_count:
        print(f"    Dead convo:      {dead_count}")
    if nonsense_count:
        print(f"    Nonsense:        {nonsense_count}")
    if other_count:
        print(f"    Other:           {other_count}")
    print(f"  Filter rate:       {len(removed)/len(examples)*100:.1f}%")

    # Show which scenarios/turns were filtered
    if removed:
        print(f"\n  Removed examples:")
        for cls in classifications:
            if cls.get("should_remove"):
                rtype = cls.get("removal_type", "")
                tag = f"[{rtype.upper()}] " if rtype else ""
                print(
                    f"    [{cls['index']}] {tag}{cls['scenario']} turn {cls['turn_index']}: "
                    f"{cls['reason'][:70]}"
                )

    # Usage stats
    usage = judge.get_usage()
    print(f"\n  Judge usage ({args.judge_model}):")
    print(f"    Calls: {usage['num_calls']}")
    print(f"    Tokens: {usage['total_tokens']}")
    print(f"    Estimated cost: ${usage['estimated_cost_usd']:.4f}")

    if args.dry_run:
        print(f"\n🏃 Dry run — no files written.")
        return

    # ── Write filtered dataset ──
    print(f"\n💾 Writing filtered dataset...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  ✅ Saved {len(kept)} examples to {output_path}")

    # ── Save classifications if requested ──
    if args.save_classifications:
        cls_path = output_path.replace(".json", "_classifications.json")
        with open(cls_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input_file": input_path,
                    "judge_model": args.judge_model,
                    "judge_reasoning": args.judge_reasoning,
                    "total_examples": len(examples),
                    "kept": len(kept),
                    "removed": len(removed),
                    "classifications": classifications,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")
        print(f"  ✅ Saved classifications to {cls_path}")

    # ── Generate HTML viewer ──
    viewer_path = output_path.replace(".json", "_viewer.html")
    print(f"\n🌐 Generating interactive viewer...")
    viewer_html = generate_filter_viewer_html(
        all_examples=examples,
        classifications=classifications,
        input_file=input_path,
        judge_model=args.judge_model,
        judge_reasoning=args.judge_reasoning,
    )
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(viewer_html)
    print(f"  ✅ Saved viewer to {viewer_path}")

    print(f"\n{'─' * 66}")
    print(f"  ✅ Done! Filtered dataset ready.")
    print(f"  {len(kept)}/{len(examples)} examples kept ({len(removed)} removed)")
    print(f"  File:   {output_path}")
    print(f"  Viewer: {viewer_path}")
    print(f"{'─' * 66}\n")


if __name__ == "__main__":
    main()
