#!/usr/bin/env python3
"""
Impersona-Env — Dataset Visualizer
===================================

Interactive web-based visualizer for inspecting the generated dataset.
Shows each training example with its goal annotation, conversation history,
target response, and dataset-level statistics.

Usage (from the repo root):
    python -m IMPersona2.dataset_visualizer --list                        # list available datasets
    python -m IMPersona2.dataset_visualizer --dataset "experiment_output/Jane Doe/dataset.json"
    python -m IMPersona2.dataset_visualizer --port 8080
"""

import argparse
import json
import os
import sys
from collections import Counter
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser


def load_dataset_bundle(path: str):
    """Load the bundled dataset file. Returns (examples, metadata)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept a bundled dataset.json or a plain list of examples
    if isinstance(data, list):
        return data, {}
    return data.get("examples", []), data.get("metadata", {})


def build_html(examples: list, metadata: dict, dataset_path: str) -> str:
    """Build a self-contained HTML page for dataset visualization."""

    # Compute per-conversation color map (for visual grouping)
    all_conv_indices = sorted(set(
        ex.get("metadata", {}).get("conversation_idx", 0)
        for ex in examples
    ))
    conv_colors = {}
    palette = [
        "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336",
        "#00BCD4", "#795548", "#607D8B", "#E91E63", "#CDDC39",
        "#3F51B5", "#FF5722",
    ]
    for i, c in enumerate(all_conv_indices):
        conv_colors[c] = palette[i % len(palette)]

    # Collect unique conversation goals for the overview
    conv_goals = {}
    for ex in examples:
        cidx = ex.get("metadata", {}).get("conversation_idx", 0)
        goal = ex.get("metadata", {}).get("goal", "")
        if cidx not in conv_goals and goal:
            conv_goals[cidx] = goal
    conv_goals_json = json.dumps(conv_goals, ensure_ascii=False)

    # Build examples JSON for the frontend
    examples_json = json.dumps(examples, ensure_ascii=False)
    conv_colors_json = json.dumps(conv_colors)

    # Goal source counts
    goal_source_counts = Counter(
        ex.get("metadata", {}).get("goal_source", "unknown") or "unknown"
        for ex in examples
    )
    goal_source_counts_json = json.dumps(dict(goal_source_counts.most_common()))

    stats = metadata.get("stats") or metadata.get("dataset_stats")
    filter_stats = metadata.get("filter_stats")
    stats_combined = {}
    if stats:
        stats_combined["dataset_stats"] = stats
    if filter_stats:
        stats_combined["filter_stats"] = filter_stats
    stats_json = json.dumps(stats_combined, indent=2) if stats_combined else "{}"

    rel_summary = metadata.get("relationship_summary")
    rel_summary_json = json.dumps({"summary": rel_summary}, ensure_ascii=False) if rel_summary else "null"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Impersona-Env — Dataset Visualizer</title>
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #21262d;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --accent2: #3fb950;
    --danger: #f85149;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}

  /* Header */
  .header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 20px 0; border-bottom: 1px solid var(--border); margin-bottom: 24px;
  }}
  .header h1 {{ font-size: 24px; font-weight: 600; }}
  .header h1 span {{ color: var(--accent); }}
  .header .meta {{ color: var(--text-muted); font-size: 14px; }}

  /* Stats Panel */
  .stats-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px; margin-bottom: 24px;
  }}
  .stat-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }}
  .stat-card .label {{ color: var(--text-muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .stat-card .value {{ font-size: 24px; font-weight: 600; margin-top: 4px; }}
  .stat-card .detail {{ color: var(--text-muted); font-size: 13px; margin-top: 4px; }}

  /* Goal Distribution */
  .goal-chart {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px; margin-bottom: 24px;
  }}
  .goal-chart h3 {{ margin-bottom: 16px; font-size: 16px; }}
  .goal-bar-container {{ display: flex; flex-direction: column; gap: 8px; }}
  .goal-bar-row {{
    display: flex; align-items: center; gap: 12px;
  }}
  .goal-bar-label {{ width: 200px; font-size: 13px; color: var(--text-muted); text-align: right; flex-shrink: 0; }}
  .goal-bar-track {{ flex: 1; background: var(--surface2); border-radius: 4px; height: 24px; position: relative; overflow: hidden; }}
  .goal-bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s ease; display: flex; align-items: center; padding-left: 8px; font-size: 12px; font-weight: 600; }}
  .goal-bar-count {{ width: 50px; font-size: 13px; color: var(--text-muted); text-align: left; flex-shrink: 0; }}

  /* Controls */
  .controls {{
    display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; align-items: center;
  }}
  .controls select, .controls input {{
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text); padding: 8px 12px; border-radius: 6px; font-size: 14px;
  }}
  .controls input {{ flex: 1; min-width: 200px; }}
  .controls label {{ color: var(--text-muted); font-size: 13px; }}

  /* Example Cards */
  .example-list {{ display: flex; flex-direction: column; gap: 16px; }}
  .example-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; overflow: hidden; transition: border-color 0.2s;
  }}
  .example-card:hover {{ border-color: var(--accent); }}
  .example-header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; border-bottom: 1px solid var(--border);
    background: var(--surface2);
  }}
  .example-header .idx {{ font-weight: 600; color: var(--accent); }}
  .example-header .timestamp {{ color: var(--text-muted); font-size: 13px; }}
  .goal-badge {{
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600; color: #fff;
  }}
  .example-body {{ padding: 16px; }}
  .section-label {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px;
    color: var(--text-muted); margin-bottom: 6px; margin-top: 14px;
  }}
  .section-label:first-child {{ margin-top: 0; }}
  .system-prompt {{
    background: var(--surface2); padding: 10px 14px; border-radius: 6px;
    font-size: 13px; color: var(--text-muted); white-space: pre-wrap;
    word-break: break-word; border-left: 3px solid var(--accent);
  }}
  .conversation {{
    background: var(--bg); padding: 12px 14px; border-radius: 6px;
    font-size: 13px; white-space: pre-wrap; word-break: break-word;
    max-height: 300px; overflow-y: auto;
  }}
  .conversation .msg-me {{ color: var(--accent2); }}
  .conversation .msg-other {{ color: var(--accent); }}
  .conversation .msg-ts {{ color: var(--text-muted); font-size: 11px; }}
  .output-box {{
    background: #1a2a1a; border: 1px solid #2d4a2d; padding: 12px 14px;
    border-radius: 6px; font-size: 14px; white-space: pre-wrap;
    word-break: break-word; color: var(--accent2);
  }}
  .cot-box {{
    background: #2a1a2a; border: 1px solid #4a2d4a; padding: 12px 14px;
    border-radius: 6px; font-size: 13px; white-space: pre-wrap;
    word-break: break-word; color: #c9a0dc; font-style: italic;
    border-left: 3px solid #9b59b6;
  }}
  .metadata-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 8px; margin-top: 12px;
  }}
  .metadata-item {{
    font-size: 12px; color: var(--text-muted);
  }}
  .metadata-item strong {{ color: var(--text); }}

  /* Pagination */
  .pagination {{
    display: flex; justify-content: center; gap: 8px; margin-top: 24px; margin-bottom: 40px;
  }}
  .pagination button {{
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text); padding: 8px 16px; border-radius: 6px;
    cursor: pointer; font-size: 14px;
  }}
  .pagination button:hover {{ background: var(--surface2); border-color: var(--accent); }}
  .pagination button:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  .pagination .page-info {{ display: flex; align-items: center; color: var(--text-muted); font-size: 14px; }}

  /* Toggle */
  .toggle-btn {{
    background: var(--surface2); border: 1px solid var(--border);
    color: var(--text); padding: 6px 14px; border-radius: 6px;
    cursor: pointer; font-size: 13px; transition: all 0.2s;
  }}
  .toggle-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}

  /* Conversation Navigation */
  .conv-nav {{
    display: flex; justify-content: center; align-items: center; gap: 16px;
    padding: 16px 0; margin-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }}
  .conv-nav button {{
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text); padding: 10px 20px; border-radius: 6px;
    cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.2s;
  }}
  .conv-nav button:hover {{ background: var(--surface2); border-color: var(--accent); }}
  .conv-nav button:disabled {{ opacity: 0.3; cursor: not-allowed; }}
  .conv-nav-info {{
    text-align: center; min-width: 300px;
  }}
  .conv-nav-info .conv-title {{ font-size: 16px; font-weight: 600; color: var(--text); }}
  .conv-nav-info .conv-detail {{ font-size: 13px; color: var(--text-muted); margin-top: 2px; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <div>
      <h1>IMPersona <span>II</span> — Dataset Visualizer</h1>
      <div class="meta">Source: {os.path.basename(dataset_path)}</div>
    </div>
  </div>

  <!-- Stats -->
  <div class="stats-grid" id="stats-grid"></div>

  <!-- Relationship Summary -->
  <div class="goal-chart" id="relationship-section" style="display:none;">
    <h3>🤝 Relationship Summary</h3>
    <div id="relationship-content" style="font-size: 14px; line-height: 1.7; color: var(--text); padding: 8px 0;"></div>
    <div id="relationship-meta" style="font-size: 12px; color: var(--text-muted); margin-top: 8px;"></div>
  </div>

  <!-- Conversation Navigation -->
  <div class="conv-nav" id="conv-nav">
    <button id="btn-prev" onclick="prevConv()">← Prev</button>
    <div class="conv-nav-info" id="conv-nav-info"></div>
    <button id="btn-next" onclick="nextConv()">Next →</button>
  </div>

  <!-- Conversation Goal Banner -->
  <div class="goal-chart" id="conv-goal-banner" style="display:none;">
    <h3>🎯 Conversation Goal</h3>
    <div id="conv-goal-text" style="font-size: 14px; line-height: 1.7; color: var(--text); padding: 4px 0;"></div>
  </div>

  <!-- Controls -->
  <div class="controls">
    <input type="text" id="search-input" placeholder="Search across all conversations...">
    <button class="toggle-btn active" id="toggle-cot" onclick="toggleCot()">Chain of Thought</button>
    <button class="toggle-btn active" id="toggle-system" onclick="toggleSystem()">System Prompt</button>
    <button class="toggle-btn active" id="toggle-metadata" onclick="toggleMetadata()">Metadata</button>
  </div>

  <!-- Examples -->
  <div class="example-list" id="example-list"></div>

</div>

<script>
const ALL_EXAMPLES = {examples_json};
const CONV_COLORS = {conv_colors_json};
const CONV_GOALS = {conv_goals_json};
const STATS = {stats_json};
const REL_SUMMARY = {rel_summary_json};

// Group examples by conversation index
const CONV_INDICES = [...new Set(ALL_EXAMPLES.map(ex => ex.metadata?.conversation_idx ?? 0))].sort((a, b) => a - b);
const EXAMPLES_BY_CONV = {{}};
ALL_EXAMPLES.forEach(ex => {{
  const cidx = ex.metadata?.conversation_idx ?? 0;
  if (!EXAMPLES_BY_CONV[cidx]) EXAMPLES_BY_CONV[cidx] = [];
  EXAMPLES_BY_CONV[cidx].push(ex);
}});

let currentConvPos = 0;  // index into CONV_INDICES
let showSystem = true;
let showMetadata = true;
let showCot = true;

// ─── Stats ───
function renderStats() {{
  const grid = document.getElementById('stats-grid');
  const ds = STATS.dataset_stats || {{}};
  const cards = [
    {{ label: 'Total Examples', value: ALL_EXAMPLES.length }},
    {{ label: 'Conversations', value: CONV_INDICES.length }},
    {{ label: 'Date Range', value: ds.date_range ? ds.date_range.earliest + ' → ' + ds.date_range.latest : 'N/A', small: true }},
    {{ label: 'Avg Input Words', value: ds.input_word_counts ? ds.input_word_counts.mean : 'N/A',
       detail: ds.input_word_counts ? 'min: ' + ds.input_word_counts.min + ' / max: ' + ds.input_word_counts.max : '' }},
    {{ label: 'Avg Output Words', value: ds.output_word_counts ? ds.output_word_counts.mean : 'N/A',
       detail: ds.output_word_counts ? 'min: ' + ds.output_word_counts.min + ' / max: ' + ds.output_word_counts.max : '' }},
    {{ label: 'Other Speaker', value: ds.other_speaker || 'N/A' }},
  ];
  grid.innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="label">${{c.label}}</div>
      <div class="value" style="font-size: ${{c.small ? '16px' : '24px'}}">${{c.value}}</div>
      ${{c.detail ? '<div class="detail">' + c.detail + '</div>' : ''}}
    </div>
  `).join('');
}}

// ─── Formatting ───
function formatConversation(inputText) {{
  return inputText
    .replace(/\\[([^\\]]+)\\] Me:/g, '<span class="msg-ts">[$1]</span> <span class="msg-me">Me:</span>')
    .replace(/\\[([^\\]]+)\\] ([^:]+):/g, '<span class="msg-ts">[$1]</span> <span class="msg-other">$2:</span>');
}}

function escapeHtml(text) {{
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}}

// ─── Conversation Navigation ───
function renderConvNav() {{
  const cidx = CONV_INDICES[currentConvPos];
  const examples = EXAMPLES_BY_CONV[cidx] || [];
  const info = document.getElementById('conv-nav-info');
  const dateRange = examples.length > 0
    ? examples[0].timestamp + (examples.length > 1 ? ' → ' + examples[examples.length - 1].timestamp : '')
    : '';
  info.innerHTML = `
    <div class="conv-title">Conversation ${{cidx}}</div>
    <div class="conv-detail">${{examples.length}} examples · ${{dateRange}} · ${{currentConvPos + 1}} of ${{CONV_INDICES.length}}</div>
  `;
  document.getElementById('btn-prev').disabled = currentConvPos === 0;
  document.getElementById('btn-next').disabled = currentConvPos >= CONV_INDICES.length - 1;

  // Show goal banner
  const goal = CONV_GOALS[String(cidx)] || CONV_GOALS[cidx] || '';
  const banner = document.getElementById('conv-goal-banner');
  if (goal) {{
    banner.style.display = 'block';
    document.getElementById('conv-goal-text').textContent = goal;
  }} else {{
    banner.style.display = 'none';
  }}
}}

function prevConv() {{
  if (currentConvPos > 0) {{
    currentConvPos--;
    renderAll();
    window.scrollTo({{ top: document.getElementById('conv-nav').offsetTop - 10, behavior: 'smooth' }});
  }}
}}

function nextConv() {{
  if (currentConvPos < CONV_INDICES.length - 1) {{
    currentConvPos++;
    renderAll();
    window.scrollTo({{ top: document.getElementById('conv-nav').offsetTop - 10, behavior: 'smooth' }});
  }}
}}

// ─── Example Cards ───
function renderExamples() {{
  const list = document.getElementById('example-list');
  const cidx = CONV_INDICES[currentConvPos];
  let examples = EXAMPLES_BY_CONV[cidx] || [];

  // Apply search filter
  const searchQuery = document.getElementById('search-input').value.toLowerCase();
  if (searchQuery) {{
    examples = examples.filter(ex => {{
      const combined = (ex.input + ' ' + ex.output + ' ' + ex.instruction + ' ' + (ex.metadata?.goal || '')).toLowerCase();
      return combined.includes(searchQuery);
    }});
  }}

  list.innerHTML = examples.map((ex, i) => {{
    const convColor = CONV_COLORS[cidx] || '#666';

    return `
      <div class="example-card">
        <div class="example-header">
          <div>
            <span class="idx">Example ${{i + 1}} of ${{examples.length}}</span>
          </div>
          <div class="timestamp">${{ex.timestamp}}</div>
        </div>
        <div class="example-body">
          ${{showSystem ? `
            <div class="section-label">System Prompt (instruction)</div>
            <div class="system-prompt">${{escapeHtml(ex.instruction)}}</div>
          ` : ''}}

          <div class="section-label">Conversation History (input)</div>
          <div class="conversation">${{formatConversation(escapeHtml(ex.input))}}</div>

          ${{(() => {{
            const output = ex.output || '';
            const hasCot = output.includes('<think>') && output.includes('</think>');
            let cotText = '';
            let responseText = output;
            if (hasCot) {{
              const parts = output.split('</think>');
              cotText = parts[0].replace('<think>', '').trim();
              responseText = parts.slice(1).join('</think>').trim();
            }}
            if (!cotText && ex.metadata?.chain_of_thought) {{
              cotText = ex.metadata.chain_of_thought;
            }}
            let html = '';
            if (showCot && cotText) {{
              html += '<div class="section-label">💭 Chain of Thought (internal monologue)</div>';
              html += '<div class="cot-box">' + escapeHtml(cotText) + '</div>';
            }}
            html += '<div class="section-label">Target Response (output)</div>';
            html += '<div class="output-box">' + escapeHtml(responseText) + '</div>';
            return html;
          }})()}}

          ${{showMetadata ? `
            <div class="section-label">Metadata</div>
            <div class="metadata-grid">
              <div class="metadata-item"><strong>Other speaker:</strong> ${{ex.metadata?.other_speaker || 'N/A'}}</div>
              <div class="metadata-item"><strong>History turns:</strong> ${{ex.metadata?.num_history_turns || 'N/A'}}</div>
              <div class="metadata-item"><strong>Goal source:</strong> ${{ex.metadata?.goal_source || 'N/A'}}</div>
            </div>
          ` : ''}}
        </div>
      </div>
    `;
  }}).join('');

  if (examples.length === 0) {{
    list.innerHTML = '<div style="text-align:center; color:var(--text-muted); padding:40px;">No matching examples in this conversation.</div>';
  }}
}}

// ─── Toggles ───
function toggleCot() {{
  showCot = !showCot;
  document.getElementById('toggle-cot').classList.toggle('active');
  renderExamples();
}}
function toggleSystem() {{
  showSystem = !showSystem;
  document.getElementById('toggle-system').classList.toggle('active');
  renderExamples();
}}
function toggleMetadata() {{
  showMetadata = !showMetadata;
  document.getElementById('toggle-metadata').classList.toggle('active');
  renderExamples();
}}

// ─── Search ───
document.getElementById('search-input').addEventListener('input', renderExamples);

// ─── Keyboard Navigation ───
document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') prevConv();
  if (e.key === 'ArrowRight') nextConv();
}});

// ─── Relationship Summary ───
function renderRelationshipSummary() {{
  if (!REL_SUMMARY) return;
  const section = document.getElementById('relationship-section');
  section.style.display = 'block';
  document.getElementById('relationship-content').textContent = REL_SUMMARY.summary || '';
  const metaParts = [];
  if (REL_SUMMARY.other_speaker) metaParts.push('Speaker: ' + REL_SUMMARY.other_speaker);
  if (REL_SUMMARY.model) metaParts.push('Model: ' + REL_SUMMARY.model);
  if (REL_SUMMARY.num_conversations_analyzed) metaParts.push('Conversations analyzed: ' + REL_SUMMARY.num_conversations_analyzed);
  document.getElementById('relationship-meta').textContent = metaParts.join(' · ');
}}

// ─── Render All ───
function renderAll() {{
  renderConvNav();
  renderExamples();
}}

// ─── Init ───
renderStats();
renderRelationshipSummary();
renderAll();
</script>
</body>
</html>"""
    return html


def _find_available_datasets(output_dir: str = "experiment_output") -> list:
    """Scan output_dir for per-conversation/model dataset.json files.

    Finds output_dir/<contact>/<model>/dataset.json, and also
    output_dir/<contact>/dataset.json (no model subfolder).
    """
    datasets = []
    if not os.path.isdir(output_dir):
        return datasets
    for contact in sorted(os.listdir(output_dir)):
        contact_dir = os.path.join(output_dir, contact)
        if not os.path.isdir(contact_dir):
            continue

        # contact/<model>/dataset.json
        for entry in sorted(os.listdir(contact_dir)):
            ds_path = os.path.join(contact_dir, entry, "dataset.json")
            if os.path.isdir(os.path.join(contact_dir, entry)) and os.path.isfile(ds_path):
                try:
                    with open(ds_path, "r", encoding="utf-8") as f:
                        bundle = json.load(f)
                    meta = bundle.get("metadata", {})
                    datasets.append({
                        "conversation_id": contact,
                        "model": entry,
                        "path": ds_path,
                        "num_examples": meta.get("num_examples", "?"),
                        "other_speaker": meta.get("other_speaker", "?"),
                        "num_conversations": meta.get("num_conversations", "?"),
                    })
                except Exception:
                    datasets.append({
                        "conversation_id": contact,
                        "model": entry,
                        "path": ds_path,
                        "num_examples": "?",
                        "other_speaker": "?",
                        "num_conversations": "?",
                    })

        # contact/dataset.json (no model subfolder)
        flat_path = os.path.join(contact_dir, "dataset.json")
        if os.path.isfile(flat_path):
            try:
                with open(flat_path, "r", encoding="utf-8") as f:
                    bundle = json.load(f)
                meta = bundle.get("metadata", {})
                datasets.append({
                    "conversation_id": contact,
                    "model": "(no model folder)",
                    "path": flat_path,
                    "num_examples": meta.get("num_examples", "?"),
                    "other_speaker": meta.get("other_speaker", "?"),
                    "num_conversations": meta.get("num_conversations", "?"),
                })
            except Exception:
                pass
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Impersona-Env — Dataset Visualizer")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to a specific dataset JSON to visualize")
    parser.add_argument("--list", action="store_true",
                        help="List all available processed datasets and exit")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port for the visualization server")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open the browser automatically")

    args = parser.parse_args()

    # ── List mode ──
    if args.list:
        datasets = _find_available_datasets()
        if not datasets:
            print("No processed datasets found in experiment_output/")
            print("Run `python run.py --step generate` first.")
            sys.exit(1)
        print(f"\n📂 Available datasets ({len(datasets)} found):\n")
        for ds in datasets:
            model_tag = ds.get('model', '?')
            print(f"  {ds['conversation_id']:20s}  "
                  f"model: {model_tag:25s}  "
                  f"{ds['num_examples']:>4} examples  "
                  f"speaker: {ds['other_speaker']:20s}  "
                  f"convs: {ds['num_conversations']}")
        print(f"\nVisualize one with:")
        print(f"  python dataset_visualizer.py --dataset {datasets[0]['path']}")
        sys.exit(0)

    # ── Resolve dataset path ──
    dataset_path = args.dataset
    if dataset_path is None:
        # Auto-detect: if there's exactly one, use it; otherwise prompt
        datasets = _find_available_datasets()
        if len(datasets) == 1:
            dataset_path = datasets[0]["path"]
        elif len(datasets) > 1:
            print(f"Multiple datasets found. Please specify one with --dataset:")
            for ds in datasets:
                print(f"  python dataset_visualizer.py --dataset {ds['path']}")
            sys.exit(1)
        else:
            print("❌ No datasets found. Run `python run.py --step generate` first.")
            sys.exit(1)

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        print("   Run `python run.py --step generate` first to create the dataset.")
        sys.exit(1)

    print(f"Loading dataset from {dataset_path} ...")
    examples, metadata = load_dataset_bundle(dataset_path)
    print(f"Loaded {len(examples)} examples")
    if metadata.get("relationship_summary"):
        print(f"Loaded relationship summary ({len(metadata['relationship_summary'])} chars)")
    if metadata.get("conversation_goals"):
        print(f"Loaded {len(metadata['conversation_goals'])} conversation goals")

    # Build HTML
    html_content = build_html(examples, metadata, dataset_path)

    # Write to the dataset's own directory
    output_html = os.path.join(os.path.dirname(dataset_path), "visualizer.html")
    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Built visualization → {output_html}")

    # Serve it
    os.chdir(os.path.dirname(output_html) or ".")
    handler = SimpleHTTPRequestHandler
    server = HTTPServer(("localhost", args.port), handler)
    url = f"http://localhost:{args.port}/{os.path.basename(output_html)}"
    print(f"\n🌐 Serving at {url}")
    print("   Press Ctrl+C to stop.\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
