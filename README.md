# Impersona-Env

Code for the blog post **[Impersona-Env: User Models as Environments for Model Personalization](https://benshi34.github.io/#/blog/impersona-env)**.

We train a model of a specific person (a *user model*) on their iMessage
history, then let a chat assistant interact with it indefinitely. The user
model's private reactions to each assistant reply become DPO preference
pairs, so the assistant learns that person's situational preferences and
collaboration patterns.

The pipeline produces two artifacts:

1. **An IMPersona (user model)** — a base model fine-tuned to text *as you*,
   emitting a `<think>` block of reconstructed internal reasoning before each
   message.
2. **A personalized assistant** — a LoRA adapter on an assistant model,
   DPO-trained on the IMPersona's reactions. This is the model that talks
   *to you*.

Both base models, and every other model the pipeline uses, are set in
[`configs/models.yaml`](configs/models.yaml); nothing assumes a particular model.

Plus a blind arena for comparing the assistant against memory-retrieval and
raw-scale baselines in your normal chat use.

## The loop

```
  ── User model ──────────────────────────────────────────────────────────
  1 export ─→ 2 select ─→ 3 generate ─→ 4 train
  iMessages   contacts    relationship    SFT the user model (Modal)
                          + context + CoT
                          annotations
                                            │
  ── Assistant ─────────────────────────────▼────────────────────────────
  5 eval ─→ 6 dpo ─→ 7 prepare-dpo ─→ 8 train-dpo
  rollouts   judge flags     combine,       DPO the assistant (Modal)
  across     disagreements,  dedup,             │
  130        rewrite reply   sharegpt           │
  scenarios  → (chosen,                         │
     ▲          rejected)                       │
     └──────────── iterate (optional) ──────────┘
                                                │
  ── Final evaluation ──────────────────────────▼────────────────────────
  9 memory ─→ 10 arena
  retrieval    blind A/B battles vs. the untrained assistant (with and
  index        without memory search) and a larger model with memory search
```

## Prerequisites

- macOS (for iMessage export)
- Python 3.11+
- [imessage-exporter](https://github.com/ReagentX/imessage-exporter) (`brew install imessage-exporter`; `run.py` can install it)
- A [Modal](https://modal.com) account (GPU training and serving)
- API keys for whichever providers your models use (see
  [`configs/models.yaml`](configs/models.yaml)). With the default models:
  - **OpenAI** — dataset annotation, memory-search embeddings
  - **Anthropic** — eval judge, DPO guideline translation
  - **OpenRouter** — arena baselines
  - **HuggingFace** — a token for downloading the base models. If one is
    gated (the default user model is), accept its license on its Hugging Face
    page first.

## Setup

```bash
git clone https://github.com/benshi34/impersona-env.git
cd impersona-env

python3 -m venv .venv
source .venv/bin/activate     # later commands (python, modal) run inside this venv
pip install -r requirements.txt

cp .env.example .env        # then fill in your keys

# Log in to Modal (inside the activated .venv — that's where `modal` is installed)
modal setup

# Secrets the deployed functions read
modal secret create huggingface-secret HF_TOKEN=<your-hf-token>
modal secret create impersona-api-keys ANTHROPIC_API_KEY=<your-anthropic-key>

# Deploy, from the repo root
modal deploy modal_app.py
```

`impersona-api-keys` holds the API keys that the eval judge and DPO
translation need when they run on Modal (`judge` and `translate` in
`configs/models.yaml`; both default to Claude). It accepts any of
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `OPENROUTER_API_KEY`, so include
whichever providers you use. To change it later, run the same command with
`--force`.

## Running it

Everything runs from the terminal through `run.py`. With no arguments it walks
through each step in order, asking whether to run, skip, or re-run; `--step`
jumps straight to one.

```bash
python run.py                     # interactive walkthrough, steps 1–10
python run.py --step export       # 1  Export iMessages + contacts
python run.py --step select       # 2  Choose contacts to train on
python run.py --step generate     # 3  Build annotated SFT data
python run.py --step train        # 4  SFT the IMPersona on Modal
python run.py --step eval         # 5  Roll out IMPersona vs. assistant, judge
python run.py --step fidelity     # 5b Style + fact-recall judges (optional)
python run.py --step dpo          # 6  Disagreements → preference pairs
python run.py --step prepare-dpo  # 7  Combine + format DPO data
python run.py --step train-dpo    # 8  DPO the assistant on Modal
python run.py --step memory       # 9  Build the memory-search index
python run.py --step arena        # 10 Blind arena
```

All personal data (exports, datasets, eval runs, arena verdicts) is written
inside the repo directory and is gitignored.

### 1. Export

Dumps every iMessage conversation to `imessage_export_mac/*.txt` with
imessage-exporter, and your contacts to `contacts.vcf` (via the Contacts app,
or a `.vcf` you provide) so phone numbers resolve to names.

### 2. Select

Ranks conversations by how many training examples they'd produce. You merge
duplicates (the same person on phone + email), rename, and pick which to
include. Output: `selected_conversations.json`.

### 3. Generate

You'll first enter your name and nicknames, saved to `persona_config.json`
and used in every prompt. Then for each selected contact,
`IMPersona2.generate_dataset`:

```
raw .txt → parse + segment (48h gaps)
        → relationship summary per contact
        → situational context preceding each conversation
        → reconstructed internal reasoning per reply (with strict skip rules)
        → aggressive quality filters
        → experiment_output/<contact>/<model>_<version>/dataset.json + visualizer.html
```

The annotation model is `annotation` in `configs/models.yaml`. Datasets are
written to a per-model folder, so switching models puts the new datasets
alongside the old ones instead of overwriting them.

### 4. Train the IMPersona

Combines every contact's `dataset.json` into `training_data.json`, uploads it,
and runs LlamaFactory LoRA SFT on the user model's base (`user_model.base`)
on Modal. It uses 1, 2 or 4 H100s depending on the parameter count in the
model's name (`--gpus` on `run_sft_worker.py` overrides). The adapter is saved
to the Modal volume as `adapter-user-model`. Chat with it from the terminal with `python run_impersona.py`.

### 5. Eval

Rolls the IMPersona out against an assistant across the 130 scenarios in
`IMPersona2/eval_scenarios.json`. The scenarios reflect how the author uses
a chatbot (many are about grad school and research), so edit them to match the
kinds of conversations you actually have. The IMPersona's reasoning is
recorded each turn. An LLM judge scores coherence (1–7) and labels per-turn
**disagreements**, i.e. turns where the IMPersona's private reasoning critiques
the assistant's last reply. The assistant is `assistant.base` served on
Modal (`VLLMPartner`, optionally with a DPO adapter), or any API model.

`run.py` asks about two options that matter for the final setup (both are
also set in `configs/dpo_pipeline.yaml` for the orchestrator):
- `include_prior_cot` — show prior-turn `<think>` blocks in history. Requires an
  IMPersona trained on data processed by `python -m IMPersona2.add_prior_cot`.
- `critic_mode` — an addendum nudging the IMPersona to react authentically to
  the assistant.

### 5b. Fidelity evals (optional)

Two judges check the user model against your real data, beyond coherence.
Run them from `python run.py --step fidelity`.

- **Style.** Does the user model text like you? The judge compares its
  messages in each rollout with a fixed random sample of 60 of your real
  message bursts from `training_data.json`, and scores 1–7. It grades length,
  message splitting, capitalization, slang, tone, and register, not content.
- **Fact recall.** Does it recall and act on true facts about you? This is a
  pipeline, because the facts must be found in your data and checked by you:
  1. `propose`: for each scenario, an LLM picks facts you would plausibly
     draw on there, phrased as questions. They go into one shared catalog, so
     a fact used by many scenarios is only handled once.
  2. `search`: a search agent answers each question from your messages
     through the memory index (step 9). It returns an answer with verbatim
     quotes, or not found.
  3. `review`: you accept, edit, or reject each answer in the terminal.
  4. judge: for each rollout, the verified facts for its scenario go into the
     judge prompt. Each fact is labeled used correctly, contradicted, missed,
     or not applicable. The score is the share of applicable facts used
     correctly.

  The fact bank is stored in `.fact_bank/` (gitignored). Each stage can also
  be run on its own:

  ```bash
  python -m IMPersona2.fact_recall propose    # --scenarios / --category to limit
  python -m IMPersona2.fact_recall search
  python -m IMPersona2.fact_recall review     # resumable
  python -m IMPersona2.fact_recall status
  ```

Both judges run on an existing eval run and add `style` and `fact_recall` to
its `results.json`:

```bash
python -m IMPersona2.eval --eval-only eval_output/<run> --judges style,facts
```

### 6. Generate DPO data

For each flagged turn, a translator model converts the IMPersona's private
reaction into a guideline that only references what is *observable* in the
conversation. The same assistant then rewrites its reply following that
guideline. The pair is `chosen = rewrite`, `rejected = original`. Pairs whose
`chosen` text mentions the persona's name are dropped. Output:
`eval_output/<run>/dpo_dataset.json`.

### 7. Prepare DPO data

Combines every `dpo_dataset.json`, deduplicates, drops over-length examples,
and converts to multi-turn sharegpt. Output: `eval_output/dpo_combined.json`.

Optional extra passes:
- `python -m IMPersona2.filter_dpo_dataset --input …` — LLM-judge filter that
  drops pairs from dead or nonsensical conversations.
- `python -m IMPersona2.combine_dpo_datasets` — combine with filters on run
  name, rewrite model, or IMPersona version.

### 8. Train DPO

Runs LlamaFactory DPO on the assistant (`assistant.base`) on 2× H100 with
DeepSpeed ZeRO-3, via `run_dpo_worker.py`. Continue from a previous round's
adapter to iterate. Defaults: LR 3e-6, β 0.1, LoRA rank 32 / α 64, `cutoff_len`
4096 (8192 ran out of memory for the default assistant).

```bash
python run_dpo_worker.py --dataset eval_output/dpo_combined.json \
    --adapter adapter-dpo-r2 --init-adapter adapter-dpo-r1
```

#### Running steps 5–8 on autopilot

The orchestrator runs eval → DPO data → DPO train for N rounds entirely on
Modal, checkpointing state so re-running the same command resumes:

```bash
modal run modal_app.py::run_pipeline_local --config-path configs/dpo_pipeline.yaml
```

Edit `configs/dpo_pipeline.yaml` for rounds, trials per scenario, the
IMPersona adapter, and training hyperparameters.

### 9. Memory index

Builds the retrieval corpus for the arena's memory-search baselines from the
same conversations the IMPersona was trained on. It writes one chunk per
conversation session (long threads are split at their largest time gap),
then embeds the chunks with the `embeddings` model (~$0.30 for a
3k-chunk corpus). Output: `.memory_corpus/`.

The underlying scripts are `scripts/build_memory_corpus.py` and
`scripts/build_memory_index.py`.

### 10. Arena

A blind head-to-head chat in the terminal. Each battle draws two of four
configs at random. You chat normally, then vote `/a`, `/b`, `/tie`, or `/bad`.

1. **`untrained`** — the untrained assistant (floor)
2. **`untrained-memory`** — the untrained assistant with a search tool over your messages
3. **`dpo`** — the trained assistant (Modal)
4. **`large-memory`** — a larger model with the same search tool (scale + retrieval)

The untrained and large models are `arena.baseline` and `arena.large_baseline`
in `configs/models.yaml`, served via OpenRouter.

```bash
python arena.py --dpo-adapter <your-dpo-adapter>
python arena.py --models untrained,untrained-memory   # skip Modal
python scripts/decode_arena.py                        # win rates
```

Verdicts are base64-encoded into `.arena_data/*.bin` so you don't see
standings while collecting.

## Configuration

| File | What it controls |
|---|---|
| `configs/models.yaml` | Which model each task uses |
| `.env` | API keys (plus optional one-off model overrides) |
| `persona_config.json` | Your name and nicknames (written by step 3) |
| `IMPersona2/experiment_config.py` | Parsing thresholds, filters, generation prompts |
| `IMPersona2/eval_scenarios.json` | The 130 rollout scenarios |
| `configs/dpo_pipeline.yaml` | Iterative DPO orchestrator |

## Citation

```bibtex
@misc{shi2026assistant,
  author       = {Shi, Quan},
  title        = {Impersona-Env: User Models as Environments for Model Personalization},
  year         = {2026},
  month        = {September},
  howpublished = {\url{https://benshi34.github.io/#/blog/impersona-env}},
  note         = {Blog post}
}

@article{shi2025impersona,
  title   = {IMPersona: Evaluating Individual Level LM Impersonation},
  author  = {Shi, Quan and Jimenez, Carlos E. and Dong, Stephen and Seo, Brian
             and Yao, Caden and Kelch, Adam and Narasimhan, Karthik},
  journal = {arXiv preprint arXiv:2504.04332},
  year    = {2025}
}
```
