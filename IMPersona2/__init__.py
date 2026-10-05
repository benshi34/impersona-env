"""
Impersona-Env — user-model and assistant-training package
=========================================================

Builds a user model (IMPersona) from iMessage exports, rolls it out against
an assistant, and turns its private reactions into DPO preference pairs.

Modules:
    experiment_config     — Pipeline configuration dataclasses
    model_config          — Per-task model selection (configs/models.yaml)
    generate_dataset      — SFT dataset generation (parse → context → CoT → save)
    llm_annotator         — LLM-based annotation (relationship, goals, CoT)
    dataset_visualizer    — Interactive HTML visualizer for generated datasets
    add_prior_cot         — Expose prior-turn CoT in training history
    eval                  — IMPersona ↔ assistant rollouts + LLM judges
    style_judge           — Does the user model text like the real person?
    fact_recall           — Human-verified fact bank + fact-recall judge
    generate_dpo_dataset  — Disagreements → guideline → rewrite → preference pairs
    filter_dpo_dataset    — LLM-judge filter for low-quality pairs
    prepare_dpo_dataset   — alpaca → multi-turn sharegpt DPO conversion
    combine_dpo_datasets  — Merge DPO datasets across eval runs
    memory_search         — Retrieval corpus + dense index (arena baseline)
    utils                 — Shared utilities (vCard parsing, etc.)
"""
