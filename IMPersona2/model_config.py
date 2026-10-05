"""
Per-task model selection, read from configs/models.yaml.

    from IMPersona2.model_config import model_for
    model_for("judge")                 # → "claude-opus-4-6"
    model_for("judge", "reasoning")    # → None
    model_for("user_model", "base")    # → "meta-llama/Llama-3.1-8B-Instruct"

Resolution order for each value: environment variable override (only for the
keys listed in ENV_OVERRIDES) → configs/models.yaml → built-in default.
"""

import os
from pathlib import Path

MODELS_CONFIG_FILE = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"

# Used when configs/models.yaml is missing or omits a key (e.g. inside a Modal
# container that wasn't built with the configs/ directory).
DEFAULTS = {
    "annotation": {"model": "gpt-5.5", "reasoning": "medium", "base_url": None},
    "user_model": {"base": "meta-llama/Llama-3.1-8B-Instruct", "template": None},
    "assistant": {"base": "Qwen/Qwen3.5-27B-FP8", "template": None, "thinking": False},
    "judge": {"model": "claude-opus-4-6", "reasoning": None, "base_url": None},
    "translate": {"model": "claude-opus-4-6", "reasoning": None, "base_url": None},
    "rewrite": {"model": "claude-opus-4-6", "base_url": None},
    "dpo_filter": {"model": "gpt-5.4", "reasoning": "medium", "base_url": None},
    "style_judge": {"model": "claude-opus-4-6", "reasoning": None, "base_url": None},
    "fact_propose": {"model": "claude-opus-4-6", "reasoning": None, "base_url": None},
    "fact_search": {"model": "claude-opus-4-6", "reasoning": None, "base_url": None},
    "fact_judge": {"model": "claude-opus-4-6", "reasoning": None, "base_url": None},
    "embeddings": {"model": "text-embedding-3-large"},
    "arena": {"baseline": "qwen/qwen3.5-27b", "large_baseline": "qwen/qwen3.5-397b-a17b"},
}

ENV_OVERRIDES = {
    ("annotation", "model"): "ANNOTATION_MODEL",
    ("judge", "model"): "JUDGE_MODEL",
    ("judge", "reasoning"): "JUDGE_REASONING",
    ("judge", "base_url"): "JUDGE_BASE_URL",
    ("translate", "model"): "TRANSLATE_MODEL",
    ("translate", "reasoning"): "TRANSLATE_REASONING",
}

_cache = None


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(MODELS_CONFIG_FILE.parent.parent / ".env", override=False)
    except Exception:
        pass


def load_models_config() -> dict:
    """Return configs/models.yaml as a dict (empty if missing or unreadable)."""
    global _cache
    if _cache is None:
        _cache = {}
        try:
            import yaml
            _cache = yaml.safe_load(MODELS_CONFIG_FILE.read_text()) or {}
        except (OSError, ImportError):
            pass
    return _cache


def template_for(role: str, model_name: str) -> str:
    """LlamaFactory chat template for a model.

    Uses the role's `template` from configs/models.yaml when it is set and
    `model_name` is that role's configured base model; otherwise detects the
    template from the model name.
    """
    override = model_for(role, "template")
    if override and model_name == model_for(role, "base"):
        return override
    from modal_app import _get_lf_template
    return _get_lf_template(model_name)


def model_for(task: str, key: str = "model"):
    """Model setting for a pipeline task (see configs/models.yaml)."""
    env_var = ENV_OVERRIDES.get((task, key))
    if env_var:
        _load_env()
        value = (os.environ.get(env_var) or "").strip()
        if value:
            return value

    section = load_models_config().get(task) or {}
    if key in section:
        return section[key]
    return DEFAULTS.get(task, {}).get(key)
