# agent/config.py
"""
Mutable runtime configuration for the cognitive agent.

LLM settings are stored here so the agent can evolve them at runtime.
The `create_llm` helper builds a fresh LLM from the *current* config
each time a node executes — so config changes take effect immediately.
"""
from __future__ import annotations

from agent.logging_config import get_logger

log = get_logger("config")

# ── LLM configurations ──────────────────────────────────────────────────────
# Keyed by logical node name. The agent can change model and temperature.
LLM_CONFIG: dict[str, dict] = {
    "mind_llm":          {"model": "gemini-2.5-flash-lite", "temperature": 1.0},
    "mind_generate_llm": {"model": "gemini-2.5-flash-lite", "temperature": 0.7},
    "mind_update_llm":   {"model": "gemini-2.5-flash-lite", "temperature": 0.0},
    "brain_llm":         {"model": "gemini-2.5-flash",      "temperature": 1.0},
    "brain_update_llm":  {"model": "gemini-2.5-flash-lite", "temperature": 0.0},
}

# Models the agent is allowed to switch to
ALLOWED_MODELS = frozenset({
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
})

TEMP_MIN, TEMP_MAX = 0.0, 2.0


# ── Accessors ────────────────────────────────────────────────────────────────

def get_llm_config(key: str) -> dict:
    """Return a copy of the named LLM's current config."""
    if key not in LLM_CONFIG:
        raise ValueError(f"Unknown LLM key '{key}'. Available: {sorted(LLM_CONFIG)}")
    return dict(LLM_CONFIG[key])


def update_llm_config(
    key: str,
    model: str | None = None,
    temperature: float | None = None,
) -> tuple[dict, dict]:
    """
    Mutate an LLM config in place.  Returns (old_config, new_config).
    Raises ValueError on invalid input.
    """
    if key not in LLM_CONFIG:
        raise ValueError(f"Unknown LLM key '{key}'. Available: {sorted(LLM_CONFIG)}")

    old = dict(LLM_CONFIG[key])

    if model is not None:
        if model not in ALLOWED_MODELS:
            raise ValueError(f"Model '{model}' not allowed. Allowed: {sorted(ALLOWED_MODELS)}")
        LLM_CONFIG[key]["model"] = model

    if temperature is not None:
        if not (TEMP_MIN <= temperature <= TEMP_MAX):
            raise ValueError(f"Temperature {temperature} out of range [{TEMP_MIN}, {TEMP_MAX}]")
        LLM_CONFIG[key]["temperature"] = temperature

    new = dict(LLM_CONFIG[key])
    log.info("LLM config '%s' updated: %s → %s", key, old, new)
    return old, new


def list_llm_configs() -> dict[str, dict]:
    """Return a deep copy of all LLM configs."""
    return {k: dict(v) for k, v in LLM_CONFIG.items()}


def create_llm(config_key: str, structured_output_schema=None):
    """
    Create a fresh LLM instance from the *current* runtime config.

    Call this inside graph nodes (not at module scope) so that
    config changes made by the evolution system take effect immediately.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    cfg = get_llm_config(config_key)
    llm = ChatGoogleGenerativeAI(model=cfg["model"], temperature=cfg["temperature"])
    if structured_output_schema is not None:
        return llm.with_structured_output(structured_output_schema)
    return llm
