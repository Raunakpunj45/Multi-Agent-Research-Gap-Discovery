"""Central LLM provider and model configuration for every agent.

Provider selection (``LLM_PROVIDER``):
    ``groq`` (default) or ``gemini``.

Model selection, per agent role (first match wins):
    ``<PROVIDER>_MODEL_<ROLE>``  e.g. ``GROQ_MODEL_CRITIC``
    ``<PROVIDER>_MODEL``         e.g. ``GROQ_MODEL`` (all roles)
    the built-in default in ``DEFAULT_MODELS``

API keys, per agent key group (first match wins; values are never logged):
    groq:   ``GROQ_API_KEY_<GROUP>`` then ``GROQ_API_KEY``
    gemini: unchanged six-key scheme in ``gemini_config.create_gemini_client``

Every agent keeps building a ``GenerateContentConfig`` and reading
``response.parsed``; the Groq client adapter accepts that interface, so prompts,
schemas, validation, retry loops and evidence rules are provider-independent.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

PROVIDER_ENV = "LLM_PROVIDER"
DEFAULT_PROVIDER = "groq"
SUPPORTED_PROVIDERS = ("groq", "gemini")

# Key groups are shared with the original six-key Gemini configuration.
KEY_GROUPS = ("ANALYSIS", "COMPARISON", "GAP", "VALIDATION", "CRITIC", "QUESTION")

# Model roles: one per agent. Planner and the Single-LLM baseline use the
# ANALYSIS key group but have their own model role.
ROLE_KEY_GROUP = {
    "PLANNER": "ANALYSIS",
    "ANALYSIS": "ANALYSIS",
    "BASELINE": "ANALYSIS",
    "COMPARISON": "COMPARISON",
    "GAP": "GAP",
    "VALIDATION": "VALIDATION",
    "CRITIC": "CRITIC",
    "QUESTION": "QUESTION",
}

# Groq defaults: the two production models with strict (constrained-decoding)
# structured outputs. The high-volume per-paper extraction, query planning and
# per-gap evidence validation use the smaller model; gap detection, the critic
# and hypothesis generation use the larger one. Cross-paper comparison is
# deterministic on the default path (no model); COMPARISON applies only to the
# legacy LLM comparison. The Gemini defaults are the previously hard-coded models.
DEFAULT_MODELS = {
    "groq": {
        "PLANNER": "openai/gpt-oss-20b",
        "ANALYSIS": "openai/gpt-oss-20b",
        "BASELINE": "openai/gpt-oss-120b",
        "COMPARISON": "openai/gpt-oss-120b",
        "GAP": "openai/gpt-oss-120b",
        "VALIDATION": "openai/gpt-oss-20b",
        "CRITIC": "openai/gpt-oss-120b",
        "QUESTION": "openai/gpt-oss-120b",
    },
    "gemini": {
        "PLANNER": "gemini-3.5-flash",
        "ANALYSIS": "gemini-3.5-flash-lite",
        "BASELINE": "gemini-3.5-flash",
        "COMPARISON": "gemini-3.5-flash",
        "GAP": "gemini-3.5-flash",
        "VALIDATION": "gemini-3.5-flash",
        "CRITIC": "gemini-3.5-flash",
        "QUESTION": "gemini-3.5-flash",
    },
}


def _load_env() -> None:
    """Load ``.env`` without overriding variables already set (patched in tests)."""
    load_dotenv()


def get_provider() -> str:
    """Return the configured provider name, validated."""
    _load_env()
    provider = (os.getenv(PROVIDER_ENV) or DEFAULT_PROVIDER).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported {PROVIDER_ENV}={provider!r}; "
            f"use one of {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return provider


def resolve_model(role: str, provider: str | None = None) -> str:
    """Model ID for an agent role under the configured (or given) provider."""
    if role not in ROLE_KEY_GROUP:
        raise ValueError(f"Unknown LLM model role: {role!r}.")
    provider = provider or get_provider()
    prefix = provider.upper()
    configured = os.getenv(f"{prefix}_MODEL_{role}") or os.getenv(f"{prefix}_MODEL")
    return (configured or DEFAULT_MODELS[provider][role]).strip()


def create_llm_client(group: str):
    """Create an agent-owned client for a key group under the configured provider."""
    if group not in KEY_GROUPS:
        raise ValueError(f"Unknown LLM API key group: {group!r}.")
    provider = get_provider()
    if provider == "gemini":
        from .gemini_config import create_gemini_client

        return create_gemini_client(group)
    from .groq_client import create_groq_client

    return create_groq_client(group)
