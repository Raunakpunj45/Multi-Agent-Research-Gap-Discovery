"""Gemini client configuration scoped to an agent group."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai


GROUP_KEY_ENV = {
    "ANALYSIS": "GEMINI_API_KEY_ANALYSIS",
    "COMPARISON": "GEMINI_API_KEY_COMPARISON",
    "GAP": "GEMINI_API_KEY_GAP",
    "VALIDATION": "GEMINI_API_KEY_VALIDATION",
    "CRITIC": "GEMINI_API_KEY_CRITIC",
    "QUESTION": "GEMINI_API_KEY_QUESTION",
}

LEGACY_GROUP_KEY_ENV = {
    "ANALYSIS": "GEMINI_API_KEY_ANALYSIS",
    "COMPARISON": "GEMINI_API_KEY_COMPARISON_GAP",
    "GAP": "GEMINI_API_KEY_COMPARISON_GAP",
    "VALIDATION": "GEMINI_API_KEY_VALIDATION_CRITIC",
    "CRITIC": "GEMINI_API_KEY_VALIDATION_CRITIC",
    "QUESTION": "GEMINI_API_KEY_QUESTION",
}


def create_gemini_client(group: str):
    """Create an agent-owned Gemini client using its group key or legacy fallback."""
    try:
        group_key_name = GROUP_KEY_ENV[group]
    except KeyError as error:
        raise ValueError(f"Unknown Gemini API key group: {group!r}.") from error

    load_dotenv()
    legacy_group_key_name = LEGACY_GROUP_KEY_ENV[group]
    api_key = (
        os.getenv(group_key_name)
        or os.getenv(legacy_group_key_name)
        or os.getenv("GEMINI_API_KEY")
    )
    if not api_key:
        required_names = list(dict.fromkeys((
            group_key_name, legacy_group_key_name, "GEMINI_API_KEY",
        )))
        raise ValueError(
            f"Set {' or '.join(required_names)} to configure Gemini access for {group}."
        )
    return genai.Client(api_key=api_key)
