"""Fixed output-token budgets, one per LLM task.

Every structured LLM request declares the completion budget its task needs
(visible JSON plus the model's hidden reasoning tokens). The Groq adapter
reserves exactly this budget and refuses, before sending, any request whose
estimated input plus this budget does not fit the model's per-request token
budget. Inputs that are too large therefore fail fast and cheaply instead of
being sent with a shrunken completion cap that truncates the JSON.

Budgets are deliberately modest: a task that cannot answer within its budget
must be made smaller (fewer items, shorter fields), not given more tokens.

Override per task with ``LLM_OUTPUT_BUDGET_<TASK>`` (for example
``LLM_OUTPUT_BUDGET_GAP_DETECTION=2500``). Budgets never raise a provider
limit: the Groq preflight still checks input + budget against the model's
configured tokens-per-minute budget.

Gemini keeps its previous behaviour (no explicit ``max_output_tokens``),
because Gemini thinking models count thinking tokens differently; set
``LLM_OUTPUT_BUDGETS_FOR_GEMINI=1`` to apply the same budgets there.
"""
from __future__ import annotations

import os

from google.genai import types

# Tokens = visible structured output + reasoning headroom (gpt-oss, low effort).
TASK_OUTPUT_BUDGETS: dict[str, int] = {
    "planner": 1536,
    "paper_analysis": 2560,
    "comparison": 4096,
    "comparison_synthesis": 4096,
    "comparison_grounding": 1536,
    "gap_detection": 3072,
    # Default path: one GPT-OSS-20B request per gap assessing all its evidence.
    "evidence_gap": 3072,
    # Legacy per-chunk validation (EvidenceValidationAgent(mode="per_chunk")).
    "evidence_chunk": 1024,
    "evidence_batch": 2560,
    "evidence_joint": 2048,
    "evidence_adjudication": 1024,
    # Default path: one ACCEPT/REJECT request per validated gap.
    "critic_verdict": 2048,
    # Legacy critic modes (CriticAgent(review_mode="auto"/"single"/"split")).
    "critic": 3072,
    "critic_sub_review": 1536,
    "critic_gap_review": 2048,
    "research_questions": 2048,
    "baseline": 4096,
}

OUTPUT_BUDGET_ENV = "LLM_OUTPUT_BUDGET"
GEMINI_OPT_IN_ENV = "LLM_OUTPUT_BUDGETS_FOR_GEMINI"


def output_budget(task: str) -> int:
    """Completion-token budget for a task (environment override wins)."""
    if task not in TASK_OUTPUT_BUDGETS:
        raise ValueError(f"Unknown LLM task for output budget: {task!r}.")
    name = f"{OUTPUT_BUDGET_ENV}_{task.upper()}"
    value = os.getenv(name)
    if value is not None and value.strip():
        try:
            number = int(value)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer.") from error
        if number < 1:
            raise ValueError(f"{name} must be positive.")
        return number
    return TASK_OUTPUT_BUDGETS[task]


def _applies_to_provider() -> bool:
    from .llm_provider import get_provider

    if get_provider() == "groq":
        return True
    return (os.getenv(GEMINI_OPT_IN_ENV) or "").strip().lower() in ("1", "true", "yes")


def request_input_budget(client, model: str, task: str) -> int | None:
    """Largest input a token-managed client accepts with the task's output budget.

    Returns None for clients without request sizing (Gemini, simple fakes).
    """
    max_input = getattr(client, "max_input_tokens", None)
    if not callable(max_input) or not callable(getattr(client, "estimate_input_tokens", None)):
        return None
    try:
        budget = max_input(model, output_tokens=output_budget(task))
    except TypeError:  # clients that only reserve a minimum completion
        budget = max_input(model)
    if isinstance(budget, bool) or not isinstance(budget, int):
        return None
    return budget


def generation_config(task: str, **kwargs) -> types.GenerateContentConfig:
    """``GenerateContentConfig`` carrying the task's fixed output budget."""
    if "max_output_tokens" not in kwargs and _applies_to_provider():
        kwargs["max_output_tokens"] = output_budget(task)
    return types.GenerateContentConfig(**kwargs)
