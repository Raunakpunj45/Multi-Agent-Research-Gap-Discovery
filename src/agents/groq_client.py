"""Groq adapter that serves the agents' existing structured-generation calls.

Agents call ``client.models.generate_content(model=..., contents=...,
config=GenerateContentConfig(...))`` and read ``response.parsed``. This module
maps that call onto a Groq chat completion with a JSON-schema
``response_format`` and returns the same shapes:

* a Pydantic ``response_schema`` class yields a validated model instance, or
  ``None`` when the output does not validate (as the Gemini SDK does);
* a ``dict`` schema (``response_json_schema`` / dict ``response_schema``)
  yields the decoded JSON object, which the caller validates itself.

Strict mode (constrained decoding) is used for models that support it. Strict
mode needs every property listed in ``required``. Optional scalar properties
without a default are therefore sent as nullable, and ``null`` values for
them are removed again before validation, so callers see exactly the shapes
they saw before.

Errors are re-raised as ``GroqAPIError`` with a sanitized message that the
shared retry classifier in ``paper_analysis_agent`` understands (HTTP status,
"per day" for daily limits, retry delay). The SDK's own retries are disabled
so that the agents' bounded retry and daily-quota stop logic stays in charge.
The API key is never included in messages, logs or ``repr``.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

GROQ_KEY_ENV = "GROQ_API_KEY"
REASONING_EFFORT_ENV = "GROQ_REASONING_EFFORT"
REASONING_EFFORTS = ("low", "medium", "high")

# ---------------------------------------------------------------------------
# Tokens-per-minute (TPM) budget
#
# Groq rejects a request whose requested tokens exceed the model's TPM limit
# (HTTP 413) and throttles once the per-minute total is used (HTTP 429). Groq
# does not document whether the reserved max_completion_tokens counts toward
# the requested tokens, so every request is sized so that the estimated input
# PLUS its completion cap fits the budget; that avoids a 413 either way.
#
# GROQ_TPM_LIMIT[_<MODEL>]           TPM limit; default 8000 (Groq's base limit
#                                    for openai/gpt-oss-*). 0 disables the
#                                    preflight check, the cap and pacing.
# GROQ_MAX_COMPLETION_TOKENS[_<MODEL>]  completion cap ceiling; default 4096.
# GROQ_CHARS_PER_TOKEN               estimator ratio; default 4.0 (Groq counted
#                                    ~4.4 characters per token on this project's
#                                    prompts, so 4.0 over-estimates).
# <MODEL> is the model ID upper-cased with non-alphanumerics as "_", e.g.
# GROQ_TPM_LIMIT_OPENAI_GPT_OSS_120B.
# ---------------------------------------------------------------------------
TPM_LIMIT_ENV = "GROQ_TPM_LIMIT"
MAX_COMPLETION_ENV = "GROQ_MAX_COMPLETION_TOKENS"
CHARS_PER_TOKEN_ENV = "GROQ_CHARS_PER_TOKEN"
DEFAULT_TPM_LIMIT = 8000
DEFAULT_MAX_COMPLETION_TOKENS = 4096
MIN_COMPLETION_TOKENS = 1024
TPM_SAFETY_FACTOR = 0.9
DEFAULT_CHARS_PER_TOKEN = 4.0
MESSAGE_OVERHEAD_TOKENS = 16
RATE_WINDOW_SECONDS = 60.0

# Models documented by Groq as supporting strict structured outputs.
STRICT_SCHEMA_MODELS = frozenset({
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "meta-models/muse-spark-open",
})

# JSON Schema keywords kept in strict mode. Validation-only keywords (default,
# minItems, pattern, ...) are dropped; Pydantic re-validates every response.
_STRICT_KEYWORDS = frozenset({
    "type", "properties", "required", "items", "enum", "anyOf", "$ref", "$defs",
    "description", "title", "additionalProperties",
})


class GroqAPIError(RuntimeError):
    """Sanitized provider error carrying the fields the retry classifier reads."""

    def __init__(self, message: str, *, status_code: int | None = None,
                 retry_after_seconds: float | None = None,
                 error_code: str | None = None):
        super().__init__(message)
        self.code = status_code
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        # Provider error code, e.g. "json_validate_failed" (schema-invalid output).
        self.error_code = error_code


class GroqRequestTooLargeError(GroqAPIError):
    """Raised before sending: the request cannot fit the model's TPM budget.

    Status 413 and a message without retry keywords make it non-retryable for
    every agent: resending an identical request can never succeed.
    """

    def __init__(self, message: str, *, estimated_input_tokens: int, budget_tokens: int):
        super().__init__(message, status_code=413)
        self.estimated_input_tokens = estimated_input_tokens
        self.budget_tokens = budget_tokens


class GroqOutputTruncatedError(GroqAPIError):
    """The completion stopped at max_completion_tokens; its JSON is incomplete."""

    def __init__(self, message: str, *, max_completion_tokens: int | None):
        super().__init__(message, status_code=None)
        self.max_completion_tokens = max_completion_tokens


def _kilo(tokens: int) -> str:
    """Render a token count without bare 3-digit numbers (e.g. "0.5k").

    The shared retry classifier treats a standalone 429/500/502/503/504 in an
    error message as retryable, so raw counts never appear in these messages.
    """
    return f"{tokens / 1000:.1f}k"


def _model_env_name(base: str, model: str) -> str:
    return f"{base}_{re.sub(r'[^A-Za-z0-9]+', '_', model).strip('_').upper()}"


def _env_number(base: str, model: str, default: float) -> float:
    for name in (_model_env_name(base, model), base):
        value = os.getenv(name)
        if value is not None and value.strip():
            try:
                number = float(value)
            except ValueError as error:
                raise ValueError(f"{name} must be a number.") from error
            if number < 0:
                raise ValueError(f"{name} must not be negative.")
            return number
    return default


def tpm_limit(model: str) -> int:
    """Configured TPM limit for a model; 0 means unmanaged."""
    return int(_env_number(TPM_LIMIT_ENV, model, DEFAULT_TPM_LIMIT))


def tpm_budget(model: str) -> int | None:
    """Tokens one request (input + completion cap) may use, or None if unmanaged."""
    limit = tpm_limit(model)
    return int(limit * TPM_SAFETY_FACTOR) if limit > 0 else None


def max_completion_tokens(model: str) -> int:
    return max(1, int(_env_number(MAX_COMPLETION_ENV, model, DEFAULT_MAX_COMPLETION_TOKENS)))


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for text (no tokenizer download needed)."""
    ratio = float(os.getenv(CHARS_PER_TOKEN_ENV) or DEFAULT_CHARS_PER_TOKEN)
    if ratio <= 0:
        raise ValueError(f"{CHARS_PER_TOKEN_ENV} must be positive.")
    return math.ceil(len(text or "") / ratio)


def estimate_request_tokens(request: dict) -> int:
    """Estimated input tokens of a chat request, including its JSON schema."""
    tokens = 0
    for message in request.get("messages", []):
        tokens += estimate_tokens(str(message.get("content") or "")) + MESSAGE_OVERHEAD_TOKENS
    if "response_format" in request:
        tokens += estimate_tokens(json.dumps(request["response_format"], ensure_ascii=False))
    return tokens


class _TokenWindow:
    """Sliding 60-second token window shared by every client using a model.

    acquire() waits until the tokens used in the last minute plus the new
    reservation fit the budget; settle() replaces the reservation with the
    actual usage Groq reports (or 0 when the request failed).
    """

    def __init__(self, budget: int):
        self.budget = budget
        self.events: deque[list[float]] = deque()
        self.lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self.events and self.events[0][0] <= now - RATE_WINDOW_SECONDS:
            self.events.popleft()

    def acquire(self, tokens: int) -> list[float]:
        with self.lock:
            now = time.monotonic()
            while True:
                self._prune(now)
                used = sum(event[1] for event in self.events)
                if not self.events or used + tokens <= self.budget:
                    entry = [now, float(tokens)]
                    self.events.append(entry)
                    return entry
                wait = max(0.05, self.events[0][0] + RATE_WINDOW_SECONDS - now)
                time.sleep(wait)
                # Advance logically too, so a patched or coarse clock cannot
                # make this loop spin.
                now = max(time.monotonic(), now + wait)

    def settle(self, entry: list[float], tokens: float) -> None:
        with self.lock:
            entry[1] = float(max(0.0, tokens))


_WINDOWS: dict[str, _TokenWindow] = {}
_WINDOWS_LOCK = threading.Lock()


def _token_window(model: str, budget: int) -> _TokenWindow:
    with _WINDOWS_LOCK:
        window = _WINDOWS.get(model)
        if window is None or window.budget != budget:
            window = _WINDOWS[model] = _TokenWindow(budget)
        return window


@dataclass
class GroqResponse:
    parsed: Any
    text: str | None
    finish_reason: str | None = None


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def groq_key_names(group: str) -> tuple[str, str]:
    return (f"{GROQ_KEY_ENV}_{group}", GROQ_KEY_ENV)


def create_groq_client(group: str, *, http_client=None) -> "GroqStructuredClient":
    """Create an agent-owned Groq client; the key is read from the environment."""
    from . import llm_provider

    if group not in llm_provider.KEY_GROUPS:
        raise ValueError(f"Unknown LLM API key group: {group!r}.")
    llm_provider._load_env()
    names = groq_key_names(group)
    api_key = next((os.getenv(name) for name in names if os.getenv(name)), None)
    if not api_key:
        raise ValueError(
            f"Set {' or '.join(names)} to configure Groq access for {group}."
        )
    import groq

    sdk_client = groq.Groq(api_key=api_key, max_retries=0, http_client=http_client)
    return GroqStructuredClient(sdk_client)


class _Models:
    def __init__(self, owner: "GroqStructuredClient"):
        self._owner = owner

    def generate_content(self, *, model: str, contents, config=None) -> GroqResponse:
        return self._owner._generate(model, contents, config)


class GroqStructuredClient:
    """Exposes ``models.generate_content`` on top of the official Groq SDK."""

    provider = "groq"

    def __init__(self, sdk_client):
        self._sdk = sdk_client
        self.models = _Models(self)

    def __repr__(self) -> str:  # never show credentials
        return "GroqStructuredClient()"

    # -- request sizing, also used by agents to plan staged requests ---------

    def max_input_tokens(self, model: str, output_tokens: int | None = None) -> int | None:
        """Largest estimated input that still leaves room for the completion.

        ``output_tokens`` is the task's fixed output budget; without it the
        minimum completion room is reserved.
        """
        budget = tpm_budget(model)
        if budget is None:
            return None
        reserve = MIN_COMPLETION_TOKENS if output_tokens is None else int(output_tokens)
        return budget - reserve

    def estimate_input_tokens(self, *, model: str, contents, config=None) -> int:
        """Estimate exactly the request generate_content would send."""
        request, _ = self._build_request(model, contents, config)
        return estimate_request_tokens(request)

    def _generate(self, model: str, contents, config) -> GroqResponse:
        request, plan = self._build_request(model, contents, config)
        reservation = self._apply_token_budget(model, request)
        window_entry = None
        if reservation is not None:
            window_entry = _token_window(model, tpm_budget(model)).acquire(reservation)
        try:
            completion = self._create(request)
        except BaseException:
            if window_entry is not None:
                _token_window(model, tpm_budget(model)).settle(window_entry, 0)
            raise
        if window_entry is not None:
            usage = getattr(getattr(completion, "usage", None), "total_tokens", None)
            if isinstance(usage, (int, float)):
                _token_window(model, tpm_budget(model)).settle(window_entry, usage)
        choice = completion.choices[0]
        text = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            cap = request.get("max_completion_tokens")
            limit = _kilo(cap) if cap else "the model default"
            raise GroqOutputTruncatedError(
                f"Groq response for {model} stopped at max_completion_tokens "
                f"({limit}) before the structured output was complete. "
                f"Lower GROQ_REASONING_EFFORT or raise {MAX_COMPLETION_ENV}/"
                f"{TPM_LIMIT_ENV} for this model.",
                max_completion_tokens=cap,
            )
        parsed = plan.parse(text) if plan is not None else None
        return GroqResponse(parsed=parsed, text=text, finish_reason=finish_reason)

    def _apply_token_budget(self, model: str, request: dict) -> int | None:
        """Preflight: cap the completion so input + cap fits; refuse otherwise."""
        budget = tpm_budget(model)
        if budget is None:
            return None
        estimated = estimate_request_tokens(request)
        fixed = request.get("max_completion_tokens")
        if fixed is not None:
            # Fixed per-task output budget: reserve it exactly, never shrink it.
            cap = min(int(fixed), max_completion_tokens(model))
            if estimated + cap > budget:
                raise GroqRequestTooLargeError(
                    f"Groq request not sent: its estimated input ({_kilo(estimated)} tokens) "
                    f"plus the task's fixed output budget ({_kilo(cap)} tokens) exceeds "
                    f"the {_kilo(budget)}-token per-request budget for {model} "
                    f"({TPM_LIMIT_ENV} {_kilo(tpm_limit(model))} tokens per minute). "
                    "Resending it unchanged cannot succeed; the task's input must be "
                    "made smaller.",
                    estimated_input_tokens=estimated, budget_tokens=budget,
                )
            request["max_completion_tokens"] = cap
            return estimated + cap
        cap = min(max_completion_tokens(model), budget - estimated)
        if cap < MIN_COMPLETION_TOKENS:
            raise GroqRequestTooLargeError(
                f"Groq request not sent: its estimated input ({_kilo(estimated)} tokens) "
                f"leaves less than {_kilo(MIN_COMPLETION_TOKENS)} completion tokens within "
                f"the {_kilo(budget)}-token per-request budget for {model} "
                f"({TPM_LIMIT_ENV} {_kilo(tpm_limit(model))} tokens per minute). "
                "Resending it unchanged cannot succeed; reduce the request or raise "
                f"{TPM_LIMIT_ENV} for this model if your Groq plan allows more.",
                estimated_input_tokens=estimated, budget_tokens=budget,
            )
        request["max_completion_tokens"] = cap
        return estimated + cap

    def _build_request(self, model: str, contents, config) -> tuple[dict, "SchemaPlan | None"]:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("A Groq model ID is required.")
        if model.lower().startswith("gemini"):
            raise ValueError(
                f"Model {model!r} is a Gemini model but LLM_PROVIDER is groq; "
                "set a GROQ_MODEL_<ROLE> override or LLM_PROVIDER=gemini."
            )
        messages = []
        system = _as_text(getattr(config, "system_instruction", None))
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": _as_text(contents) or ""})
        request: dict[str, Any] = {"model": model, "messages": messages}

        temperature = getattr(config, "temperature", None)
        if temperature is not None:
            request["temperature"] = temperature

        max_output = getattr(config, "max_output_tokens", None)
        if isinstance(max_output, int) and not isinstance(max_output, bool) and max_output > 0:
            request["max_completion_tokens"] = max_output

        schema_source = getattr(config, "response_json_schema", None)
        if schema_source is None:
            schema_source = getattr(config, "response_schema", None)
        plan = None
        if schema_source is not None:
            plan = build_schema_plan(schema_source, model)
            request["response_format"] = plan.response_format
        elif getattr(config, "response_mime_type", None) == "application/json":
            request["response_format"] = {"type": "json_object"}

        effort = (os.getenv(REASONING_EFFORT_ENV) or "").strip().lower()
        if effort and model.startswith("openai/gpt-oss"):
            if effort not in REASONING_EFFORTS:
                raise ValueError(
                    f"{REASONING_EFFORT_ENV} must be one of {', '.join(REASONING_EFFORTS)}."
                )
            request["reasoning_effort"] = effort
        return request, plan

    def _create(self, request: dict):
        import groq

        try:
            return self._sdk.chat.completions.create(**request)
        except groq.APIStatusError as error:
            raise _status_error(error) from None
        except groq.APITimeoutError:
            raise GroqAPIError("Groq API request timed out (UNAVAILABLE).") from None
        except groq.APIConnectionError:
            raise GroqAPIError("Groq API connection failed (UNAVAILABLE).") from None


def _as_text(value) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(filter(None, (_as_text(item) for item in value)))
    parts = getattr(value, "parts", None)
    if parts:
        return "\n".join(filter(None, (getattr(part, "text", None) for part in parts)))
    text = getattr(value, "text", None)
    return text if isinstance(text, str) else str(value)


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

_ORG_ID = re.compile(r"\borg_[A-Za-z0-9]+")
_KEY_LIKE = re.compile(r"\bgsk_[A-Za-z0-9]+")
_LIMIT = re.compile(r"\b(requests|tokens) per (minute|hour|day)\s*\((\w+)\)", re.IGNORECASE)
_TRY_AGAIN = re.compile(r"try again in ([0-9][0-9hms.]*)", re.IGNORECASE)


def _clean(text: str) -> str:
    text = _KEY_LIKE.sub("[redacted]", _ORG_ID.sub("org_[redacted]", text or ""))
    return " ".join(text.split())


def _error_fields(error) -> tuple[str, str | None]:
    body = getattr(error, "body", None)
    inner = body.get("error", body) if isinstance(body, dict) else None
    if isinstance(inner, dict):
        message = inner.get("message") or ""
        code = inner.get("code") or inner.get("type")
        return str(message), (str(code) if code else None)
    return str(getattr(error, "message", "") or ""), None


def _retry_after(error) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", None)
    value = headers.get("retry-after") if headers is not None else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _status_error(error) -> GroqAPIError:
    status = getattr(error, "status_code", None)
    message, code = _error_fields(error)
    retry_after = _retry_after(error)
    if status == 429:
        limit = _LIMIT.search(message)
        if limit:
            kind = (f"{limit.group(1).lower()} per {limit.group(2).lower()} "
                    f"({limit.group(3).upper()})")
        elif re.search(r"\b(?:RPD|TPD)\b|per day|daily", message, re.IGNORECASE):
            kind = "a per day limit"
        else:
            kind = "a request rate limit"
        text = f"Groq rate limit exceeded (HTTP 429) on {kind}"
        wait = _TRY_AGAIN.search(message)
        if wait:
            text += f"; provider suggests retrying in {wait.group(1).rstrip('.')}"
        return GroqAPIError(text + ".", status_code=429, retry_after_seconds=retry_after)
    detail = _clean(message)[:300]
    text = f"Groq API error (HTTP {status})"
    if code:
        text += f" [{_clean(code)[:60]}]"
    if detail:
        text += f": {detail}"
    return GroqAPIError(text, status_code=status, retry_after_seconds=retry_after,
                        error_code=_clean(code)[:60] if code else None)


# ---------------------------------------------------------------------------
# Structured-output schema handling
# ---------------------------------------------------------------------------


@dataclass
class SchemaPlan:
    response_format: dict
    schema: dict
    strict: bool
    model_class: type[BaseModel] | None = None
    nullable_added: dict[int, set[str]] = field(default_factory=dict)

    def parse(self, text: str | None):
        """Decode, undo strict-mode nulls, then validate like the Gemini SDK."""
        if not text:
            return None
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return None
        _prune_added_nulls(data, self.schema, self.schema.get("$defs", {}),
                           self.nullable_added)
        if self.model_class is None:
            return data
        try:
            return self.model_class.model_validate(data)
        except ValidationError:
            return None


def build_schema_plan(schema_source, model: str) -> SchemaPlan:
    if isinstance(schema_source, type) and issubclass(schema_source, BaseModel):
        model_class, schema, name = (schema_source, schema_source.model_json_schema(),
                                     schema_source.__name__)
    elif isinstance(schema_source, dict):
        model_class, schema, name = None, copy.deepcopy(schema_source), "response"
    else:
        raise TypeError(
            f"Unsupported response schema type for Groq: {type(schema_source).__name__}."
        )
    name = re.sub(r"[^A-Za-z0-9_-]", "_", str(schema.get("title") or name))[:64] or "response"

    strict_schema, added, strict_ok = strict_json_schema(schema)
    if model in STRICT_SCHEMA_MODELS and strict_ok:
        return SchemaPlan(
            response_format={"type": "json_schema", "json_schema": {
                "name": name, "schema": strict_schema, "strict": True,
            }},
            schema=strict_schema, strict=True, model_class=model_class,
            nullable_added=added,
        )
    return SchemaPlan(
        response_format={"type": "json_schema", "json_schema": {
            "name": name, "schema": schema, "strict": False,
        }},
        schema=schema, strict=False, model_class=model_class,
    )


def _is_nullable(node: dict) -> bool:
    if node.get("type") == "null":
        return True
    if isinstance(node.get("type"), list) and "null" in node["type"]:
        return True
    return any(isinstance(b, dict) and _is_nullable(b) for b in node.get("anyOf", []))


def _is_container(node: dict) -> bool:
    return (node.get("type") in ("array", "object") or "properties" in node
            or "items" in node or "$ref" in node)


def strict_json_schema(schema: dict) -> tuple[dict, dict[int, set[str]], bool]:
    """Return (strict schema, nullable properties added, strict-compatible?).

    Every object gets ``additionalProperties: false`` and lists all of its
    properties as required. A property that was optional, has no default and
    is a non-nullable scalar becomes nullable; its ``null`` is removed again by
    ``SchemaPlan.parse`` so an absent value stays absent. Free-form objects
    (``dict[str, X]``) cannot be expressed in strict mode; the caller then uses
    best-effort mode instead.
    """
    root = copy.deepcopy(schema)
    added: dict[int, set[str]] = {}
    compatible = True

    def visit(node):
        nonlocal compatible
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        properties = node.get("properties")
        is_object = node.get("type") == "object" or properties is not None
        if is_object:
            if not properties or isinstance(node.get("additionalProperties"), dict):
                compatible = False
            else:
                required = set(node.get("required", []))
                for name, prop in list(properties.items()):
                    if (name not in required and isinstance(prop, dict)
                            and "default" not in prop and not _is_nullable(prop)
                            and not _is_container(prop)):
                        properties[name] = {"anyOf": [prop, {"type": "null"}]}
                        added.setdefault(id(node), set()).add(name)
                node["required"] = list(properties)
                node["additionalProperties"] = False
        for key in list(node):
            if key not in _STRICT_KEYWORDS:
                node.pop(key)
        for key in ("items", "anyOf"):
            if key in node:
                visit(node[key])
        for mapping_key in ("properties", "$defs"):
            for child in (node.get(mapping_key) or {}).values():
                visit(child)

    visit(root)
    return root, added, compatible


def _prune_added_nulls(value, node, defs, added) -> None:
    if not isinstance(node, dict):
        return
    if "$ref" in node:
        target = defs.get(str(node["$ref"]).rsplit("/", 1)[-1])
        _prune_added_nulls(value, target, defs, added)
        return
    for branch in node.get("anyOf", []):
        _prune_added_nulls(value, branch, defs, added)
    if isinstance(value, dict) and isinstance(node.get("properties"), dict):
        for name in added.get(id(node), ()):
            if name in value and value[name] is None:
                del value[name]
        for name, child in node["properties"].items():
            if name in value:
                _prune_added_nulls(value[name], child, defs, added)
    elif isinstance(value, list) and isinstance(node.get("items"), dict):
        for item in value:
            _prune_added_nulls(item, node["items"], defs, added)
