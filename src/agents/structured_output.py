"""Structured-output contract and bounded schema repair for model responses.

A provider can reject or return a response that does not match the requested
JSON schema (for example Groq HTTP 400 ``json_validate_failed``: "missing
properties: 'overgeneralization'"). This module

* renders an explicit output contract (every required key with its type) from
  the SAME Pydantic model that produces the provider schema, so the prompt and
  the schema cannot disagree;
* recognizes schema-invalid responses, describes the precise problem, and
  builds one corrective follow-up request.

It never fills a missing field: a response is accepted only if it validates
against the full local model, otherwise the caller records the failure.
"""
from __future__ import annotations

from pydantic import BaseModel, ValidationError

MAX_SCHEMA_REPAIRS = 1


class StructuredOutputError(ValueError):
    """The model's structured output stayed schema-invalid after bounded repair."""


class MalformedStructuredOutput(ValueError):
    """A response carried no JSON that validates against the requested schema."""


def _type_name(prop: dict, defs: dict) -> str:
    if "$ref" in prop:
        return _object_name(defs[prop["$ref"].split("/")[-1]], defs)
    if "anyOf" in prop:
        names = [_type_name(option, defs) for option in prop["anyOf"]]
        return " or ".join(dict.fromkeys(names))
    kind = prop.get("type")
    if kind == "array":
        return f"array of {_type_name(prop.get('items', {}), defs)}"
    if kind == "object":
        return _object_name(prop, defs)
    return {"string": "string", "boolean": "boolean (true/false)", "integer": "integer",
            "number": "number", "null": "null"}.get(kind, str(kind or "value"))


def _object_name(schema: dict, defs: dict) -> str:
    required = set(schema.get("required", []))
    fields = []
    for name, prop in schema.get("properties", {}).items():
        marker = "" if name in required else " [optional]"
        fields.append(f"{name}: {_type_name(prop, defs)}{marker}")
    return "objects {" + "; ".join(fields) + "}"


def schema_contract(model: type[BaseModel]) -> str:
    """Explicit output contract listing every key of ``model`` with its type."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    required = set(schema.get("required", []))
    lines = ["Output contract: return ONE JSON object containing EVERY key below "
             "with the stated type. Never omit a required key, never add other keys, "
             "and never leave a required string empty; if there is nothing to report, "
             "say so explicitly in that field. Use [] for an empty list."]
    for name, prop in schema.get("properties", {}).items():
        marker = "required" if name in required else "optional, may be null"
        description = f" - {prop['description']}" if prop.get("description") else ""
        lines.append(f"- {name} ({_type_name(prop, defs)}; {marker}){description}")
    return "\n".join(lines)


def required_fields(model: type[BaseModel]) -> list[str]:
    return list(model.model_json_schema().get("required", []))


def describe_validation_error(error: ValidationError) -> str:
    parts = []
    for item in error.errors()[:8]:
        location = ".".join(str(part) for part in item.get("loc", ())) or "(root)"
        kind = item.get("type", "")
        if kind == "missing":
            parts.append(f"missing required field '{location}'")
        else:
            parts.append(f"{location}: {item.get('msg', kind)}")
    return "; ".join(parts) or str(error)


def schema_problem(error: Exception) -> str | None:
    """Precise description when ``error`` is a schema-invalid response, else None."""
    if isinstance(error, ValidationError):
        return describe_validation_error(error)
    if isinstance(error, MalformedStructuredOutput):
        return str(error)
    text = str(error)
    if (getattr(error, "error_code", None) == "json_validate_failed"
            or "json_validate_failed" in text):
        return text
    return None


def diagnose_unparsed(response, model: type[BaseModel]) -> MalformedStructuredOutput:
    """Explain why a response produced no parsed object (diagnosis only)."""
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        return MalformedStructuredOutput("the response contained no JSON output")
    try:
        model.model_validate_json(text)
    except ValidationError as error:
        return MalformedStructuredOutput(describe_validation_error(error))
    return MalformedStructuredOutput("the response JSON could not be parsed into the schema")


def repair_request(contents: str, problem: str, model: type[BaseModel]) -> str:
    """The original request plus a correction naming the rejected output's problem."""
    return (
        f"{contents}\n\n---\nYour previous response was rejected by schema validation: "
        f"{problem}\nReturn the complete JSON object again. Every required key must be "
        f"present with the correct type: {', '.join(required_fields(model))}. Base every "
        "value only on the supplied evidence; do not omit a field to avoid assessing it."
    )
