"""Deterministic fingerprints and lightweight provenance for saved artifacts."""
import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


class ArtifactProvenance(BaseModel):
    stage: str
    input_fingerprints: dict[str, str] = Field(default_factory=dict)


def fingerprint_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fingerprint_json(value: Any) -> str:
    """Hash canonical JSON so equivalent model input ordering is stable."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return fingerprint_bytes(payload)


def require_fingerprint(provenance: ArtifactProvenance | None,
                        name: str, expected: str, artifact: str) -> None:
    if provenance is None:
        raise ValueError(f"{artifact} is missing provenance; regenerate it before use.")
    actual = provenance.input_fingerprints.get(name)
    if actual is None:
        raise ValueError(
            f"{artifact} is missing the {name!r} input fingerprint; regenerate it before use."
        )
    if actual != expected:
        raise ValueError(
            f"{artifact} is stale: {name!r} input fingerprint does not match the current artifact."
        )


def require_recorded_fingerprint(provenance: ArtifactProvenance | None,
                                 name: str, artifact: str) -> str:
    if provenance is None:
        raise ValueError(f"{artifact} is missing provenance; regenerate it before use.")
    value = provenance.input_fingerprints.get(name)
    if value is None:
        raise ValueError(
            f"{artifact} is missing the {name!r} input fingerprint; regenerate it before use."
        )
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{artifact} has an invalid {name!r} SHA-256 fingerprint.")
    return value
