"""Backward-compatible serialization for artifacts that gained optional fields.

Saved artifacts are linked by SHA-256 fingerprints of their canonical JSON
(``fingerprint_json(model.model_dump(mode="json"))``). Adding a field with a
default to an artifact model would add ``"field": null`` to every re-loaded
historical artifact and silently change its fingerprint, so every downstream
lineage check would report the old run as stale.

Models listing a field in ``_omit_when_none`` leave it out of their serialized
form while it is ``None``: historical artifacts serialize exactly as before
(identical fingerprints), and new artifacts that set the field carry it.
"""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, model_serializer


class OmitUnsetOptionalFields(BaseModel):
    """Base model that omits its listed optional fields while they are None."""

    _omit_when_none: ClassVar[tuple[str, ...]] = ()

    @model_serializer(mode="wrap")
    def _omit_unset_optional_fields(self, handler):
        data = handler(self)
        if isinstance(data, dict):
            for name in self._omit_when_none:
                if name in data and data[name] is None:
                    del data[name]
        return data
