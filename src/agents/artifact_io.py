"""Crash-safe JSON artifact writes that never silently destroy earlier outputs.

Two policies are provided:

* ``write_text_no_overwrite`` creates a new file atomically and raises
  ``FileExistsError`` if the target already exists. Used for run artifacts
  (LangGraph runs, resumes, targeted recoveries) that must never replace one
  another.
* ``write_text_preserving_previous`` replaces a stage's canonical output file
  (for example ``data/processed/critic_review.json``) atomically, but first
  keeps the previous version under ``<dir>/archive/`` so that an incomplete
  re-run (for example after an API quota failure) cannot erase an earlier
  successful result.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ARCHIVE_DIR_NAME = "archive"


def _write_temp(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory,
        prefix=f".{name}.", suffix=".tmp", delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temp_path.unlink(missing_ok=True)
            raise
    return temp_path


def _exclusive_copy(source: Path, target: Path) -> None:
    """Create ``target`` exclusively from ``source`` (fallback when links fail)."""
    with open(source, "rb") as reader, open(target, "xb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())


def _link_or_copy_exclusive(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except FileExistsError:
        raise
    except OSError:
        # Some file systems (FAT/exFAT, some network shares) do not support hard
        # links. Exclusive creation keeps the no-overwrite guarantee.
        _exclusive_copy(source, target)


def write_text_no_overwrite(path: Path, text: str) -> Path:
    """Atomically create ``path``; raise ``FileExistsError`` if it already exists."""
    path = Path(path)
    temp_path = _write_temp(path.parent, path.name, text)
    try:
        _link_or_copy_exclusive(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def archive_path_for(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = path.suffix or ".json"
    return path.parent / ARCHIVE_DIR_NAME / f"{path.stem}-{stamp}-{uuid4().hex[:8]}{suffix}"


def write_text_preserving_previous(path: Path, text: str) -> Path | None:
    """Atomically replace ``path`` after archiving any existing version.

    Returns the archive path of the previous version, or ``None`` when no
    earlier file existed. The new content is fully written and synced before
    the previous version is touched, so a failure never leaves a partial file.
    """
    path = Path(path)
    temp_path = _write_temp(path.parent, path.name, text)
    archived = None
    try:
        if path.exists():
            archived = archive_path_for(path)
            archived.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy_exclusive(path, archived)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return archived
