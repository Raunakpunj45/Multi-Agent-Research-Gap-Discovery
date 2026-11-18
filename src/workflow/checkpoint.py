"""Stage-wise checkpoints for topic runs, with automatic resume.

A run directory holds:

``manifest.json``
    The research settings the run was started with (topic, analysis query,
    paper limits, revision limit). A run is only ever resumed with exactly the
    same settings.
``state.json``
    The complete workflow state after the last completed step, together with
    the name of that step (``checkpoint_stage``). Paper analysis is also
    checkpointed after every paper (``analyze:partial``) so that an
    interrupted analysis never re-spends tokens on papers already analyzed.
``archive/``
    Every earlier completed-stage snapshot (never deleted), for auditing.

Snapshots are written atomically. Structured artifacts are stored with their
provenance; loading re-validates every artifact against its Pydantic model and
the workflow nodes re-check the lineage fingerprints exactly as in an
uninterrupted run, so a resumed run cannot silently combine stale artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from src.agents.artifact_io import _write_temp, write_text_preserving_previous
from src.agents.critic_agent import CriticResult
from src.agents.cross_paper_comparison_agent import ExcludedPaper, MultiPaperComparison
from src.agents.evidence_validation_agent import EvidenceValidationResult
from src.agents.paper_analysis_agent import PaperAnalysisResult
from src.agents.planner_agent import SearchPlan
from src.agents.research_gap_detection_agent import ResearchGapDetectionResult
from src.agents.research_question_agent import ResearchQuestionResult
from src.retrieval import PaperRecord

CHECKPOINT_VERSION = 1
PARTIAL_ANALYSIS_STAGE = "analyze:partial"
DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / "data" / "runs"

# Settings that define a run. A checkpoint is resumed only when all match.
RUN_SETTING_KEYS = (
    "research_topic", "analysis_query", "max_results_per_source",
    "minimum_papers", "max_papers", "max_revisions",
)

_MODEL_FIELDS = {
    "search_plan": SearchPlan,
    "comparison": MultiPaperComparison,
    "gap_detection": ResearchGapDetectionResult,
    "evidence_validation": EvidenceValidationResult,
    "critic_result": CriticResult,
    "research_questions": ResearchQuestionResult,
}
_MODEL_LIST_FIELDS = {
    "paper_analyses": PaperAnalysisResult,
    "excluded_papers": ExcludedPaper,
}
_PAPER_LIST_FIELDS = ("papers", "evaluation_papers")


class CheckpointError(ValueError):
    """A checkpoint cannot be used for this run."""


def jsonable(value):
    """JSON-compatible form of workflow state values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def serialize_state(state: dict) -> dict:
    return {key: jsonable(value) for key, value in state.items()}


def deserialize_state(data: dict) -> dict:
    """Rebuild typed workflow state; every artifact is re-validated."""
    state = dict(data)
    try:
        for key, model in _MODEL_FIELDS.items():
            if state.get(key) is not None:
                state[key] = model.model_validate(state[key])
        for key, model in _MODEL_LIST_FIELDS.items():
            if state.get(key) is not None:
                state[key] = [model.model_validate(item) for item in state[key]]
        for key in _PAPER_LIST_FIELDS:
            if state.get(key) is not None:
                state[key] = [PaperRecord(**item) for item in state[key]]
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"Checkpoint state is not valid: {error}") from error
    return state


def run_settings(state: dict) -> dict:
    return {key: state.get(key) for key in RUN_SETTING_KEYS}


def default_run_dir(settings: dict, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Deterministic directory for a topic and its run settings."""
    digest = hashlib.sha256(
        json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    slug = re.sub(r"[^a-z0-9]+", "-", (settings.get("research_topic") or "run").lower())
    slug = slug.strip("-")[:48].strip("-") or "run"
    return Path(runs_dir) / f"{slug}-{digest}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunCheckpoint:
    """Reads and writes one run directory's manifest and state snapshot."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / "manifest.json"
        self.state_path = self.run_dir / "state.json"

    # -- manifest ----------------------------------------------------------

    def ensure_manifest(self, settings: dict) -> None:
        """Create the manifest, or verify that it matches these settings."""
        if self.manifest_path.exists():
            manifest = self._read_json(self.manifest_path)
            if manifest.get("settings") != settings:
                raise CheckpointError(
                    f"Run directory {self.run_dir} belongs to different run settings; "
                    "use a different run directory or start fresh."
                )
            return
        self._atomic_write(self.manifest_path, {
            "checkpoint_version": CHECKPOINT_VERSION,
            "created_at": _now(),
            "settings": settings,
        })

    # -- state ---------------------------------------------------------------

    def has_state(self) -> bool:
        return self.state_path.exists()

    def load(self) -> tuple[dict, str]:
        """Return (typed state, last completed step) from the snapshot."""
        payload = self._read_json(self.state_path)
        if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise CheckpointError("Unsupported checkpoint version.")
        stage = payload.get("checkpoint_stage")
        if not isinstance(stage, str) or not stage:
            raise CheckpointError("Checkpoint does not record its last completed step.")
        return deserialize_state(payload.get("state") or {}), stage

    def snapshot_sha256(self) -> str | None:
        if not self.state_path.exists():
            return None
        return hashlib.sha256(self.state_path.read_bytes()).hexdigest()

    def save(self, state: dict, stage: str) -> None:
        """Persist the state after a completed step (previous snapshot archived)."""
        text = self._snapshot_text(state, stage)
        if stage == PARTIAL_ANALYSIS_STAGE:
            self._atomic_replace(self.state_path, text)
        else:
            write_text_preserving_previous(self.state_path, text)

    def archive_and_reset(self) -> None:
        """Start fresh: archive the current snapshot (never delete it)."""
        if self.state_path.exists():
            write_text_preserving_previous(self.state_path, "")
            self.state_path.unlink()

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _snapshot_text(state: dict, stage: str) -> str:
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_stage": stage,
            "saved_at": _now(),
            "state": serialize_state(state),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointError(f"Could not read checkpoint file {path}: {error}") from error
        if not isinstance(data, dict):
            raise CheckpointError(f"Checkpoint file {path} is not a JSON object.")
        return data

    @staticmethod
    def _atomic_replace(path: Path, text: str) -> None:
        temp = _write_temp(path.parent, path.name, text)
        try:
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def _atomic_write(self, path: Path, payload: dict) -> None:
        self._atomic_replace(path, json.dumps(payload, ensure_ascii=False, indent=2))
