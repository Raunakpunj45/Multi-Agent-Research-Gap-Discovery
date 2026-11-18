"""Topic runs with stage checkpoints, automatic resume and dry-run estimates.

``run_topic`` is the one entry point for a researcher's topic (CLI and UI).
Every completed step is checkpointed in the run directory; running the same
topic with the same settings again resumes after the last completed step
instead of re-spending tokens. Every finished run (complete or stopped) writes
a deterministic Markdown report, ``report.md``, into its run directory.
``dry_run_topic`` estimates every stage's requests for the same run without
sending anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.agents.artifact_io import write_text_preserving_previous
from src.workflow.checkpoint import (
    DEFAULT_RUNS_DIR,
    CheckpointError,
    RunCheckpoint,
    default_run_dir,
    run_settings,
)
from src.workflow.dry_run import DryRunAssumptions, DryRunReport, estimate_run
from src.workflow.langgraph_app import (
    DEFAULT_MAX_PAPERS,
    DEFAULT_MAX_REVISIONS,
    build_research_graph,
    default_workflow_agents,
)
from src.workflow.report import render_report

REPORT_FILENAME = "report.md"


@dataclass
class TopicRunConfig:
    research_topic: str
    analysis_query: str | None = None
    max_results_per_source: int = 5
    minimum_papers: int = 2
    max_papers: int = DEFAULT_MAX_PAPERS
    max_revisions: int = DEFAULT_MAX_REVISIONS
    pdf_paths: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.research_topic or not self.research_topic.strip():
            raise ValueError("A research topic is required.")
        if self.minimum_papers < 2:
            raise ValueError("minimum_papers must be at least 2.")
        if self.max_papers < self.minimum_papers:
            raise ValueError("max_papers cannot be less than minimum_papers.")
        if self.max_revisions < 0:
            raise ValueError("max_revisions cannot be negative.")
        if self.max_results_per_source < 1:
            raise ValueError("max_results_per_source must be at least 1.")


@dataclass
class TopicRunResult:
    state: dict
    run_dir: Path
    resumed_from: str | None
    report_path: Path | None = None


def write_report(state: dict, run_dir: Path) -> Path:
    """Render the run's Markdown report into its run directory (previous kept)."""
    path = Path(run_dir) / REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_preserving_previous(path, render_report(state))
    return path


class RunInterrupted(RuntimeError):
    """A step failed; completed steps are checkpointed for automatic resume."""

    def __init__(self, run_dir: Path, last_completed: str | None, error: Exception):
        self.run_dir = run_dir
        self.last_completed = last_completed
        self.error = error
        where = f"after step '{last_completed}'" if last_completed else "before any step completed"
        super().__init__(
            f"The run stopped {where}: {type(error).__name__}: {error}. "
            f"Completed steps are saved in {run_dir}; run the same topic again to resume."
        )


def initial_state(config: TopicRunConfig) -> dict:
    config.validate()
    topic = config.research_topic.strip()
    return {
        "research_topic": topic,
        "analysis_query": (config.analysis_query or topic).strip(),
        "max_results_per_source": config.max_results_per_source,
        "minimum_papers": config.minimum_papers,
        "max_papers": config.max_papers,
        "max_revisions": config.max_revisions,
        "revision_count": 0,
        "pdf_paths": dict(config.pdf_paths),
        "run_lineage": [],
    }


def _resolve_run_dir(state: dict, run_dir: Path | None, runs_dir: Path) -> Path:
    return Path(run_dir) if run_dir else default_run_dir(run_settings(state), runs_dir)


def load_resumable_state(config: TopicRunConfig, *, run_dir: Path | None = None,
                         runs_dir: Path = DEFAULT_RUNS_DIR) -> tuple[dict, str | None, Path]:
    """The state a run would start from: its checkpoint if one exists."""
    fresh = initial_state(config)
    directory = _resolve_run_dir(fresh, run_dir, runs_dir)
    checkpoint = RunCheckpoint(directory)
    if not checkpoint.has_state():
        return fresh, None, directory
    state, stage = checkpoint.load()
    if run_settings(state) != run_settings(fresh):
        raise CheckpointError(
            f"The checkpoint in {directory} was made with different run settings."
        )
    # Local PDF overrides are not research settings; the current ones apply.
    state["pdf_paths"] = dict(config.pdf_paths)
    return state, stage, directory


class _ProgressCheckpoint(RunCheckpoint):
    """A RunCheckpoint that also reports every saved step to a callback."""

    def __init__(self, run_dir: Path, progress):
        super().__init__(run_dir)
        self._progress = progress

    def save(self, state: dict, stage: str) -> None:
        super().save(state, stage)
        try:
            self._progress(stage, state)
        except Exception:  # a display callback must never break the run
            pass


def run_topic(config: TopicRunConfig, *, run_dir: Path | None = None, fresh: bool = False,
              agents=None, runs_dir: Path = DEFAULT_RUNS_DIR,
              progress=None) -> TopicRunResult:
    """Run (or automatically resume) the workflow for a research topic.

    ``progress(stage, state)``, when given, is called after every checkpointed
    step and after every analyzed or failed paper (stage ``analyze:partial``).
    """
    start = initial_state(config)
    directory = _resolve_run_dir(start, run_dir, runs_dir)
    checkpoint = (_ProgressCheckpoint(directory, progress) if progress
                  else RunCheckpoint(directory))
    checkpoint.ensure_manifest(run_settings(start))
    if fresh:
        checkpoint.archive_and_reset()

    resumed_from = None
    if checkpoint.has_state():
        snapshot_sha256 = checkpoint.snapshot_sha256()
        state, resumed_from, _ = load_resumable_state(config, run_dir=directory)
        state["checkpoint_stage"] = resumed_from
        state["run_lineage"] = list(state.get("run_lineage") or []) + [{
            "action": "auto_resume",
            "resumed_after_step": resumed_from,
            "checkpoint_sha256": snapshot_sha256,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }]
    else:
        state = start

    graph = build_research_graph(agents or default_workflow_agents(), checkpoint=checkpoint)
    try:
        result = graph.invoke(
            state, config={"recursion_limit": 30 + 5 * int(state.get("max_revisions", 0))},
        )
    except Exception as error:
        last = None
        if checkpoint.has_state():
            try:
                _, last = checkpoint.load()
            except CheckpointError:
                last = None
        raise RunInterrupted(directory, last, error) from error
    report_path = write_report(result, directory)
    return TopicRunResult(state=result, run_dir=directory, resumed_from=resumed_from,
                          report_path=report_path)


def dry_run_topic(config: TopicRunConfig, *, run_dir: Path | None = None,
                  runs_dir: Path = DEFAULT_RUNS_DIR, vector_store=None,
                  assumptions: DryRunAssumptions | None = None) -> tuple[DryRunReport, str | None, Path]:
    """Estimate every stage of this topic's run without sending any request.

    Uses the run's checkpoint when one exists, so stages whose inputs are
    already saved are measured from the real requests they would send.
    """
    state, stage, directory = load_resumable_state(config, run_dir=run_dir, runs_dir=runs_dir)
    if assumptions is None:
        from src.agents.research_gap_detection_agent import DEFAULT_MAX_GAPS

        assumptions = DryRunAssumptions(
            papers=config.max_papers,
            gaps=DEFAULT_MAX_GAPS,
            max_revisions=config.max_revisions,
        )
    if vector_store is None and state.get("gap_detection") is not None:
        from src.embeddings import LocalEmbeddingModel
        from src.vectorstore import ChromaVectorStore

        vector_store = ChromaVectorStore.for_embedder(LocalEmbeddingModel())
    return estimate_run(state, assumptions=assumptions, vector_store=vector_store), stage, directory
