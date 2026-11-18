"""Service layer for the Streamlit demo: one fresh, isolated topic run.

This module adds no second pipeline. It calls the same ``run_topic`` runner as
``experiments/run_langgraph.py`` (same agents, Groq configuration, evidence
validation, Critic and acceptance rules) and only

* allocates a NEW run directory and output path that did not exist before, so a
  demo run can never resume or overwrite an earlier run;
* bounds the cost-relevant settings (papers, results per source);
* forwards progress, and returns counts, gap rows and the Markdown report.
"""
from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.agents.artifact_io import write_text_no_overwrite
from src.workflow.checkpoint import CheckpointError, serialize_state
from src.workflow.dry_run import format_report
from src.workflow.runner import (
    RunInterrupted,
    TopicRunConfig,
    dry_run_topic,
    run_topic,
)
from src.workflow.summary import PaperCounts, gap_rows, paper_counts

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = ROOT_DIR / "data" / "runs"
DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

DEMO_TARGET_PAPERS = 10
# Cost bounds enforced for UI runs (each analyzed paper is one analysis request;
# more results per source means more retrieval calls and more candidates to rank).
MAX_TARGET_PAPERS = 15
MAX_RESULTS_PER_SOURCE = 10


@dataclass(frozen=True)
class DemoSettings:
    topic: str
    target_papers: int = DEMO_TARGET_PAPERS
    minimum_papers: int = 2
    max_results_per_source: int = 5

    def validate(self) -> None:
        if not self.topic or not self.topic.strip():
            raise ValueError("Enter a research topic.")
        if not 2 <= self.minimum_papers <= self.target_papers:
            raise ValueError("Minimum papers must be at least 2 and at most the target.")
        if not 2 <= self.target_papers <= MAX_TARGET_PAPERS:
            raise ValueError(f"Target papers must be between 2 and {MAX_TARGET_PAPERS}.")
        if not 1 <= self.max_results_per_source <= MAX_RESULTS_PER_SOURCE:
            raise ValueError(
                f"Results per source must be between 1 and {MAX_RESULTS_PER_SOURCE}.")

    def config(self) -> TopicRunConfig:
        self.validate()
        return TopicRunConfig(
            research_topic=self.topic.strip(),
            max_results_per_source=self.max_results_per_source,
            minimum_papers=self.minimum_papers,
            max_papers=self.target_papers,
        )


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    output_path: Path


@dataclass
class DemoResult:
    status: str                    # COMPLETE, STOPPED or INTERRUPTED
    stop_reason: str | None
    run_dir: Path
    output_path: Path | None
    report_path: Path | None
    report_markdown: str | None
    counts: PaperCounts | None
    gaps: list[dict] = field(default_factory=list)
    target_papers: int = DEMO_TARGET_PAPERS
    state: dict | None = None

    @property
    def target_reached(self) -> bool:
        return bool(self.counts) and self.counts.analyzed >= self.target_papers


def _slug(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:40].strip("-")
    return slug or "topic"


def new_run_paths(topic: str, *, runs_dir: Path | None = None,
                  processed_dir: Path | None = None, now=None,
                  token=None) -> RunPaths:
    """A run directory and output path that do not exist yet (never reused)."""
    runs_dir = runs_dir or DEFAULT_RUNS_DIR
    processed_dir = processed_dir or DEFAULT_PROCESSED_DIR
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(100):
        name = f"ui-{_slug(topic)}-{stamp}-{token or secrets.token_hex(3)}"
        run_dir = Path(runs_dir) / name
        output = Path(processed_dir) / f"langgraph_run-{name}.json"
        if not run_dir.exists() and not output.exists():
            return RunPaths(run_dir=run_dir, output_path=output)
        token = None  # a supplied token collided: fall back to random ones
    raise RuntimeError("Could not allocate a new, unused run directory.")


def estimate(settings: DemoSettings, *, runs_dir: Path | None = None) -> str:
    """Dry-run estimate of requests and tokens for a fresh run; sends nothing."""
    paths = new_run_paths(settings.topic, runs_dir=runs_dir)
    report, _, _ = dry_run_topic(settings.config(), run_dir=paths.run_dir)
    return format_report(report)


def run_demo(settings: DemoSettings, *, agents=None, progress=None,
             runs_dir: Path | None = None,
             processed_dir: Path | None = None) -> DemoResult:
    """Run the existing workflow once in a new, isolated run directory."""
    config = settings.config()
    paths = new_run_paths(config.research_topic, runs_dir=runs_dir,
                          processed_dir=processed_dir)
    if agents is None:
        from src.workflow.langgraph_app import default_workflow_agents

        agents = default_workflow_agents()
    try:
        outcome = run_topic(config, run_dir=paths.run_dir, fresh=True, agents=agents,
                            progress=progress)
    except RunInterrupted as error:
        return DemoResult(
            status="INTERRUPTED", stop_reason=str(error), run_dir=paths.run_dir,
            output_path=None, report_path=None, report_markdown=None, counts=None,
            target_papers=settings.target_papers)
    except CheckpointError as error:  # cannot happen for a new directory; fail loudly
        raise RuntimeError(f"Fresh run directory was not usable: {error}") from error
    if outcome.resumed_from is not None:
        raise RuntimeError("A demo run resumed an existing checkpoint; this must never happen.")
    state = outcome.state
    paths.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_no_overwrite(paths.output_path,
                            json.dumps(serialize_state(state), ensure_ascii=False, indent=2) + "\n")
    report = (outcome.report_path.read_text(encoding="utf-8")
              if outcome.report_path is not None else None)
    return DemoResult(
        status=state.get("workflow_status") or "UNKNOWN",
        stop_reason=state.get("stop_reason"),
        run_dir=outcome.run_dir, output_path=paths.output_path,
        report_path=outcome.report_path, report_markdown=report,
        counts=paper_counts(state), gaps=gap_rows(state),
        target_papers=settings.target_papers, state=state,
    )


STAGE_LABELS = {
    "retrieve": "Retrieved and ranked candidate papers",
    "analyze:partial": "Downloading and analyzing papers",
    "analyze": "Paper analysis finished",
    "compare": "Built the cross-paper evidence matrix",
    "detect_gaps": "Detected candidate research gaps",
    "validate": "Validated gaps against exact cited evidence",
    "critic": "Critic review finished",
    "revise": "Applied critic revisions",
    "questions": "Generated research questions for accepted gaps",
}


def describe_progress(stage: str, state: dict) -> str:
    """One status line for a checkpointed step (used by the UI)."""
    label = STAGE_LABELS.get(stage, f"Finished step: {stage}")
    if stage.startswith("stop_"):
        label = f"Stopped: {state.get('stop_reason') or stage}"
    if stage in ("retrieve",):
        return f"{label}: {len(state.get('papers') or [])} candidates."
    if stage.startswith("analyze"):
        counts = paper_counts(state)
        target = state.get("max_papers")
        return (f"{label}: {counts.analyzed}/{target} analyzed, {counts.failed} failed "
                f"(download {counts.failed_download}, analysis {counts.failed_analysis}).")
    if stage == "detect_gaps" and state.get("gap_detection") is not None:
        return f"{label}: {len(state['gap_detection'].gaps)} candidate gap(s)."
    return label + "."
