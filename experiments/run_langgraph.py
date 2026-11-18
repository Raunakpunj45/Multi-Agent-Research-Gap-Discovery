"""Execute the LangGraph research-gap workflow from the project root.

This runner uses the existing agents and therefore performs live retrieval and
model-backed work when invoked. Tests should use mocked WorkflowAgents instead.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.workflow.langgraph_app import (
    DEFAULT_MAX_PAPERS,
    DEFAULT_MAX_REVISIONS,
    ResearchWorkflowState,
    build_research_graph,
    default_workflow_agents,
)
from src.workflow.checkpoint import CheckpointError
from src.workflow.dry_run import estimate_run, format_report
from src.workflow.runner import (
    RunInterrupted,
    TopicRunConfig,
    dry_run_topic,
    run_topic,
)
from src.agents.artifact_io import (
    write_text_no_overwrite,
    write_text_preserving_previous,
)
from src.agents.artifact_provenance import (
    CANONICAL_JSON_SHA256,
    fingerprint_json,
    require_fingerprint,
    require_recorded_fingerprint,
)
from src.agents.cross_paper_comparison_agent import MultiPaperComparison
from src.agents.evidence_validation_agent import (
    EvidenceValidationResult,
    EvidenceValidationStatus,
    GapValidationStatus,
)
from src.agents.critic_agent import CriticResult
from src.agents.research_gap_detection_agent import ResearchGapDetectionResult
from src.evaluation.baselines import load_frozen_evaluation_corpus
from src.evaluation.identity import validate_exact_paper_ids
from src.retrieval import PaperRecord
from src.vectorstore.chroma_store import collection_name_for


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _evaluation_artifact_path(base: Path, prompt_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    token = uuid4().hex[:10]
    suffix = base.suffix or ".json"
    return base.with_name(f"{base.stem}-{prompt_id}-{stamp}-{token}{suffix}").resolve()


def _record_value(record, key: str):
    return record.get(key) if isinstance(record, dict) else getattr(record, key, None)


def _validate_evaluation_result(result: dict, expected_ids: list[str]) -> None:
    evaluation_directory = result.get("evaluation_chroma_dir")
    evaluation_collection = result.get("evaluation_chroma_collection")
    conventional_directory = (ROOT_DIR / "chroma_db").resolve()
    if (not isinstance(evaluation_directory, str)
            or Path(evaluation_directory).resolve() == conventional_directory):
        raise ValueError("Evaluation artifact must record an isolated Multi-Agent Chroma directory.")
    if (not isinstance(evaluation_collection, str) or not evaluation_collection
            or evaluation_collection == collection_name_for("BAAI/bge-small-en-v1.5")):
        raise ValueError("Evaluation artifact must record an isolated Multi-Agent Chroma collection.")

    papers = result.get("papers")
    if not isinstance(papers, list):
        raise ValueError("Evaluation workflow artifact is missing its frozen paper list.")
    validate_exact_paper_ids(
        [_record_value(paper, "source_id") or _record_value(paper, "doi") or "" for paper in papers],
        expected_ids,
        "Evaluation workflow paper artifact",
    )

    analyses = result.get("paper_analyses")
    if not isinstance(analyses, list) or len(analyses) != 6:
        raise ValueError("Evaluation workflow artifact must contain six paper analyses.")
    analyzed_ids = []
    for analysis in analyses:
        evidence = _record_value(analysis, "evidence")
        evidence_ids = [
            _record_value(item, "paper_id") or "" for item in (evidence or [])
        ]
        if not evidence_ids or len(set(evidence_ids)) != 1:
            raise ValueError("Each evaluation paper analysis must cite evidence from exactly one paper.")
        analyzed_ids.append(evidence_ids[0])
    validate_exact_paper_ids(analyzed_ids, expected_ids, "Evaluation workflow analysis artifact")

    for name, key in (
        ("comparison", "comparison"),
        ("gap detection", "gap_detection"),
        ("evidence validation", "evidence_validation"),
    ):
        artifact = result.get(key)
        paper_ids = _record_value(artifact, "paper_ids")
        if not isinstance(paper_ids, list):
            raise ValueError(f"Evaluation {name} artifact is missing paper_ids.")
        validate_exact_paper_ids(paper_ids, expected_ids, f"Evaluation {name} artifact")


def _serialize(payload) -> str:
    return json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n"


def _write_json_no_overwrite(output_path: Path, payload: dict) -> None:
    """Create a new artifact atomically; raises FileExistsError if it exists."""
    write_text_no_overwrite(output_path, _serialize(payload))


def _unique_artifact_path(base: Path, prompt_id: str, *reserved: Path) -> Path:
    blocked = {Path(item).resolve() for item in reserved}
    while True:
        candidate = _evaluation_artifact_path(base, prompt_id)
        if candidate not in blocked and not candidate.exists():
            return candidate


def _lineage_entry(mode: str, source: Path, **details) -> dict:
    """Record which saved artifact (by name and SHA-256) a run continued from."""
    return {
        "mode": mode,
        "source_artifact": source.name,
        "source_artifact_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }


# Downstream results that must not survive into a resumed run: they were
# computed from the inputs being replaced or re-reviewed and would otherwise be
# saved next to the new results as if they belonged to them.
_STALE_AFTER_EVIDENCE_RESUME = (
    "critic_result", "critic_input_fingerprints", "research_questions",
    "workflow_status", "stop_reason", "resume_from_critic",
)
_STALE_AFTER_CRITIC_RESUME = (
    "research_questions", "workflow_status", "stop_reason",
)


def _model_fingerprint(model) -> str:
    return fingerprint_json(model.model_dump(mode="json", exclude={"provenance"}))


def _prepare_evidence_resume(original_path: Path, evidence_path: Path) -> dict:
    """Restore saved upstream state and validate a replacement evidence artifact."""
    try:
        original = (json.loads(original_path.read_text(encoding="utf-8"))
                    if isinstance(original_path, Path) else original_path)
        comparison = MultiPaperComparison.model_validate(original["comparison"])
        gaps = ResearchGapDetectionResult.model_validate(original["gap_detection"])
        evidence = (EvidenceValidationResult.model_validate_json(evidence_path.read_text(encoding="utf-8"))
                    if isinstance(evidence_path, Path) else evidence_path)
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Could not load valid resume artifacts: {error}") from error

    require_recorded_fingerprint(
        comparison.provenance, "paper_analyses", "Saved comparison artifact",
    )
    gap_scheme = (gaps.provenance.fingerprint_schemes.get("comparison", CANONICAL_JSON_SHA256)
                  if gaps.provenance else None)
    if gap_scheme != CANONICAL_JSON_SHA256:
        raise ValueError("Saved gap-detection artifact does not declare the LangGraph canonical fingerprint scheme.")
    comparison_fingerprint = _model_fingerprint(comparison)
    require_fingerprint(gaps.provenance, "comparison", comparison_fingerprint,
                        "Saved gap-detection artifact")

    evidence_scheme = (evidence.provenance.fingerprint_schemes.get(
        "gap_detection", CANONICAL_JSON_SHA256,
    ) if evidence.provenance else None)
    if evidence_scheme != CANONICAL_JSON_SHA256:
        raise ValueError("Replacement evidence artifact is missing the canonical gap-detection fingerprint scheme.")
    gap_fingerprint = _model_fingerprint(gaps)
    require_fingerprint(evidence.provenance, "gap_detection", gap_fingerprint,
                        "Replacement evidence artifact")

    paper_ids = comparison.paper_ids
    def require_exact_ids(observed, expected, context):
        if (not isinstance(observed, list) or not isinstance(expected, list)
                or not expected or len(observed) != len(set(observed))
                or len(expected) != len(set(expected)) or set(observed) != set(expected)):
            raise ValueError(f"{context} paper IDs do not exactly match the saved comparison.")

    require_exact_ids(gaps.paper_ids, paper_ids, "Resume gap-detection artifact")
    require_exact_ids(evidence.paper_ids, paper_ids, "Resume evidence-validation artifact")
    expected_prompt_ids = original.get("evaluation_expected_paper_ids")
    if expected_prompt_ids is not None:
        validate_exact_paper_ids(paper_ids, expected_prompt_ids, "Resume evaluation paper IDs")

    gap_ids = [gap.gap_id for gap in gaps.gaps]
    validation_ids = [item.gap_id for item in evidence.gap_validations]
    if len(gap_ids) != len(set(gap_ids)) or len(validation_ids) != len(set(validation_ids)):
        raise ValueError("Resume artifacts contain duplicate gap IDs.")
    if set(gap_ids) != set(validation_ids):
        raise ValueError("Replacement evidence artifact gap IDs do not exactly match gap detection.")
    validations_by_id = {item.gap_id: item for item in evidence.gap_validations}
    clean_records = {}
    for gap_id, item in validations_by_id.items():
        for record in item.evidence_validations:
            if (record.status in (EvidenceValidationStatus.SUPPORTED,
                                  EvidenceValidationStatus.PARTIALLY_SUPPORTED,
                                  EvidenceValidationStatus.UNSUPPORTED)
                    and not record.assessment_error and not record.retrieval_error):
                clean_records[(gap_id, record.paper_id, record.chunk_id)] = True

    def is_resolved_diagnostic(message: str) -> bool:
        for (gap_id, paper_id, chunk_id) in clean_records:
            prefix = f"{gap_id}/{paper_id}/{chunk_id}:"
            if message.startswith(prefix) and any(
                marker in message for marker in (
                    "retrieval failed:", "exact chunk not found",
                    "ownership mismatch", "Gemini chunk assessment failed:",
                )
            ):
                return True
        for gap_id, validation in validations_by_id.items():
            if (validation.joint_assessment is not None
                    and message.startswith(f"{gap_id}: joint Gemini assessment failed:")):
                return True
        return False

    unresolved_retrieval = [message for message in evidence.retrieval_failures
                            if not is_resolved_diagnostic(message)]
    unresolved_semantic = [message for message in evidence.semantic_failures
                           if not is_resolved_diagnostic(message)]
    if unresolved_retrieval or unresolved_semantic:
        raise ValueError("Replacement evidence artifact still contains unresolved retrieval or assessment errors.")
    # Drop only stale diagnostics now contradicted by exact, successful records.
    evidence = evidence.model_copy(update={"retrieval_failures": [], "semantic_failures": []})

    for gap in gaps.gaps:
        item = validations_by_id[gap.gap_id]
        expected = [(ref.paper_id, chunk_id) for ref in gap.supporting_evidence
                    for chunk_id in ref.chunk_ids]
        actual = [(record.paper_id, record.chunk_id) for record in item.evidence_validations]
        if (not expected or len(expected) != len(set(expected))
                or len(actual) != len(set(actual)) or set(expected) != set(actual)):
            raise ValueError(f"{gap.gap_id} evidence references do not exactly match the gap definition.")
        if (item.status is GapValidationStatus.VALIDATION_INCOMPLETE
                or item.joint_assessment is None):
            raise ValueError(f"{gap.gap_id} evidence validation is incomplete.")
        if any(record.status not in (
                EvidenceValidationStatus.SUPPORTED,
                EvidenceValidationStatus.PARTIALLY_SUPPORTED,
                EvidenceValidationStatus.UNSUPPORTED,
        ) or record.assessment_error or record.retrieval_error
               for record in item.evidence_validations):
            raise ValueError(f"{gap.gap_id} has incomplete individual evidence assessments.")

    state = {key: value for key, value in dict(original).items()
             if key not in _STALE_AFTER_EVIDENCE_RESUME}
    state.update({
        "comparison": comparison,
        "gap_detection": gaps,
        "evidence_validation": evidence,
        "resume_from_evidence": True,
        "revision_count": int(original.get("revision_count", 0)),
        "max_revisions": int(original.get("max_revisions", 2)),
    })
    return state


def _prepare_critic_resume(run_path: Path) -> dict:
    """Restore a fingerprint-valid Critic result, including partial reviews."""
    try:
        original = json.loads(run_path.read_text(encoding="utf-8"))
        evidence = EvidenceValidationResult.model_validate(original["evidence_validation"])
        critic = CriticResult.model_validate(original["critic_result"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Could not load a valid saved Critic run: {error}") from error

    state = _prepare_evidence_resume(original, evidence)
    expected_fingerprints = {
        key: _model_fingerprint(state[key])
        for key in ("comparison", "gap_detection", "evidence_validation")
    }
    gap_ids = [gap.gap_id for gap in state["gap_detection"].gaps]
    review_ids = [review.gap_id for review in critic.gap_reviews]
    if (critic.input_fingerprints != expected_fingerprints
            or len(review_ids) != len(set(review_ids)) or set(review_ids) != set(gap_ids)):
        raise ValueError("Saved Critic result is stale or does not exactly cover the gaps.")
    for key in _STALE_AFTER_CRITIC_RESUME:
        state.pop(key, None)
    state.update({
        "critic_result": critic,
        "critic_input_fingerprints": expected_fingerprints,
        "resume_from_evidence": False,
        "resume_from_critic": True,
    })
    return state


def _run_topic_cli(args, parser, pdf_paths: dict) -> int:
    """Topic run with stage checkpoints and automatic resume."""
    config = TopicRunConfig(
        research_topic=args.topic,
        analysis_query=args.analysis_query,
        max_results_per_source=args.max_results_per_source,
        minimum_papers=args.minimum_papers,
        max_papers=args.max_papers or DEFAULT_MAX_PAPERS,
        max_revisions=args.max_revisions,
        pdf_paths=pdf_paths,
    )
    try:
        if args.dry_run:
            report, stage, run_dir = dry_run_topic(config, run_dir=args.run_dir)
            if stage:
                print(f"Using checkpoint in {run_dir} (last completed step: {stage}).")
            print(format_report(report))
            return 0
        outcome = run_topic(config, run_dir=args.run_dir, fresh=args.fresh,
                            agents=default_workflow_agents())
    except (CheckpointError, ValueError) as error:
        parser.error(str(error))
    except RunInterrupted as error:
        print(str(error))
        return 2
    result = outcome.state
    if outcome.resumed_from:
        print(f"Resumed after step '{outcome.resumed_from}' from {outcome.run_dir}.")
    output = (args.output or (ROOT_DIR / "data" / "processed" / "langgraph_run.json")).resolve()
    archived = write_text_preserving_previous(output, _serialize(result))
    if archived is not None:
        print(f"Previous output preserved: {archived}")
    print(f"Workflow status: {result.get('workflow_status', 'UNKNOWN')}")
    if result.get("stop_reason"):
        print(f"Stopped: {result['stop_reason']}")
    print(f"Checkpoints: {outcome.run_dir}")
    if outcome.report_path is not None:
        print(f"Report: {outcome.report_path}")
    print(f"Saved: {output}")
    return 0 if result.get("workflow_status") == "COMPLETE" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", help="Research topic to investigate.")
    parser.add_argument("--evaluation-prompt-id", choices=("PROMPT-001", "PROMPT-002", "PROMPT-003"),
                        help="Run with this frozen evaluation prompt and record its identity.")
    parser.add_argument("--manifest", type=Path, default=ROOT_DIR / "data" / "evaluation" / "dataset_manifest.jsonl")
    parser.add_argument("--analysis-query", help="Paper analysis query (defaults to --topic).")
    parser.add_argument("--max-results-per-source", type=int, default=5)
    parser.add_argument("--minimum-papers", type=int, default=2)
    parser.add_argument("--max-papers", type=int,
                        help=f"Most papers to analyze in a topic run (default {DEFAULT_MAX_PAPERS}).")
    parser.add_argument("--max-revisions", type=int, default=DEFAULT_MAX_REVISIONS,
                        help="Critic revision rounds (default 0: the default critic only "
                             "accepts or rejects gaps and never rewrites them).")
    parser.add_argument(
        "--paper-pdf", action="append", default=[], metavar="PAPER_ID=PATH",
        help="Use a local PDF for an exact retrieved paper ID; may be repeated.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate every stage's requests and token budgets; send nothing.")
    parser.add_argument("--run-dir", type=Path,
                        help="Checkpoint directory of a topic run (default: data/runs/<topic>-<hash>).")
    parser.add_argument("--fresh", action="store_true",
                        help="Start the topic run over; the earlier checkpoint is archived, not deleted.")
    parser.add_argument("--resume-from-evidence", type=Path,
                        help="Resume a saved LangGraph artifact at Critic using its saved comparison and gaps.")
    parser.add_argument("--resume-from-critic", type=Path,
                        help="Continue a saved complete Critic result through its revision or question path.")
    parser.add_argument("--resume-evidence-validation", type=Path,
                        help="Replacement evidence-validation artifact for --resume-from-evidence.")
    parser.add_argument(
        "--evaluation-chroma-dir", type=Path,
        default=ROOT_DIR / "data" / "evaluation" / "chroma_multi_agent",
        help="Separate Chroma persistence directory used only by evaluation-mode Multi-Agent ingestion.",
    )
    parser.add_argument(
        "--evaluation-chroma-collection", default="multi_agent_evaluation",
        help="Collection name in the evaluation-only Chroma directory.",
    )
    args = parser.parse_args()
    if bool(args.resume_from_evidence) != bool(args.resume_evidence_validation):
        parser.error("--resume-from-evidence and --resume-evidence-validation must be supplied together.")
    if args.resume_from_critic and (args.resume_from_evidence or args.resume_evidence_validation):
        parser.error("--resume-from-critic cannot be combined with evidence-resume options.")
    if args.resume_from_critic:
        if args.topic or args.evaluation_prompt_id or args.paper_pdf:
            parser.error("Resume mode uses saved run configuration; do not supply topic, prompt, or PDF overrides.")
        original_path = args.resume_from_critic.resolve()
        try:
            initial = _prepare_critic_resume(original_path)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        initial["run_lineage"] = list(initial.get("run_lineage") or []) + [
            _lineage_entry("resume_from_critic", original_path)
        ]
        if initial.get("evaluation_prompt_id"):
            agents = default_workflow_agents(
                evaluation_chroma_dir=Path(initial["evaluation_chroma_dir"]),
                evaluation_chroma_collection=initial["evaluation_chroma_collection"],
            )
        else:
            agents = default_workflow_agents()
        graph = build_research_graph(agents)
        result = graph.invoke(
            initial,
            config={"recursion_limit": 20 + 5 * initial["max_revisions"]},
        )
        serialized_result = _jsonable(result)
        if initial.get("evaluation_prompt_id"):
            _validate_evaluation_result(
                serialized_result, initial["evaluation_expected_paper_ids"],
            )
        output_base = args.output or original_path.with_name(f"{original_path.stem}-resumed.json")
        output_path = _unique_artifact_path(
            output_base, initial.get("evaluation_prompt_id", "resume"), original_path,
        )
        _write_json_no_overwrite(output_path, serialized_result)
        print(f"Workflow status: {result.get('workflow_status', 'UNKNOWN')}")
        if result.get("stop_reason"):
            print(f"Stopped: {result['stop_reason']}")
        print(f"Saved: {output_path}")
        return 0 if result.get("workflow_status") == "COMPLETE" else 1
    if args.resume_from_evidence:
        if args.topic or args.evaluation_prompt_id or args.paper_pdf:
            parser.error("Resume mode uses saved run configuration; do not supply topic, prompt, or PDF overrides.")
        original_path = args.resume_from_evidence.resolve()
        evidence_path = args.resume_evidence_validation.resolve()
        if original_path == evidence_path:
            parser.error("Original run and replacement evidence artifacts must be different files.")
        try:
            initial = _prepare_evidence_resume(original_path, evidence_path)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        initial["run_lineage"] = list(initial.get("run_lineage") or []) + [
            _lineage_entry(
                "resume_from_evidence", original_path,
                replacement_evidence_artifact=evidence_path.name,
                replacement_evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            )
        ]
        if initial.get("evaluation_prompt_id"):
            agents = default_workflow_agents(
                evaluation_chroma_dir=Path(initial["evaluation_chroma_dir"]),
                evaluation_chroma_collection=initial["evaluation_chroma_collection"],
            )
        else:
            agents = default_workflow_agents()
        graph = build_research_graph(agents)
        result = graph.invoke(
            initial,
            config={"recursion_limit": 20 + 5 * initial["max_revisions"]},
        )
        serialized_result = _jsonable(result)
        if initial.get("evaluation_prompt_id"):
            _validate_evaluation_result(
                serialized_result, initial["evaluation_expected_paper_ids"],
            )
        output_base = args.output or original_path.with_name(f"{original_path.stem}-resumed.json")
        output_path = _unique_artifact_path(
            output_base, initial.get("evaluation_prompt_id", "resume"),
            original_path, evidence_path,
        )
        _write_json_no_overwrite(output_path, serialized_result)
        print(f"Workflow status: {result.get('workflow_status', 'UNKNOWN')}")
        if result.get("stop_reason"):
            print(f"Stopped: {result['stop_reason']}")
        print(f"Saved: {output_path}")
        return 0 if result.get("workflow_status") == "COMPLETE" else 1

    if args.minimum_papers < 2:
        parser.error("--minimum-papers must be at least 2.")
    if args.max_revisions < 0:
        parser.error("--max-revisions cannot be negative.")
    if args.max_papers is not None and args.max_papers < args.minimum_papers:
        parser.error("--max-papers cannot be less than --minimum-papers.")
    if (args.run_dir or args.fresh) and args.evaluation_prompt_id:
        parser.error("--run-dir and --fresh apply to topic runs, not frozen evaluation runs.")

    pdf_paths = {}
    for entry in args.paper_pdf:
        paper_id, separator, raw_path = entry.partition("=")
        if not separator or not paper_id.strip() or not raw_path.strip():
            parser.error(f"Invalid --paper-pdf value {entry!r}; expected PAPER_ID=PATH.")
        pdf_paths[paper_id.strip()] = str(Path(raw_path.strip()).resolve())

    evaluation_prompt_id = args.evaluation_prompt_id
    canonical_text = None
    evaluation_papers = []
    expected_paper_ids = []
    if evaluation_prompt_id:
        corpus = load_frozen_evaluation_corpus(args.manifest)
        canonical_text = corpus.prompts[evaluation_prompt_id]["text"]
        if args.topic and args.topic != canonical_text:
            parser.error("--topic must exactly match the selected canonical evaluation prompt.")
        if args.analysis_query and args.analysis_query != canonical_text:
            parser.error("--analysis-query must exactly match the selected canonical evaluation prompt.")
        if pdf_paths:
            parser.error("--paper-pdf cannot override manifest PDF paths in frozen evaluation mode.")
        if args.minimum_papers not in (2, 6) or args.max_papers not in (None, 6):
            parser.error("Frozen evaluation mode requires exactly six papers.")
        expected_paper_ids = [paper.paper_id for paper in corpus.papers]
        validate_exact_paper_ids(expected_paper_ids, expected_paper_ids, "Evaluation manifest")
        evaluation_papers = [PaperRecord(
            title=paper.title or "",
            authors=paper.authors or [],
            year=paper.year,
            doi=(paper.external_ids or {}).get("doi"),
            abstract=None,
            source="frozen_evaluation_manifest",
            source_id=paper.paper_id,
            landing_url=None,
            pdf_url=None,
        ) for paper in corpus.papers]
        pdf_paths = {
            paper.paper_id: str((ROOT_DIR / paper.relative_path).resolve())
            for paper in corpus.papers
        }
        topic = canonical_text
    else:
        if not args.topic:
            parser.error("--topic is required unless --evaluation-prompt-id is selected.")
        return _run_topic_cli(args, parser, pdf_paths)
    initial: ResearchWorkflowState = {
        "research_topic": topic,
        "analysis_query": canonical_text or args.analysis_query or topic,
        "max_results_per_source": args.max_results_per_source,
        "minimum_papers": args.minimum_papers,
        "max_papers": args.max_papers,
        "max_revisions": args.max_revisions,
        "revision_count": 0,
        "pdf_paths": pdf_paths,
    }
    if evaluation_prompt_id:
        initial["evaluation_prompt_id"] = evaluation_prompt_id
        initial["canonical_prompt_text"] = canonical_text
        initial["evaluation_papers"] = evaluation_papers
        initial["evaluation_expected_paper_ids"] = expected_paper_ids
        initial["evaluation_chroma_dir"] = str(args.evaluation_chroma_dir.resolve())
        initial["evaluation_chroma_collection"] = args.evaluation_chroma_collection
        initial["minimum_papers"] = 6
        initial["max_papers"] = 6

    output_path = args.output or (ROOT_DIR / "data" / "processed" / (
        f"langgraph_run-{evaluation_prompt_id}.json" if evaluation_prompt_id else "langgraph_run.json"
    ))
    if evaluation_prompt_id:
        output_path = _evaluation_artifact_path(output_path, evaluation_prompt_id)
    output = output_path.resolve()
    if evaluation_prompt_id and output.exists():
        parser.error(f"Evaluation artifact already exists and will not be overwritten: {output}")

    if args.dry_run:
        print(format_report(estimate_run(initial)))
        return 0
    if evaluation_prompt_id:
        agents = default_workflow_agents(
            evaluation_chroma_dir=args.evaluation_chroma_dir,
            evaluation_chroma_collection=args.evaluation_chroma_collection,
        )
        graph = build_research_graph(agents)
    else:
        graph = build_research_graph()
    result = graph.invoke(
        initial,
        config={"recursion_limit": 20 + 5 * args.max_revisions},
    )
    serialized_result = _jsonable(result)
    if evaluation_prompt_id:
        _validate_evaluation_result(serialized_result, expected_paper_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    if evaluation_prompt_id:
        _write_json_no_overwrite(output, serialized_result)
        print(f"Workflow status: {result.get('workflow_status', 'UNKNOWN')}")
        if result.get("stop_reason"):
            print(f"Stopped: {result['stop_reason']}")
        print(f"Saved: {output}")
        return 0 if result.get("workflow_status") == "COMPLETE" else 1

    archived = write_text_preserving_previous(output, _serialize(result))
    if archived is not None:
        print(f"Previous output preserved: {archived}")

    print(f"Workflow status: {result.get('workflow_status', 'UNKNOWN')}")
    if result.get("stop_reason"):
        print(f"Stopped: {result['stop_reason']}")
    print(f"Saved: {output}")
    return 0 if result.get("workflow_status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
