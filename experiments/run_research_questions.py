"""Generate research questions from saved critic-accepted gaps.

Run from the project root:
    python experiments/run_research_questions.py
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from pydantic import ValidationError

from src.agents import (
    CriticResult,
    EvidenceValidationResult,
    MultiPaperComparison,
    ResearchGapDetectionResult,
    ResearchQuestionAgent,
)
from src.agents.cross_paper_comparison_agent import CrossPaperComparisonAgent
from src.agents.paper_analysis_agent import PaperAnalysisResult
from src.agents.research_question_agent import GenerationStatus, ResearchQuestionResult
from src.agents.research_question_agent import ResearchQuestionAgent as QuestionAgent
from src.agents.critic_agent import CriticDecision, CriticReviewStatus
from src.agents.evidence_validation_agent import (
    EvidenceValidationStatus, GapSupportStatus, GapValidationStatus,
)
from src.agents.artifact_io import write_text_preserving_previous
from src.agents.artifact_provenance import (
    fingerprint_json,
    require_fingerprint,
    require_recorded_fingerprint,
)
from src.workflow.langgraph_app import _questions_evidence_eligible
from experiments.run_langgraph import (
    _evaluation_artifact_path, _lineage_entry, _model_fingerprint,
    _prepare_critic_resume, _validate_evaluation_result, _write_json_no_overwrite,
)

QUESTION_INCOMPLETE_STOP_REASON = "Research question/hypothesis generation was incomplete."

PROCESSED = ROOT_DIR / "data" / "processed"


def load_optional(path: Path, model):
    if not path.exists():
        return None, f"Missing input file: {path}", None
    payload = None
    try:
        payload = path.read_bytes()
        raw = json.loads(payload.decode("utf-8"))
        return model.model_validate(raw), None, hashlib.sha256(payload).hexdigest()
    except (json.JSONDecodeError, ValidationError, OSError) as error:
        fingerprint = hashlib.sha256(payload).hexdigest() if payload is not None else None
        return None, f"Could not load {path}: {error}", fingerprint


def _targeted_gap_inputs(artifact_path: Path, gap_id: str):
    """Load and validate a bundled LangGraph artifact before targeted generation."""
    try:
        raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load LangGraph artifact: {error}") from error

    if not isinstance(raw, dict) or not isinstance(raw.get("research_questions"), dict):
        raise ValueError(
            "Targeted question recovery requires a saved LangGraph artifact that "
            "contains a research-question result."
        )
    if raw.get("evaluation_prompt_id"):
        # Frozen-corpus artifacts must still satisfy the evaluation identity checks.
        _validate_evaluation_result(raw, raw.get("evaluation_expected_paper_ids"))
    state = _prepare_critic_resume(artifact_path)

    analyses = [PaperAnalysisResult.model_validate(item) for item in raw["paper_analyses"]]
    comparison_agent = CrossPaperComparisonAgent()
    _, paper_text = comparison_agent._prepare(analyses)
    comparison_contents = comparison_agent._build_contents(
        paper_text, len(analyses), state["research_topic"],
    )
    if (state["comparison"].provenance.input_fingerprints.get("paper_analyses")
            != fingerprint_json(comparison_contents)):
        raise ValueError("Saved comparison artifact is stale relative to the paper analyses.")

    critic = state["critic_result"]
    gaps = state["gap_detection"]
    validation = state["evidence_validation"]
    if critic.review_status is not CriticReviewStatus.COMPLETE or critic.errors:
        raise ValueError("Critic review must be complete and error-free for targeted question recovery.")
    if QuestionAgent._integrity_errors(critic, gaps, validation):
        raise ValueError("Saved Critic, gap, and evidence-validation records failed integrity checks.")

    gap_matches = [gap for gap in gaps.gaps if gap.gap_id == gap_id]
    review_matches = [review for review in critic.gap_reviews if review.gap_id == gap_id]
    validation_matches = [item for item in validation.gap_validations if item.gap_id == gap_id]
    if not (len(gap_matches) == len(review_matches) == len(validation_matches) == 1):
        raise ValueError(f"Saved artifacts do not contain exactly one {gap_id} record at each stage.")
    gap, review, checked = gap_matches[0], review_matches[0], validation_matches[0]
    if (review.review_status is not CriticReviewStatus.COMPLETE
            or review.decision is not CriticDecision.ACCEPT):
        raise ValueError(f"{gap_id} does not have a complete Critic ACCEPT decision.")
    if (checked.status is GapValidationStatus.VALIDATION_INCOMPLETE
            or checked.joint_assessment is None
            or checked.status is GapValidationStatus.UNSUPPORTED
            or checked.joint_assessment.status is GapSupportStatus.UNSUPPORTED
            or not any(item.status in (
                EvidenceValidationStatus.SUPPORTED,
                EvidenceValidationStatus.PARTIALLY_SUPPORTED,
            ) for item in checked.evidence_validations)):
        raise ValueError(f"{gap_id} is not eligible for question generation under evidence validation.")
    if not _questions_evidence_eligible(
            gaps.model_copy(update={"gaps": [gap]}),
            validation.model_copy(update={"gap_validations": [checked]})):
        raise ValueError(f"{gap_id} is not eligible for question generation under evidence validation.")

    try:
        previous = ResearchQuestionResult.model_validate(raw["research_questions"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Saved research-question result is invalid: {error}") from error
    all_ids = [gap_item.gap_id for gap_item in gaps.gaps]
    completed_ids = [item.gap_id for item in previous.gap_results]
    skipped_ids = [item.gap_id for item in previous.skipped_gaps]
    excluded_ids = [item.gap_id for item in previous.excluded_gaps]
    groups = (completed_ids, skipped_ids, excluded_ids)
    if (previous.research_topic != gaps.research_topic
            or any(len(group) != len(set(group)) for group in groups)
            or set(completed_ids) & set(skipped_ids)
            or set(completed_ids) & set(excluded_ids)
            or set(skipped_ids) & set(excluded_ids)
            or (set(completed_ids) | set(skipped_ids) | set(excluded_ids)) != set(all_ids)
            or gap_id in completed_ids or gap_id in excluded_ids
            or skipped_ids.count(gap_id) != 1
            or any(item.generation_status is not GenerationStatus.COMPLETE
                   for item in previous.gap_results)):
        raise ValueError("Saved research-question results do not form a valid partial result for targeted recovery.")

    fingerprints = {
        key: _model_fingerprint(state[key])
        for key in ("comparison", "gap_detection", "evidence_validation")
    }
    if critic.input_fingerprints != fingerprints:
        raise ValueError("Critic fingerprints do not match the saved upstream artifacts.")
    return (
        raw, state,
        critic.model_copy(update={"gap_reviews": [review]}),
        gaps.model_copy(update={"gaps": [gap]}),
        validation.model_copy(update={"gap_validations": [checked]}),
        fingerprints, previous,
    )


def _merge_target_result(previous: ResearchQuestionResult,
                         generated: ResearchQuestionResult, gap_id: str,
                         all_gap_ids: set[str] | list[str]) -> ResearchQuestionResult:
    """Replace only ``gap_id``'s skip record; every other gap result is kept as-is."""
    gap_order = list(all_gap_ids)
    result_ids = [item.gap_id for item in generated.gap_results]
    skipped_ids = [item.gap_id for item in generated.skipped_gaps]
    if (generated.excluded_gaps
            or any(item != gap_id for item in result_ids + skipped_ids)
            or len(result_ids) > 1 or len(skipped_ids) > 1
            or (bool(result_ids) == bool(skipped_ids))):
        raise ValueError("Targeted Research Question Agent returned an invalid per-gap result.")

    new_result = generated.gap_results[0] if result_ids else None
    if new_result is not None and new_result.generation_status is not GenerationStatus.COMPLETE:
        raise ValueError("Targeted generation returned a non-complete gap result without a skip record.")
    skipped = [item for item in previous.skipped_gaps if item.gap_id != gap_id]
    if skipped_ids:
        skipped.append(generated.skipped_gaps[0])
    results = list(previous.gap_results)
    if new_result is not None:
        results.append(new_result)
    position = {value: index for index, value in enumerate(gap_order)}
    results.sort(key=lambda item: position.get(item.gap_id, len(position)))
    skipped.sort(key=lambda item: position.get(item.gap_id, len(position)))
    # Defensive invariant: previously generated gaps must be carried over unchanged.
    preserved = [item for item in results if item.gap_id != gap_id]
    previous_by_id = {item.gap_id: item for item in previous.gap_results}
    if (len(preserved) != len(previous.gap_results)
            or any(item != previous_by_id.get(item.gap_id) for item in preserved)):
        raise ValueError("Targeted recovery would alter previously generated gap results.")
    result_ids_merged = [item.gap_id for item in results]
    excluded_ids = {item.gap_id for item in previous.excluded_gaps}
    errors = [item for item in previous.errors if not item.startswith(f"{gap_id}:")]
    errors.extend(item for item in generated.errors if item not in errors)
    missing_inputs = list(dict.fromkeys(previous.missing_inputs + generated.missing_inputs))
    complete = (
        set(result_ids_merged) | excluded_ids == set(gap_order)
        and len(result_ids_merged) + len(excluded_ids) == len(set(gap_order))
        and bool(result_ids_merged)
        and not skipped and not errors and not missing_inputs
        and all(item.generation_status is GenerationStatus.COMPLETE for item in results)
    )
    partial_marker = "INCOMPLETE PARTIAL GENERATION:"
    caveats = [item for item in previous.caveats if not item.startswith(partial_marker)]
    caveats.extend(item for item in generated.caveats
                   if item not in caveats and not item.startswith(partial_marker))
    if not complete:
        caveats.append(
            "INCOMPLETE PARTIAL GENERATION: this output is not a final report; "
            "one or more gaps were skipped or could not be generated."
        )
    return previous.model_copy(update={
        "generation_status": GenerationStatus.COMPLETE if complete else GenerationStatus.INCOMPLETE,
        "gap_results": results,
        "skipped_gaps": skipped,
        "errors": errors,
        "missing_inputs": missing_inputs,
        "caveats": list(dict.fromkeys(caveats)),
    })


def recover_questions_for_gap(artifact_path: Path, gap_id: str,
                              output_base: Path | None = None):
    """Regenerate one skipped gap and save a merged, no-overwrite run artifact.

    The source artifact is never modified. Previously generated gaps are carried
    over unchanged, and the new artifact records a lineage entry naming the
    source artifact (with its SHA-256), the targeted gap, and the outcome.
    """
    artifact_path = Path(artifact_path)
    raw, state, critic, gaps, validation, fingerprints, previous = _targeted_gap_inputs(
        artifact_path, gap_id,
    )
    gap_order = [item.gap_id for item in state["gap_detection"].gaps]
    generated = ResearchQuestionAgent().generate_questions(
        critic, gaps, validation, critic_input_fingerprints=fingerprints,
    )
    merged = _merge_target_result(previous, generated, gap_id, gap_order)
    target_skip = next((item for item in merged.skipped_gaps if item.gap_id == gap_id), None)
    raw["research_questions"] = merged.model_dump(mode="json")
    if (merged.generation_status is GenerationStatus.COMPLETE
            and raw.get("stop_reason") == QUESTION_INCOMPLETE_STOP_REASON):
        raw["workflow_status"] = "COMPLETE"
        raw["stop_reason"] = None
    elif merged.generation_status is not GenerationStatus.COMPLETE:
        raw["workflow_status"] = "STOPPED"
        raw["stop_reason"] = raw.get("stop_reason") or QUESTION_INCOMPLETE_STOP_REASON
    raw["run_lineage"] = list(raw.get("run_lineage") or []) + [_lineage_entry(
        "targeted_question_recovery", artifact_path,
        gap_id=gap_id,
        outcome="GENERATED" if target_skip is None else "NOT_GENERATED",
        reason=None if target_skip is None else target_skip.reason,
        preserved_gap_ids=[item.gap_id for item in previous.gap_results],
        resulting_generation_status=merged.generation_status.value,
    )]
    base = output_base or artifact_path.with_name(
        f"{artifact_path.stem}-questions-recovered.json",
    )
    label = raw.get("evaluation_prompt_id") or "recovery"
    while True:
        output_path = _evaluation_artifact_path(base, label)
        if output_path == artifact_path.resolve() or output_path.exists():
            continue
        try:
            _write_json_no_overwrite(output_path, raw)
            return output_path, merged
        except FileExistsError:
            continue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=PROCESSED / "multi_paper_comparison.json")
    parser.add_argument("--critic", type=Path, default=PROCESSED / "critic_review.json")
    parser.add_argument("--gaps", type=Path, default=PROCESSED / "research_gap_detection.json")
    parser.add_argument("--validation", type=Path, default=PROCESSED / "evidence_validation.json")
    parser.add_argument("--output", type=Path,
                        help="Output file path, or output name base for unique targeted-recovery artifacts.")
    parser.add_argument("--resume-from-langgraph", type=Path,
                        help="Bundled LangGraph artifact to recover one skipped question gap from.")
    parser.add_argument("--resume-research-questions", metavar="GAP_ID",
                        help="Generate only this skipped gap, e.g. GAP-004.")
    args = parser.parse_args()

    if bool(args.resume_from_langgraph) != bool(args.resume_research_questions):
        parser.error("--resume-from-langgraph and --resume-research-questions must be supplied together.")
    if args.resume_from_langgraph:
        try:
            output_path, result = recover_questions_for_gap(
                args.resume_from_langgraph.resolve(), args.resume_research_questions,
                args.output,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(f"Research question generation: {result.generation_status.value}")
        print(f"Targeted gap: {args.resume_research_questions}")
        for skipped in result.skipped_gaps:
            print(f"Skipped {skipped.gap_id}: {skipped.reason}")
        print(f"Saved: {output_path}")
        if result.generation_status is GenerationStatus.INCOMPLETE:
            raise SystemExit(1)
        return

    args.output = args.output or PROCESSED / "research_questions.json"

    critic, critic_error, _ = load_optional(args.critic, CriticResult)
    comparison, comparison_error, comparison_fingerprint = load_optional(
        args.comparison, MultiPaperComparison
    )
    gaps, gaps_error, gap_fingerprint = load_optional(args.gaps, ResearchGapDetectionResult)
    validation, validation_error, validation_fingerprint = load_optional(
        args.validation, EvidenceValidationResult
    )
    current_fingerprints = {}
    if comparison_fingerprint is not None:
        current_fingerprints["comparison"] = comparison_fingerprint
    if gap_fingerprint is not None:
        current_fingerprints["gap_detection"] = gap_fingerprint
    if validation_fingerprint is not None:
        current_fingerprints["evidence_validation"] = validation_fingerprint
    lineage_errors = []
    if comparison is not None:
        try:
            require_recorded_fingerprint(
                comparison.provenance, "paper_analyses", "Comparison artifact",
            )
        except ValueError as error:
            lineage_errors.append(str(error))
            comparison = None
    if comparison is None:
        gaps = None
        validation = None
    if gaps is not None and current_fingerprints.get("comparison"):
        try:
            require_fingerprint(
                gaps.provenance, "comparison",
                current_fingerprints["comparison"], "Gap detection artifact",
            )
        except ValueError as error:
            lineage_errors.append(str(error))
            gaps = None
    if validation is not None and gap_fingerprint is not None:
        try:
            require_fingerprint(
                validation.provenance, "gap_detection", gap_fingerprint,
                "Evidence validation artifact",
            )
        except ValueError as error:
            lineage_errors.append(str(error))
            validation = None
    result = ResearchQuestionAgent().generate_questions(
        critic, gaps, validation, critic_input_fingerprints=current_fingerprints
    )
    result.errors.extend(
        error for error in (
            critic_error, comparison_error, gaps_error, validation_error, *lineage_errors
        ) if error
    )

    if result.generation_status.value == "INCOMPLETE":
        print("Research question generation: INCOMPLETE — partial output, not a final report")
    else:
        print(f"Research question generation: {result.generation_status.value}")
    for skipped in result.skipped_gaps:
        print(f"Skipped {skipped.gap_id}: {skipped.reason}")
    for excluded in result.excluded_gaps:
        print(f"Excluded {excluded.gap_id}: {excluded.reason}")
    for gap in result.gap_results:
        print(f"\n[{gap.gap_id}] {gap.gap_title}")
        for question in gap.research_questions:
            print(f"  Research question: {question.question}")
            for ref in question.source_evidence:
                print(f"    evidence: {ref.paper_id} -> {ref.chunk_ids}")
        if gap.hypothesis is not None:
            print(f"  Alternative hypothesis: {gap.hypothesis.alternative_hypothesis}")
            print(f"  Null hypothesis: {gap.hypothesis.null_hypothesis}")
        else:
            print(f"  Hypothesis: {gap.hypothesis_status.value} — {gap.hypothesis_limitation}")
    for error in result.errors:
        print(f"Error: {error}")

    archived = write_text_preserving_previous(
        args.output, result.model_dump_json(indent=2) + "\n",
    )
    if archived is not None:
        print(f"Previous output preserved: {archived}")
    print(f"Saved: {args.output}")
    if result.generation_status.value == "INCOMPLETE":
        print("Research question generation incomplete; saved output requires follow-up.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
