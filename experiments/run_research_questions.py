"""Generate research questions from saved critic-accepted gaps.

Run from the project root:
    python experiments/run_research_questions.py
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
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
from src.agents.artifact_provenance import (
    require_fingerprint,
    require_recorded_fingerprint,
)

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=PROCESSED / "multi_paper_comparison.json")
    parser.add_argument("--critic", type=Path, default=PROCESSED / "critic_review.json")
    parser.add_argument("--gaps", type=Path, default=PROCESSED / "research_gap_detection.json")
    parser.add_argument("--validation", type=Path, default=PROCESSED / "evidence_validation.json")
    parser.add_argument("--output", type=Path, default=PROCESSED / "research_questions.json")
    args = parser.parse_args()

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent,
            prefix=f".{args.output.name}.", suffix=".tmp", delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            temp_file.write(result.model_dump_json(indent=2))
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, args.output)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    print(f"Saved: {args.output}")
    if result.generation_status.value == "INCOMPLETE":
        print("Research question generation incomplete; saved output requires follow-up.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
