"""Critique saved comparison, gap-detection, and evidence-validation outputs.

Run from the project root:
    python experiments/run_critic.py
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.agents.artifact_io import write_text_preserving_previous

from pydantic import ValidationError

from src.agents import (
    CriticAgent,
    CriticResult,
    CriticReviewStatus,
    EvidenceValidationResult,
    MultiPaperComparison,
    ResearchGapDetectionResult,
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
        fingerprint = hashlib.sha256(payload).hexdigest()
        return model.model_validate(raw), None, fingerprint
    except (json.JSONDecodeError, ValidationError, OSError) as error:
        fingerprint = hashlib.sha256(payload).hexdigest() if payload is not None else None
        return None, f"Could not load {path}: {error}", fingerprint


def load_previous(path: Path):
    if not path.exists():
        return None
    try:
        return CriticResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValidationError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=PROCESSED / "multi_paper_comparison.json")
    parser.add_argument("--gaps", type=Path, default=PROCESSED / "research_gap_detection.json")
    parser.add_argument("--validation", type=Path, default=PROCESSED / "evidence_validation.json")
    parser.add_argument("--output", type=Path, default=PROCESSED / "critic_review.json")
    args = parser.parse_args()

    comparison, comparison_error, comparison_fingerprint = load_optional(
        args.comparison, MultiPaperComparison
    )
    gaps, gaps_error, gaps_fingerprint = load_optional(
        args.gaps, ResearchGapDetectionResult
    )
    validation, validation_error, validation_fingerprint = load_optional(
        args.validation, EvidenceValidationResult
    )
    lineage_errors = []
    if comparison is not None:
        try:
            require_recorded_fingerprint(
                getattr(comparison, "provenance", None),
                "paper_analyses", "Comparison artifact",
            )
        except ValueError as error:
            lineage_errors.append(str(error))
            comparison = None
    if gaps is not None and comparison_fingerprint is not None:
        try:
            require_fingerprint(
                getattr(gaps, "provenance", None),
                "comparison", comparison_fingerprint,
                "Gap detection artifact",
            )
        except ValueError as error:
            lineage_errors.append(str(error))
            gaps = None
    if validation is not None and gaps_fingerprint is not None:
        try:
            require_fingerprint(
                getattr(validation, "provenance", None),
                "gap_detection", gaps_fingerprint,
                "Evidence validation artifact",
            )
        except ValueError as error:
            lineage_errors.append(str(error))
            validation = None
    fingerprints = {
        key: value for key, value in (
            ("comparison", comparison_fingerprint),
            ("gap_detection", gaps_fingerprint),
            ("evidence_validation", validation_fingerprint),
        ) if value is not None
    }
    previous = load_previous(args.output)
    agent = CriticAgent()
    result = agent.critique(
        comparison, gaps, validation, input_fingerprints=fingerprints,
        previous_result=previous,
    )
    load_errors = [error for error in (
        comparison_error, gaps_error, validation_error, *lineage_errors
    ) if error]
    if load_errors:
        result.errors.extend(load_errors)
        # Input loading or lineage failures can never yield a complete review.
        result.review_status = CriticReviewStatus.INCOMPLETE

    for review in result.gap_reviews:
        print(f"{review.gap_id}: {review.decision.value} ({review.review_status.value})")
        for criticism in review.criticisms:
            refs = "; ".join(
                f"{ref.paper_id} -> {', '.join(ref.chunk_ids)}"
                for ref in criticism.source_evidence
            ) or "no direct source citation"
            print(f"  {criticism.category.value}/{criticism.severity.value}: {criticism.issue}")
            print(f"    evidence: {refs}")
        for suggestion in review.revision_suggestions:
            print(f"  revise {suggestion.field}: {suggestion.proposed_text}")
    print(f"Critic result: {result.review_status.value}")
    for error in result.errors:
        print(f"  error: {error}")

    archived = write_text_preserving_previous(
        args.output, result.model_dump_json(indent=2) + "\n",
    )
    if archived is not None:
        print(f"Previous output preserved: {archived}")
    print(f"Saved: {args.output}")
    if (result.review_status.value == "INCOMPLETE"
            or any(review.review_status.value == "INCOMPLETE"
                   for review in result.gap_reviews)):
        print("Critic run incomplete; returning exit code 1.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
