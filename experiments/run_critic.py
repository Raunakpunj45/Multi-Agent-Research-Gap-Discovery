"""Critique saved comparison, gap-detection, and evidence-validation outputs.

Run from the project root:
    python experiments/run_critic.py
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from pydantic import ValidationError

from src.agents import (
    CriticAgent,
    EvidenceValidationResult,
    MultiPaperComparison,
    ResearchGapDetectionResult,
)

PROCESSED = ROOT_DIR / "data" / "processed"


def load_optional(path: Path, model):
    if not path.exists():
        return None, f"Missing input file: {path}"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return model.model_validate(raw), None
    except (json.JSONDecodeError, ValidationError, OSError) as error:
        return None, f"Could not load {path}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=PROCESSED / "multi_paper_comparison.json")
    parser.add_argument("--gaps", type=Path, default=PROCESSED / "research_gap_detection.json")
    parser.add_argument("--validation", type=Path, default=PROCESSED / "evidence_validation.json")
    parser.add_argument("--output", type=Path, default=PROCESSED / "critic_review.json")
    args = parser.parse_args()

    comparison, comparison_error = load_optional(args.comparison, MultiPaperComparison)
    gaps, gaps_error = load_optional(args.gaps, ResearchGapDetectionResult)
    validation, validation_error = load_optional(args.validation, EvidenceValidationResult)
    agent = CriticAgent()
    result = agent.critique(comparison, gaps, validation)
    load_errors = [error for error in (comparison_error, gaps_error, validation_error) if error]
    if load_errors:
        result.errors.extend(load_errors)

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


if __name__ == "__main__":
    main()
