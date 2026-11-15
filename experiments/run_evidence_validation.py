"""Validate saved candidate gaps against exact stored chunks using Gemini.

Run from the project root with GEMINI_API_KEY configured:
    python experiments/run_evidence_validation.py
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

from src.agents import EvidenceValidationAgent, MultiPaperComparison, ResearchGapDetectionResult
from src.agents.artifact_provenance import (
    ArtifactProvenance,
    require_fingerprint,
    require_recorded_fingerprint,
)
from src.agents.evidence_validation_agent import (
    EvidenceValidationResult,
    EvidenceValidationStatus,
    GapValidationStatus,
)

INPUT_PATH = ROOT_DIR / "data" / "processed" / "research_gap_detection.json"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "evidence_validation.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument(
        "--comparison", type=Path,
        default=ROOT_DIR / "data" / "processed" / "multi_paper_comparison.json",
        help="Comparison artifact used to verify gap-detection lineage.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--retry-joint", metavar="GAP_ID",
        help="Retry only the missing joint assessment for GAP_ID in the saved output JSON.",
    )
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"Gap detection file not found: {args.input}")
    gap_bytes = args.input.read_bytes()
    try:
        result = ResearchGapDetectionResult.model_validate_json(
            gap_bytes
        )
    except (ValidationError, json.JSONDecodeError) as error:
        parser.error(f"Invalid gap detection JSON: {error}")
    if not args.comparison.exists():
        parser.error(f"Comparison file not found for lineage verification: {args.comparison}")
    comparison_bytes = args.comparison.read_bytes()
    try:
        comparison = MultiPaperComparison.model_validate_json(comparison_bytes)
        require_recorded_fingerprint(
            comparison.provenance, "paper_analyses", "Comparison artifact",
        )
    except (ValidationError, json.JSONDecodeError, ValueError) as error:
        parser.error(f"Invalid or unproven comparison artifact: {error}")
    comparison_fingerprint = hashlib.sha256(comparison_bytes).hexdigest()
    try:
        require_fingerprint(
            result.provenance, "comparison", comparison_fingerprint,
            "Gap detection artifact",
        )
    except ValueError as error:
        parser.error(str(error))
    gap_fingerprint = hashlib.sha256(gap_bytes).hexdigest()

    agent = EvidenceValidationAgent()
    if args.retry_joint:
        if not args.output.exists():
            parser.error(f"Saved evidence validation file not found: {args.output}")
        try:
            saved = EvidenceValidationResult.model_validate_json(
                args.output.read_text(encoding="utf-8")
            )
        except (ValidationError, json.JSONDecodeError) as error:
            parser.error(f"Invalid saved evidence validation JSON: {error}")
        try:
            require_fingerprint(
                saved.provenance, "gap_detection", gap_fingerprint,
                "Saved evidence validation artifact",
            )
        except ValueError as error:
            parser.error(str(error))
        gap = next((item for item in result.gaps if item.gap_id == args.retry_joint), None)
        old = next((item for item in saved.gap_validations
                    if item.gap_id == args.retry_joint), None)
        if gap is None or old is None:
            parser.error(f"Gap {args.retry_joint} must exist in both input and saved validation files")
        if old.status is not GapValidationStatus.VALIDATION_INCOMPLETE or old.joint_assessment is not None:
            parser.error(f"{args.retry_joint} is not an incomplete validation with a missing joint assessment")
        # Any exception here exits before touching the saved file.
        retried = agent.retry_joint_assessment(gap, old)
        validations = [retried if item.gap_id == args.retry_joint else item
                       for item in saved.gap_validations]
        statuses = [item.status.value for item in validations]
        evidence_statuses = [
            item.status.value for validation in validations
            for item in validation.evidence_validations if item.status is not None
        ]
        updated = saved.model_copy(update={
            "gap_validations": validations,
            "validation_counts": {status.value: statuses.count(status.value)
                                  for status in GapValidationStatus},
            "evidence_status_counts": {
                status.value: evidence_statuses.count(status.value)
                for status in EvidenceValidationStatus
            },
            "semantic_failures": [
                failure for failure in saved.semantic_failures
                if not failure.startswith(f"{args.retry_joint}: joint Gemini assessment failed:")
            ],
        })
        validated = updated
    else:
        validated = agent.validate(result)
    validated.provenance = ArtifactProvenance(
        stage="evidence_validation",
        input_fingerprints={"gap_detection": gap_fingerprint},
    )
    print(f"Evidence validation: {len(validated.gap_validations)} gaps")
    for item in validated.gap_validations:
        print(f"  {item.gap_id}: {item.status.value} ({len(item.evidence_validations)} evidence items)")
        for contradiction in item.contradictions:
            print("    CONTRADICTION:")
            print(f"      A: {contradiction.proposition_a}")
            print(f"         source: {contradiction.source_a.paper_id} -> {contradiction.source_a.chunk_ids}")
            print(f"         quote: {contradiction.supporting_quote_a}")
            print(f"      B: {contradiction.proposition_b}")
            print(f"         source: {contradiction.source_b.paper_id} -> {contradiction.source_b.chunk_ids}")
            print(f"         quote: {contradiction.supporting_quote_b}")
            print(f"      Why they conflict: {contradiction.explanation}")
    print("Gap counts:", validated.validation_counts)
    print("Evidence counts:", validated.evidence_status_counts)
    if validated.retrieval_failures:
        print("Retrieval failures:")
        for failure in validated.retrieval_failures:
            print(f"  - {failure}")
    if validated.semantic_failures:
        print("Semantic assessment failures:")
        for failure in validated.semantic_failures:
            print(f"  - {failure}")
    for caveat in validated.caveats:
        print(f"Caveat: {caveat}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically only after validation and serialization succeed.
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent,
            prefix=f".{args.output.name}.", suffix=".tmp", delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            temp_file.write(validated.model_dump_json(indent=2))
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, args.output)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    print(f"Saved: {args.output}")
    if any(item.status is GapValidationStatus.VALIDATION_INCOMPLETE
           for item in validated.gap_validations):
        print("Evidence validation incomplete; saved output requires follow-up.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
