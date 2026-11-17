"""Validate saved candidate gaps against exact stored chunks using Gemini.

Run from the project root with GEMINI_API_KEY_VALIDATION, the earlier
GEMINI_API_KEY_VALIDATION_CRITIC key, or the GEMINI_API_KEY fallback configured:
    python experiments/run_evidence_validation.py
"""
import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.agents.artifact_io import (
    write_text_no_overwrite,
    write_text_preserving_previous,
)

from pydantic import ValidationError

from src.agents import EvidenceValidationAgent, MultiPaperComparison, ResearchGapDetectionResult
from src.agents.artifact_provenance import (
    ArtifactProvenance,
    CANONICAL_JSON_SHA256,
    RAW_BYTES_SHA256,
    fingerprint_json,
    require_fingerprint,
    require_fingerprint_scheme,
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
    parser.add_argument("--retry-evidence", metavar="GAP_ID", help="Retry failed individual assessments for GAP_ID.")
    parser.add_argument("--saved-validation", type=Path, help="Prior validation artifact for --retry-evidence.")
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"Gap detection file not found: {args.input}")
    gap_bytes = args.input.read_bytes()
    if args.retry_evidence:
        try:
            raw_container = json.loads(gap_bytes)
        except json.JSONDecodeError:
            raw_container = None
        if isinstance(raw_container, dict) and all(key in raw_container for key in ("comparison", "gap_detection", "evidence_validation")):
            source = args.input.resolve()
            run_dir = ROOT_DIR / "data" / "processed" / "targeted_retries" / f"{source.stem}-{uuid.uuid4().hex[:8]}"
            run_dir.mkdir(parents=True, exist_ok=False)
            paths = {key: run_dir / f"{key}.json" for key in ("comparison", "gap_detection", "evidence_validation")}
            for key, path in paths.items():
                value = raw_container[key]
                if isinstance(value, dict):
                    value = dict(value)
                    prov = dict(value.get("provenance") or {})
                    expected_scheme_key = {"comparison": "paper_analyses", "gap_detection": "comparison",
                                           "evidence_validation": "gap_detection"}[key]
                    declared = (prov.get("fingerprint_schemes") or {}).get(expected_scheme_key)
                    if declared is not None and declared != CANONICAL_JSON_SHA256:
                        parser.error(f"LangGraph {key} artifact declares an incompatible {expected_scheme_key!r} fingerprint scheme: {declared!r}.")
                    prov["fingerprint_schemes"] = {**prov.get("fingerprint_schemes", {}),
                        expected_scheme_key: CANONICAL_JSON_SHA256}
                    value["provenance"] = prov
                path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            args.input, args.comparison = paths["gap_detection"], paths["comparison"]
            args.saved_validation = paths["evidence_validation"]
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
    comparison_scheme = comparison.provenance.fingerprint_schemes.get("paper_analyses", RAW_BYTES_SHA256)
    if comparison_scheme not in (RAW_BYTES_SHA256, CANONICAL_JSON_SHA256):
        parser.error(f"Unsupported comparison fingerprint scheme: {comparison_scheme}")
    comparison_fingerprint = (fingerprint_json(comparison.model_dump(mode="json", exclude={"provenance"}))
                              if comparison_scheme == CANONICAL_JSON_SHA256
                              else hashlib.sha256(comparison_bytes).hexdigest())
    try:
        require_fingerprint_scheme(result.provenance, "comparison", "Gap detection artifact", comparison_scheme)
        require_fingerprint(
            result.provenance, "comparison", comparison_fingerprint,
            "Gap detection artifact",
        )
    except ValueError as error:
        parser.error(str(error))
    gap_scheme = result.provenance.fingerprint_schemes.get("comparison", RAW_BYTES_SHA256)
    gap_fingerprint = (fingerprint_json(result.model_dump(mode="json", exclude={"provenance"}))
                       if gap_scheme == CANONICAL_JSON_SHA256 else hashlib.sha256(gap_bytes).hexdigest())

    agent = EvidenceValidationAgent()
    saved_path = args.saved_validation if args.retry_evidence else args.output
    if args.retry_evidence:
        if args.saved_validation is None or not args.saved_validation.exists():
            parser.error("--retry-evidence requires an existing --saved-validation file.")
        if args.output.resolve() in {args.input.resolve(), args.comparison.resolve(), args.saved_validation.resolve()} or args.output.exists():
            parser.error("--retry-evidence output must be a new path distinct from every input file.")
    if args.retry_joint or args.retry_evidence:
        if not saved_path.exists():
            parser.error(f"Saved evidence validation file not found: {saved_path}")
        try:
            saved = EvidenceValidationResult.model_validate_json(
                saved_path.read_text(encoding="utf-8")
            )
        except (ValidationError, json.JSONDecodeError) as error:
            parser.error(f"Invalid saved evidence validation JSON: {error}")
        try:
            require_fingerprint_scheme(saved.provenance, "gap_detection", "Saved evidence validation artifact", gap_scheme)
            require_fingerprint(
                saved.provenance, "gap_detection", gap_fingerprint,
                "Saved evidence validation artifact",
            )
        except ValueError as error:
            parser.error(str(error))
        retry_id = args.retry_joint or args.retry_evidence
        gap = next((item for item in result.gaps if item.gap_id == retry_id), None)
        old = next((item for item in saved.gap_validations if item.gap_id == retry_id), None)
        if gap is None or old is None:
            parser.error(f"Gap {args.retry_joint} must exist in both input and saved validation files")
        if args.retry_joint and (old.status is not GapValidationStatus.VALIDATION_INCOMPLETE or old.joint_assessment is not None):
            parser.error(f"{args.retry_joint} is not an incomplete validation with a missing joint assessment")
        # Any exception here exits before touching the saved file.
        retried = (agent.retry_failed_evidence(gap, old) if args.retry_evidence
                   else agent.retry_joint_assessment(gap, old))
        validations = [retried if item.gap_id == retry_id else item
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
                if not failure.startswith(f"{retry_id}:")
            ],
        })
        validated = updated
    else:
        validated = agent.validate(result)
    validated.provenance = ArtifactProvenance(
        stage="evidence_validation",
        input_fingerprints={"gap_detection": gap_fingerprint},
        fingerprint_schemes={"gap_detection": gap_scheme},
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
    # Write atomically only after validation and serialization succeed.
    payload = validated.model_dump_json(indent=2) + "\n"
    if args.retry_evidence:
        write_text_no_overwrite(args.output, payload)
    else:
        archived = write_text_preserving_previous(args.output, payload)
        if archived is not None:
            print(f"Previous output preserved: {archived}")
    print(f"Saved: {args.output}")
    if any(item.status is GapValidationStatus.VALIDATION_INCOMPLETE
           for item in validated.gap_validations):
        print("Evidence validation incomplete; saved output requires follow-up.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
