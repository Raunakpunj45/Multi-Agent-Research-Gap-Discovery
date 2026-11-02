"""Live research gap detection (uses Gemini; not part of pytest).

Reads the saved multi-paper comparison, runs ResearchGapDetectionAgent on it,
prints the result and saves it as JSON. Makes ONE Gemini call.

Run from the project root:
    python experiments/run_research_gap_detection.py
    python experiments/run_research_gap_detection.py --topic "large language models scientific discovery"

The comparison JSON does not store the research topic, so --topic is optional
and is never guessed; without it the topic is reported as None.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from pydantic import ValidationError

from src.agents import (
    MultiPaperComparison,
    ResearchGapDetectionAgent,
    ResearchGapDetectionResult,
)

INPUT_PATH = ROOT_DIR / "data" / "processed" / "multi_paper_comparison.json"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "research_gap_detection.json"


def fail(message: str) -> None:
    print(f"\nERROR: {message}")
    sys.exit(1)


def load_comparison(path: Path) -> MultiPaperComparison:
    if not path.exists():
        fail(f"Comparison file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail(f"{path.name} is not valid JSON: {error}")

    if not isinstance(raw, dict):
        fail(f"{path.name} must contain a JSON object.")

    # Pydantic ignores unknown keys by default; report them instead of
    # letting them disappear silently.
    unknown = sorted(set(raw) - set(MultiPaperComparison.model_fields))
    if unknown:
        print(f"WARNING: fields in the JSON not in MultiPaperComparison "
              f"(ignored): {unknown}")

    try:
        comparison = MultiPaperComparison.model_validate(raw)
    except ValidationError as error:
        print("\nSCHEMA INCOMPATIBILITY: the JSON does not match "
              "MultiPaperComparison. Nothing was changed or filled in.")
        print(error)
        sys.exit(1)

    if comparison.paper_count != len(comparison.paper_ids):
        print(f"WARNING: paper_count={comparison.paper_count} but "
              f"{len(comparison.paper_ids)} paper_ids listed (not corrected).")

    return comparison


def print_gaps(result: ResearchGapDetectionResult) -> None:
    print(f"\n=== IDENTIFIED GAPS: {len(result.gaps)} ===")

    for gap in result.gaps:
        print(f"\n[{gap.gap_id}] {gap.title}")
        print(f"  Category:   {gap.gap_category.value}")
        print(f"  Origin:     {gap.gap_origin.value}")
        print(f"  Confidence: {gap.confidence.value} - {gap.confidence_reason}")
        print(f"  Description: {gap.description}")
        print(f"  Reasoning:   {gap.reasoning}")
        print(f"  Evidence summary: {gap.evidence_summary}")
        print(f"  Opportunity: {gap.research_opportunity}")
        print(f"  Supporting paper IDs: {gap.supporting_paper_ids}")
        print("  Supporting evidence (paper -> chunks):")
        for ref in gap.supporting_evidence:
            print(f"    {ref.paper_id} -> {ref.chunk_ids}")

    print("\n=== CATEGORY COUNTS ===")
    if result.gap_category_counts:
        for category, count in result.gap_category_counts.items():
            print(f"  {category}: {count}")
    else:
        print("  (none)")

    print(f"\n=== EXCLUDED CANDIDATES: {len(result.excluded_candidates)} ===")
    for item in result.excluded_candidates:
        print(f" - {item.title}\n   reason: {item.reason}")

    print("\n=== CAVEATS ===")
    for caveat in result.caveats:
        print(f" - {caveat}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default=None,
                        help="Research topic (not stored in the comparison JSON).")
    args = parser.parse_args()

    comparison = load_comparison(INPUT_PATH)
    print(f"Loaded comparison: {comparison.paper_count} papers, "
          f"{len(comparison.recurring_patterns)} recurring patterns, "
          f"{len(comparison.common_limitations)} common limitations, "
          f"{len(comparison.methodological_differences)} methodological differences, "
          f"{len(comparison.conflicting_findings)} conflicts, "
          f"{len(comparison.underexplored_directions)} underexplored directions.")
    print(f"Research topic: {args.topic!r}")

    try:
        agent = ResearchGapDetectionAgent()
        result = agent.detect_gaps(comparison, research_topic=args.topic)
    except ValueError as error:
        fail(f"Gap detection could not run: {error}")

    print_gaps(result)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
