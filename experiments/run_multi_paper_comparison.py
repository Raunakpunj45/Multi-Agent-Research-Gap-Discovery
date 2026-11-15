"""Live multi-paper comparison (uses network + Gemini; not part of pytest).

Run from the project root:
    python experiments/run_multi_paper_comparison.py "large language models scientific discovery" --max-papers 10
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.agents import PaperAnalysisPipeline
from src.agents.planner_retrieval_agent import PlannerRetrievalAgent
from src.retrieval import MultiSourceRetriever

ANALYSIS_QUERY = (
    "What are the problem, methodology, dataset, results, "
    "limitations, and future work discussed in this paper?"
)
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "multi_paper_comparison.json"


def show_evidence(refs):
    return "; ".join(f"{r.paper_id} [{', '.join(r.chunk_ids)}]" for r in refs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topic")
    parser.add_argument("--max-papers", type=int, default=10)
    parser.add_argument("--per-source", type=int, default=10)
    parser.add_argument(
        "--use-planner", action="store_true",
        help="Opt in to Planner Agent query expansion before retrieval.",
    )
    args = parser.parse_args()

    if args.use_planner:
        _, papers = PlannerRetrievalAgent().search(
            topic=args.topic, max_results_per_source=args.per_source
        )
    else:
        papers = MultiSourceRetriever(peer_reviewed_only=True).search(
            query=args.topic, max_results_per_source=args.per_source
        )
    candidates = [p for p in papers if p.pdf_url]
    print(f"\n{len(papers)} papers retrieved, {len(candidates)} with a PDF URL.")

    comparison = PaperAnalysisPipeline().analyze_and_compare_papers(
        papers=candidates,
        analysis_query=ANALYSIS_QUERY,
        research_topic=args.topic,
        max_successful=args.max_papers,
    )

    print(f"\n=== COMPARISON OF {comparison.paper_count} PAPERS ===")
    for title, items in (
        ("Recurring patterns", comparison.recurring_patterns),
        ("Common limitations", comparison.common_limitations),
        ("Methodological differences", comparison.methodological_differences),
    ):
        print(f"\n{title}: {len(items)}")
        for f in items:
            print(f" - {f.title}: {f.description}\n   evidence: {show_evidence(f.evidence)}")

    print(f"\nConflicting findings: {len(comparison.conflicting_findings)}")
    for c in comparison.conflicting_findings:
        print(f" - {c.topic}: {c.description}")

    print(f"\nUnderexplored directions (candidates only): {len(comparison.underexplored_directions)}")
    for d in comparison.underexplored_directions:
        print(f" - {d.direction}\n   evidence: {show_evidence(d.evidence)}")

    print("\nExcluded papers:", [(e.paper_id, e.reason) for e in comparison.excluded_papers])
    print("\nCaveats:")
    for c in comparison.caveats:
        print(f" - {c}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=OUTPUT_PATH.parent,
            prefix=f".{OUTPUT_PATH.name}.", suffix=".tmp", delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            temp_file.write(comparison.model_dump_json(indent=2))
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, OUTPUT_PATH)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
