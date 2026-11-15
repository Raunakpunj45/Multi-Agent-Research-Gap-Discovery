"""Execute the LangGraph research-gap workflow from the project root.

This runner uses the existing agents and therefore performs live retrieval and
model-backed work when invoked. Tests should use mocked WorkflowAgents instead.
"""
import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.workflow.langgraph_app import (
    ResearchWorkflowState,
    build_research_graph,
    default_workflow_agents,
)
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


def _write_json_no_overwrite(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link creates the final path atomically and fails if it exists.
        os.link(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", help="Research topic to investigate.")
    parser.add_argument("--evaluation-prompt-id", choices=("PROMPT-001", "PROMPT-002", "PROMPT-003"),
                        help="Run with this frozen evaluation prompt and record its identity.")
    parser.add_argument("--manifest", type=Path, default=ROOT_DIR / "data" / "evaluation" / "dataset_manifest.jsonl")
    parser.add_argument("--analysis-query", help="Paper analysis query (defaults to --topic).")
    parser.add_argument("--max-results-per-source", type=int, default=5)
    parser.add_argument("--minimum-papers", type=int, default=2)
    parser.add_argument("--max-papers", type=int)
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument(
        "--paper-pdf", action="append", default=[], metavar="PAPER_ID=PATH",
        help="Use a local PDF for an exact retrieved paper ID; may be repeated.",
    )
    parser.add_argument("--output", type=Path)
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
    if args.minimum_papers < 2:
        parser.error("--minimum-papers must be at least 2.")
    if args.max_revisions < 0:
        parser.error("--max-revisions cannot be negative.")
    if args.max_papers is not None and args.max_papers < args.minimum_papers:
        parser.error("--max-papers cannot be less than --minimum-papers.")

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
        topic = args.topic
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

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(_jsonable(result), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    print(f"Workflow status: {result.get('workflow_status', 'UNKNOWN')}")
    if result.get("stop_reason"):
        print(f"Stopped: {result['stop_reason']}")
    print(f"Saved: {output}")
    return 0 if result.get("workflow_status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
