"""Run one independent baseline for one finalized evaluation prompt.

Invoking this script makes one Gemini generation call. Tests inject mocked
generators/retrievers and never invoke this CLI.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.baselines import (
    ConventionalRAGBaseline,
    GeminiStructuredGenerator,
    SingleLLMBaseline,
    build_read_only_rag_retriever,
    load_frozen_evaluation_corpus,
)
from src.evaluation.runner import create_imported_run
from src.agents.llm_provider import get_provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=("single_llm", "conventional_rag"))
    parser.add_argument("--prompt-id", required=True, choices=("PROMPT-001", "PROMPT-002", "PROMPT-003"))
    parser.add_argument("--manifest", type=Path, default=ROOT_DIR / "data" / "evaluation" / "dataset_manifest.jsonl")
    parser.add_argument("--runs-dir", type=Path, default=ROOT_DIR / "data" / "evaluation" / "runs")
    parser.add_argument("--normalized-output", type=Path,
                        help="Optional copy of the normalized output JSON before runner import.")
    parser.add_argument("--model", default=None,
                        help="Model ID; default comes from LLM_PROVIDER and <PROVIDER>_MODEL_BASELINE (see README).")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Per-paper retrieval count for conventional RAG.")
    parser.add_argument("--chroma-dir", type=Path, default=ROOT_DIR / "chroma_db")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    corpus = load_frozen_evaluation_corpus(args.manifest)
    prompt = corpus.prompts[args.prompt_id]
    generator = GeminiStructuredGenerator(model=args.model)
    provider = get_provider()
    if args.system == "single_llm":
        baseline = SingleLLMBaseline(generator)
        output = baseline.generate(corpus, args.prompt_id, ROOT_DIR)
        system_configuration = {
            "baseline": "single_llm",
            "provider": provider,
            "model": generator.model,
            "temperature": 0,
            "context_strategy": "all locally extracted chunks from all six evaluation papers in one generation call",
        }
    else:
        retriever = build_read_only_rag_retriever(args.chroma_dir)
        baseline = ConventionalRAGBaseline(generator, retriever, top_k=args.top_k)
        output = baseline.generate(corpus, args.prompt_id)
        system_configuration = {
            "baseline": "conventional_rag",
            "provider": provider,
            "model": generator.model,
            "temperature": 0,
            "retrieval": "existing_chromadb_collection",
            "top_k_per_paper": args.top_k,
            "collection_directory": str(args.chroma_dir.resolve()),
        }

    if args.normalized_output:
        args.normalized_output.parent.mkdir(parents=True, exist_ok=True)
        args.normalized_output.write_text(output.model_dump_json(indent=2) + "\n", encoding="utf-8")

    owners = {
        citation.chunk_id: citation.paper_id
        for candidate in output.candidates
        for citation in candidate.citations
    }
    with tempfile.TemporaryDirectory(prefix="evaluation-baseline-") as temporary:
        temp = Path(temporary)
        output_path = temp / "normalized-output.json"
        owners_path = temp / "chunk-owners.json"
        output_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        owners_path.write_text(json.dumps(owners, ensure_ascii=False), encoding="utf-8")
        run_dir = create_imported_run(
            output_path,
            args.manifest,
            args.runs_dir,
            system_name=args.system,
            system_configuration=system_configuration,
            research_prompt_id=args.prompt_id,
            canonical_prompt_text=prompt["text"],
            citation_index_path=owners_path,
        )
    print(f"Baseline status: {output.completion_status.value}")
    print(f"Prompt: {args.prompt_id}")
    print(f"Candidates: {len(output.candidates)}")
    print(f"Evaluation run saved: {run_dir}")
    return 0 if output.completion_status.value == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
