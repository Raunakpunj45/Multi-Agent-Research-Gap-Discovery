"""Validate the frozen evaluation corpus or import a completed offline output.

This phase never calls a model. Use --dry-run to validate the manifest and record
a separate dry-run directory; use --input-output or --multi-agent-artifacts to
score outputs generated elsewhere.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.runner import create_dry_run, create_imported_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT_DIR / "data" / "evaluation" / "dataset_manifest.jsonl",
    )
    parser.add_argument(
        "--runs-dir", type=Path,
        default=ROOT_DIR / "data" / "evaluation" / "runs",
    )
    parser.add_argument("--system-name", default="multi_agent")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate PDF hashes and save a DRY_RUN_VALIDATED record; no model calls.")
    parser.add_argument("--input-output", type=Path,
                        help="Import a normalized EvaluationSystemOutput JSON file.")
    parser.add_argument("--multi-agent-artifacts", nargs=3, type=Path,
                        metavar=("RESEARCH_QUESTIONS", "GAP_DETECTION", "EVIDENCE_VALIDATION"),
                        help="Import existing artifacts; no pipeline stages are run.")
    parser.add_argument("--multi-agent-run", type=Path,
                        help="Import a prompt-bound run_langgraph.py state artifact.")
    parser.add_argument("--research-prompt-id", choices=("PROMPT-001", "PROMPT-002", "PROMPT-003"))
    parser.add_argument("--citation-index", type=Path,
                        help="JSON map from chunk_id to paper_id for ownership checks.")
    parser.add_argument("--export-citation-index", type=Path,
                        help="Read the existing Chroma collection and export its frozen-corpus ownership map; no model calls.")
    parser.add_argument("--chroma-dir", type=Path, default=ROOT_DIR / "chroma_db",
                        help="Existing read-only Chroma persistence directory for citation-index export.")
    parser.add_argument("--prompt", type=Path, help="File containing the exact canonical manifest prompt text.")
    parser.add_argument("--config", type=Path, help="JSON system configuration metadata.")
    parser.add_argument("--annotations", type=Path,
                        default=ROOT_DIR / "data" / "evaluation" / "reference_annotations.jsonl")
    parser.add_argument("--protocol", type=Path,
                        default=ROOT_DIR / "data" / "evaluation" / "annotation_protocol.md")
    args = parser.parse_args()

    try:
        if args.export_citation_index:
            if any((args.dry_run, args.input_output, args.multi_agent_artifacts, args.multi_agent_run)):
                parser.error("--export-citation-index cannot be combined with run/import options.")
            from src.evaluation.citations import build_chunk_ownership_index, write_chunk_ownership_index
            index = build_chunk_ownership_index(args.manifest, args.chroma_dir)
            write_chunk_ownership_index(index, args.export_citation_index)
            print(f"Citation ownership index saved: {args.export_citation_index}")
            return
        if args.dry_run:
            if args.input_output or args.multi_agent_artifacts or args.multi_agent_run:
                parser.error("--dry-run cannot be combined with imported output options.")
            run_dir = create_dry_run(args.manifest, args.runs_dir, args.system_name)
        else:
            selected = sum(bool(value) for value in (
                args.input_output, args.multi_agent_artifacts, args.multi_agent_run,
            ))
            if selected != 1:
                parser.error("Provide exactly one imported output/artifact option, or use --dry-run.")
            if not args.research_prompt_id or not args.prompt:
                parser.error("Imported runs require --research-prompt-id and --prompt with canonical prompt text.")
            config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            if not isinstance(config, dict):
                parser.error("--config must contain a JSON object.")
            run_dir = create_imported_run(
                args.input_output, args.manifest, args.runs_dir,
                system_name=args.system_name,
                system_configuration=config,
                prompt_path=args.prompt,
                protocol_path=args.protocol,
                annotations_path=args.annotations,
                citation_index_path=args.citation_index,
                multi_agent_paths=tuple(args.multi_agent_artifacts) if args.multi_agent_artifacts else None,
                multi_agent_bundle_path=args.multi_agent_run,
                research_prompt_id=args.research_prompt_id,
                canonical_prompt_text=args.prompt.read_text(encoding="utf-8"),
            )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"Evaluation record saved: {run_dir}")


if __name__ == "__main__":
    main()
