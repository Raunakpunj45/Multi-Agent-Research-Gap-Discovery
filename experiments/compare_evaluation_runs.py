"""Compare stored evaluation records without filling in missing judgments."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.models import EvaluationRunManifest


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            temp_file.write(payload)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def compare_runs(runs_dir: Path) -> dict:
    runs = []
    errors = []
    for path in sorted(Path(runs_dir).glob("*/run_manifest.json")):
        try:
            runs.append(EvaluationRunManifest.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception as error:
            errors.append({"path": str(path), "error": str(error)})

    buckets = defaultdict(list)
    for run in runs:
        key = (run.dataset_version, run.dataset_hash,
               tuple(run.research_prompt_ids), run.canonical_prompt_hash,
               run.evaluation_split, run.protocol_hash)
        buckets[key].append(run)

    groups = []
    required_systems = {"single_llm", "conventional_rag", "multi_agent"}
    for key, compatible in buckets.items():
        complete_runs = [run for run in compatible if run.status == "COMPLETE"]
        metrics_by_system = defaultdict(lambda: defaultdict(list))
        for run in complete_runs:
            for metric in run.metrics:
                if metric.value is not None:
                    metrics_by_system[run.system_name][metric.name].append(metric.value)
        systems = []
        for system_name in sorted({run.system_name for run in complete_runs}):
            score_summary = {}
            for metric_name in sorted({m.name for r in complete_runs for m in r.metrics}):
                values = metrics_by_system[system_name].get(metric_name, [])
                score_summary[metric_name] = {
                    "mean": sum(values) / len(values) if values else None,
                    "n": len(values),
                    "missing": not bool(values),
                }
            systems.append({"system_name": system_name, "scores": score_summary})
        groups.append({
            "dataset_version": key[0],
            "dataset_hash": key[1],
            "research_prompt_ids": list(key[2]),
            "canonical_prompt_hash": key[3],
            "evaluation_split": key[4],
            "protocol_hash": key[5],
            "run_ids": [run.run_id for run in compatible],
            "excluded_noncomplete_runs": [
                run.run_id for run in compatible if run.status != "COMPLETE"
            ],
            "systems": systems,
            "missing_systems": sorted(required_systems - {run.system_name for run in complete_runs}),
            "missing_human_annotations": [
                run.run_id for run in compatible if run.annotation_status != "AVAILABLE"
            ],
            "missing_canonical_prompt_hash": key[3] is None,
            "missing_research_prompt_assignment": not bool(key[2]),
            "missing_split_assignment": key[3] is None,
            "comparison_ready": (
                required_systems.issubset({run.system_name for run in complete_runs})
                and bool(key[2]) and bool(key[3])
            ),
        })
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "compatible_groups": groups,
        "incompatible_runs_are_separated_by_dataset_prompt_and_protocol": True,
        "load_errors": errors,
        "limitations": [
            "Only runs with identical dataset hash, dataset version, prompt IDs, and annotation protocol hash are grouped.",
            "Missing human scores remain null and are counted; no labels are imputed.",
            "Novelty results are corpus-bounded indications, never proof of global novelty.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path,
                        default=ROOT_DIR / "data" / "evaluation" / "runs")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = compare_runs(args.runs_dir)
    output = args.output or (
        ROOT_DIR / "data" / "evaluation" / "comparisons"
        / f"comparison-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}.json"
    )
    _atomic_write(output, json.dumps(result, indent=2))
    print(f"Comparison report saved: {output}")


if __name__ == "__main__":
    main()
