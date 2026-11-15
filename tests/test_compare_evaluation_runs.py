import json

from src.evaluation.models import EvaluationRunManifest, MetricScore
from experiments.compare_evaluation_runs import compare_runs


def write_run(root, name, dataset_hash, system, annotations="MISSING", value=None):
    run_id = f"run-{name}"
    run = EvaluationRunManifest(
        run_id=run_id, status="COMPLETE", system_name=system,
        dataset_version="v1", dataset_hash=dataset_hash,
        prompt_hash="prompt-hash", prompt_hash_status="RECORDED",
        canonical_prompt_hash="prompt-hash",
        protocol_hash="protocol-hash", research_prompt_ids=["Q1"],
        evaluation_split=None,
        annotation_status=annotations,
        metrics=[MetricScore(
            name="gap_relevance_human", value=value, denominator=1 if value is not None else 0,
            method="HUMAN", missing_reason=None if value is not None else "pending",
        )],
    )
    path = root / name / "run_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return run_id


def test_comparison_aggregates_only_compatible_runs_and_flags_missing_humans(tmp_path):
    first = write_run(tmp_path, "a", "same-dataset", "single_llm", "AVAILABLE", 0.8)
    second = write_run(tmp_path, "b", "same-dataset", "conventional_rag", "MISSING", None)
    third_same = write_run(tmp_path, "d", "same-dataset", "multi_agent", "MISSING", None)
    third = write_run(tmp_path, "c", "other-dataset", "multi_agent", "MISSING", 0.4)

    result = compare_runs(tmp_path)

    compatible = next(group for group in result["compatible_groups"]
                      if group["dataset_hash"] == "same-dataset")
    assert set(compatible["run_ids"]) == {first, second, third_same}
    assert compatible["comparison_ready"] is True
    assert compatible["missing_systems"] == []
    assert compatible["evaluation_split"] is None
    assert set(compatible["missing_human_annotations"]) == {second, third_same}
    rag = next(system for system in compatible["systems"] if system["system_name"] == "conventional_rag")
    assert rag["scores"]["gap_relevance_human"] == {"mean": None, "n": 0, "missing": True}
    other = next(group for group in result["compatible_groups"]
                 if third in group["run_ids"])
    assert other["comparison_ready"] is False
    assert set(other["missing_systems"]) == {"single_llm", "conventional_rag"}
    assert "global novelty" in " ".join(result["limitations"])


def test_prompt_hash_mismatch_separates_runs_and_prevents_cross_prompt_comparison(tmp_path):
    first = write_run(tmp_path, "a", "same-dataset", "single_llm", "MISSING", 0.8)
    second = write_run(tmp_path, "b", "same-dataset", "conventional_rag", "MISSING", 0.9)
    manifest = tmp_path / "b" / "run_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["canonical_prompt_hash"] = "different-prompt-hash"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = compare_runs(tmp_path)

    assert len(result["compatible_groups"]) == 2
    assert all(group["comparison_ready"] is False for group in result["compatible_groups"])
    assert {run for group in result["compatible_groups"] for run in group["run_ids"]} == {first, second}


def test_compare_ignores_invalid_run_manifest_but_reports_error(tmp_path):
    bad = tmp_path / "bad" / "run_manifest.json"
    bad.parent.mkdir()
    bad.write_text("{}", encoding="utf-8")
    result = compare_runs(tmp_path)
    assert result["load_errors"]
    assert result["compatible_groups"] == []
