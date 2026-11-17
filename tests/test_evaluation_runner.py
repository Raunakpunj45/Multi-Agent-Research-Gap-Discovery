import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.runner import (
    ManifestValidationError,
    create_dry_run,
    create_imported_run,
    load_dataset_manifest,
    canonical_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "evaluation" / "dataset_manifest.jsonl"


def test_manifest_has_frozen_evaluation_corpus_and_is_reproducibly_hashed(evaluation_corpus):
    first = load_dataset_manifest(evaluation_corpus.manifest, evaluation_corpus.root)
    second = load_dataset_manifest(evaluation_corpus.manifest, evaluation_corpus.root)
    assert len(first.papers) == 8
    assert first.manifest_hash == second.manifest_hash
    assert all(paper.split is None for paper in first.papers)
    included = [p for p in first.papers if p.inclusion_decision["decision"] == "INCLUDE_IN_PRIMARY_EVALUATION"]
    excluded = [p for p in first.papers if p.inclusion_decision["decision"] == "EXCLUDE_FROM_PRIMARY_EVALUATION"]
    assert len(included) == 6
    assert len(excluded) == 2
    assert len(first.header.research_prompts) == 3
    assert all(p["status"] == "FINALIZED_SHARED_PROMPT" for p in first.header.research_prompts)
    prompt_ids = [p["prompt_id"] for p in first.header.research_prompts]
    assert all(p.research_prompt_ids == prompt_ids for p in included)
    assert first.header.evaluation_configuration["frozen"] is True
    assert first.header.evaluation_configuration["systems"] == ["Single LLM", "Conventional RAG", "Multi-Agent"]


def test_included_openalex_identities_match_independent_title_search_audit():
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    included = [
        record for record in records
        if record.get("record_type") == "paper"
        and record["inclusion_decision"]["decision"] == "INCLUDE_IN_PRIMARY_EVALUATION"
    ]
    candidates_path = ROOT / "data" / "evaluation" / "openalex_title_search_candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))["results"]
    candidates_by_paper = {candidate["paper_id"]: candidate for candidate in candidates}
    method = "independent exact-title search with bibliographic field comparison"

    assert len(included) == 6
    assert (ROOT / "data" / "evaluation" / "openalex_identity_verification.json").is_file()
    for paper in included:
        candidate = candidates_by_paper[paper["paper_id"]]
        assert candidate["status"] == "MATCH"
        assert candidate["confidence"] == "HIGH"
        assert paper["external_ids"]["openalex"] == candidate["candidate_openalex_id"].rsplit("/", 1)[-1]
        assert paper["identity_verification_status"] == "VERIFIED"
        assert paper["identity_verification_method"] == method

    excluded = [
        record for record in records
        if record.get("record_type") == "paper"
        and record["inclusion_decision"]["decision"] == "EXCLUDE_FROM_PRIMARY_EVALUATION"
    ]
    assert len(excluded) == 2
    assert all(paper["external_ids"]["openalex"] is None for paper in excluded)
    assert all(paper["identity_verification_status"] == "DOI_VISIBLE_IN_PDF_OPENALEX_MAPPING_NOT_ASSESSED" for paper in excluded)


def test_manifest_rejects_changed_pdf_hash(evaluation_corpus, tmp_path):
    records = [json.loads(line) for line in evaluation_corpus.manifest.read_text(encoding="utf-8").splitlines()]
    records[1]["sha256"] = "0" * 64
    altered = tmp_path / "altered.jsonl"
    altered.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="hash mismatch"):
        load_dataset_manifest(altered, evaluation_corpus.root)


def test_manifest_rejects_unsafe_path_and_duplicate_paper_ids(evaluation_corpus, tmp_path):
    records = [json.loads(line) for line in evaluation_corpus.manifest.read_text(encoding="utf-8").splitlines()]
    records[1]["relative_path"] = "../outside.pdf"
    unsafe = tmp_path / "unsafe.jsonl"
    unsafe.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="Unsafe paper path"):
        load_dataset_manifest(unsafe, evaluation_corpus.root)

    records = [json.loads(line) for line in evaluation_corpus.manifest.read_text(encoding="utf-8").splitlines()]
    records[2]["paper_id"] = records[1]["paper_id"]
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="must be unique"):
        load_dataset_manifest(duplicate, evaluation_corpus.root)


def test_dry_runs_store_separate_no_call_records(evaluation_corpus, tmp_path):
    first = create_dry_run(evaluation_corpus.manifest, tmp_path / "runs", "validation-only", repo_root=evaluation_corpus.root)
    second = create_dry_run(evaluation_corpus.manifest, tmp_path / "runs", "validation-only", repo_root=evaluation_corpus.root)
    assert first != second and first.is_dir() and second.is_dir()
    first_record = json.loads((first / "run_manifest.json").read_text(encoding="utf-8"))
    second_record = json.loads((second / "run_manifest.json").read_text(encoding="utf-8"))
    assert first_record["status"] == "DRY_RUN_VALIDATED"
    assert first_record["dataset_hash"] == second_record["dataset_hash"]
    assert first_record["prompt_hash"] is None
    assert first_record["prompt_hash_status"] == "NOT_PROVIDED"
    assert first_record["annotation_status"] == "MISSING"


def test_imported_output_records_config_prompt_revision_and_isolated_scores(evaluation_corpus, tmp_path):
    manifest = load_dataset_manifest(evaluation_corpus.manifest, evaluation_corpus.root)
    included = [p for p in manifest.papers if p.inclusion_decision["decision"] == "INCLUDE_IN_PRIMARY_EVALUATION"]
    paper_id = included[0].paper_id
    prompt_id = "PROMPT-001"
    prompt_text = canonical_prompt(manifest, prompt_id)
    output = {
        "system_name": "single_llm", "dataset_version": manifest.header.dataset_version,
        "paper_ids": [p.paper_id for p in included], "research_prompt_id": prompt_id,
        "candidates": [{
            "gap_id": "G1", "title": "candidate", "description": "bounded",
            "citations": [{"paper_id": paper_id, "chunk_id": "C1"}],
        }],
    }
    output_path = tmp_path / "normalized.json"
    output_path.write_text(json.dumps(output), encoding="utf-8")
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    ownership_path = tmp_path / "owners.json"
    ownership_path.write_text(json.dumps({"C1": paper_id}), encoding="utf-8")

    run_dir = create_imported_run(
        output_path, evaluation_corpus.manifest, tmp_path / "runs", system_name="single_llm",
        system_configuration={"temperature": 0}, prompt_path=prompt_path,
        citation_index_path=ownership_path,
        research_prompt_id=prompt_id, canonical_prompt_text=prompt_text,
        repo_root=evaluation_corpus.root,
    )
    record = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    scores = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))
    assert record["dataset_hash"] == manifest.manifest_hash
    assert record["prompt_hash"] == hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    assert record["canonical_prompt_hash"] == hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    assert record["system_configuration"] == {"temperature": 0}
    assert record["code_revision"] is None or isinstance(record["code_revision"], str)
    assert (run_dir / "outputs.json").is_file()
    citation_score = next(score for score in scores if score["name"] == "citation_correctness_automatic")
    assert citation_score["value"] == 1.0


def test_imported_citations_without_ownership_index_fail_closed(evaluation_corpus, tmp_path):
    manifest = load_dataset_manifest(evaluation_corpus.manifest, evaluation_corpus.root)
    included = [p for p in manifest.papers if p.inclusion_decision["decision"] == "INCLUDE_IN_PRIMARY_EVALUATION"]
    prompt_id = "PROMPT-001"
    prompt_text = canonical_prompt(manifest, prompt_id)
    output_path = tmp_path / "normalized.json"
    output_path.write_text(json.dumps({
        "system_name": "baseline", "dataset_version": manifest.header.dataset_version,
        "research_prompt_id": prompt_id,
        "paper_ids": [p.paper_id for p in included], "candidates": [{
            "gap_id": "G1", "title": "t", "description": "d",
            "citations": [{"paper_id": included[0].paper_id, "chunk_id": "C1"}],
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="ownership cannot be verified"):
        create_imported_run(
            output_path, evaluation_corpus.manifest, tmp_path / "runs", system_name="baseline",
            research_prompt_id=prompt_id, canonical_prompt_text=prompt_text,
            repo_root=evaluation_corpus.root,
        )


def test_runner_rejects_unassigned_research_prompt_instead_of_inventing_mapping(evaluation_corpus, tmp_path):
    manifest = load_dataset_manifest(evaluation_corpus.manifest, evaluation_corpus.root)
    output_path = tmp_path / "normalized.json"
    output_path.write_text(json.dumps({
        "system_name": "baseline", "dataset_version": manifest.header.dataset_version,
        "research_prompt_id": "Q-UNASSIGNED", "paper_ids": [manifest.papers[0].paper_id],
        "candidates": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown finalized evaluation prompt"):
        create_imported_run(
            output_path, evaluation_corpus.manifest, tmp_path / "runs", system_name="baseline",
            research_prompt_id="Q-UNASSIGNED", canonical_prompt_text="invented",
            repo_root=evaluation_corpus.root,
        )


def test_import_rejects_mismatched_canonical_prompt_text(evaluation_corpus, tmp_path):
    with pytest.raises(ValueError, match="does not exactly match"):
        create_imported_run(
            tmp_path / "unused.json", evaluation_corpus.manifest, tmp_path / "runs",
            system_name="single_llm", research_prompt_id="PROMPT-001",
            canonical_prompt_text="different text",
            repo_root=evaluation_corpus.root,
        )


def test_multi_agent_import_requires_prompt_bound_artifacts_and_all_six_papers(evaluation_corpus, tmp_path):
    manifest = load_dataset_manifest(evaluation_corpus.manifest, evaluation_corpus.root)
    included = [p for p in manifest.papers if p.inclusion_decision["decision"] == "INCLUDE_IN_PRIMARY_EVALUATION"]
    paper_ids = [p.paper_id for p in included]
    prompt_id = "PROMPT-002"
    prompt_text = canonical_prompt(manifest, prompt_id)
    rq = {"research_topic": prompt_text, "generation_status": "COMPLETE", "errors": [], "gap_results": []}
    gaps = {"research_topic": prompt_text, "paper_ids": paper_ids, "gaps": [], "provenance": {}}
    validation = {"research_topic": prompt_text, "paper_ids": paper_ids, "gap_validations": []}
    artifact_paths = []
    for name, payload in (("rq", rq), ("gaps", gaps), ("validation", validation)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        artifact_paths.append(path)
    owners = tmp_path / "owners.json"
    owners.write_text("{}", encoding="utf-8")

    run = create_imported_run(
        None, evaluation_corpus.manifest, tmp_path / "runs", system_name="multi_agent",
        multi_agent_paths=tuple(artifact_paths), citation_index_path=owners,
        research_prompt_id=prompt_id, canonical_prompt_text=prompt_text,
        repo_root=evaluation_corpus.root,
    )
    record = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert record["research_prompt_ids"] == [prompt_id]
    assert record["canonical_prompt_hash"] == hashlib.sha256(prompt_text.encode()).hexdigest()
    assert record["evaluation_split"] is None

    for invalid_ids in (
        paper_ids[:-1],
        [*paper_ids[:-1], paper_ids[0]],
        [*paper_ids[:-1], "unexpected-paper"],
    ):
        gaps["paper_ids"] = invalid_ids
        artifact_paths[1].write_text(json.dumps(gaps), encoding="utf-8")
        with pytest.raises(ValueError, match="must contain exactly the six frozen paper IDs"):
            create_imported_run(
                None, evaluation_corpus.manifest, tmp_path / "runs", system_name="multi_agent",
                multi_agent_paths=tuple(artifact_paths), citation_index_path=owners,
                research_prompt_id=prompt_id, canonical_prompt_text=prompt_text,
                repo_root=evaluation_corpus.root,
            )
    gaps["paper_ids"] = paper_ids
    artifact_paths[1].write_text(json.dumps(gaps), encoding="utf-8")

    bundle_path = tmp_path / "langgraph-PROMPT-002.json"
    bundle_path.write_text(json.dumps({
        "evaluation_prompt_id": prompt_id,
        "canonical_prompt_text": prompt_text,
        "evaluation_chroma_dir": str(tmp_path / "multi-agent-chroma"),
        "evaluation_chroma_collection": "multi_agent_evaluation",
        "papers": [{"source_id": paper_id} for paper_id in paper_ids],
        "paper_analyses": [
            {"evidence": [{"paper_id": paper_id}]} for paper_id in paper_ids
        ],
        "comparison": {"paper_ids": paper_ids},
        "research_questions": rq,
        "gap_detection": gaps,
        "evidence_validation": validation,
    }), encoding="utf-8")
    bundle_run = create_imported_run(
        None, evaluation_corpus.manifest, tmp_path / "runs", system_name="multi_agent",
        multi_agent_bundle_path=bundle_path, citation_index_path=owners,
        research_prompt_id=prompt_id, canonical_prompt_text=prompt_text,
        repo_root=evaluation_corpus.root,
    )
    assert json.loads((bundle_run / "run_manifest.json").read_text(encoding="utf-8"))["research_prompt_ids"] == [prompt_id]

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["canonical_prompt_text"] = "a different prompt"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain the canonical prompt text"):
        create_imported_run(
            None, evaluation_corpus.manifest, tmp_path / "runs", system_name="multi_agent",
            multi_agent_bundle_path=bundle_path, citation_index_path=owners,
            research_prompt_id=prompt_id, canonical_prompt_text=prompt_text,
            repo_root=evaluation_corpus.root,
        )

    bundle["canonical_prompt_text"] = prompt_text
    bundle["papers"] = [{"source_id": paper_id} for paper_id in paper_ids[:-1]]
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="Multi-Agent bundle paper list"):
        create_imported_run(
            None, evaluation_corpus.manifest, tmp_path / "runs", system_name="multi_agent",
            multi_agent_bundle_path=bundle_path, citation_index_path=owners,
            research_prompt_id=prompt_id, canonical_prompt_text=prompt_text,
            repo_root=evaluation_corpus.root,
        )

    gaps["research_topic"] = "another prompt"
    artifact_paths[1].write_text(json.dumps(gaps), encoding="utf-8")
    with pytest.raises(ValueError, match="not generated with the canonical evaluation prompt"):
        create_imported_run(
            None, evaluation_corpus.manifest, tmp_path / "runs", system_name="multi_agent",
            multi_agent_paths=tuple(artifact_paths), citation_index_path=owners,
            research_prompt_id=prompt_id, canonical_prompt_text=prompt_text,
            repo_root=evaluation_corpus.root,
        )
