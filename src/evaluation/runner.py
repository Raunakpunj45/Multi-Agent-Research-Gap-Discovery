"""Offline manifest validation and isolated evaluation-run storage."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from .adapters import (
    adapt_multi_agent_artifacts,
    normalize_common_output,
)
from .metrics import score_output
from .identity import validate_exact_paper_ids
from src.vectorstore.chroma_store import collection_name_for
from .models import (
    AnnotationRecord,
    EvaluationRunManifest,
    EvaluationSystemOutput,
    MetricScore,
    OutputCompletionStatus,
)


class DatasetHeader(BaseModel):
    model_config = {"extra": "forbid"}
    record_type: str
    dataset_version: str
    curation_status: str
    notes: str = ""
    research_prompts: list[dict] = Field(default_factory=list)
    evaluation_configuration: dict | None = None


class DatasetPaper(BaseModel):
    model_config = {"extra": "allow"}
    record_type: str
    paper_id: str
    external_ids: dict[str, str | None] = Field(default_factory=dict)
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    relative_path: str
    sha256: str
    research_prompt_ids: list[str] | None = None
    split: str | None = None
    inclusion_decision: str | dict | None = None
    curation_status: str


class DatasetManifest(BaseModel):
    header: DatasetHeader
    papers: list[DatasetPaper]
    manifest_hash: str


class ManifestValidationError(ValueError):
    pass


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_prompt(manifest: DatasetManifest, prompt_id: str) -> str:
    prompts = {
        item.get("prompt_id"): item
        for item in manifest.header.research_prompts
        if item.get("status") == "FINALIZED_SHARED_PROMPT"
    }
    prompt = prompts.get(prompt_id)
    if not prompt or not isinstance(prompt.get("text"), str) or not prompt["text"]:
        raise ValueError(f"Unknown finalized evaluation prompt: {prompt_id}")
    configured_ids = (manifest.header.evaluation_configuration or {}).get("shared_prompts", [])
    if prompt_id not in configured_ids:
        raise ValueError(f"Prompt {prompt_id} is not assigned to the frozen evaluation corpus.")
    for paper in manifest.papers:
        if (isinstance(paper.inclusion_decision, dict)
                and paper.inclusion_decision.get("decision") == "INCLUDE_IN_PRIMARY_EVALUATION"
                and prompt_id not in (paper.research_prompt_ids or [])):
            raise ValueError(f"Prompt {prompt_id} is not assigned to included paper {paper.paper_id}.")
    return prompt["text"]


def load_dataset_manifest(path: Path, repo_root: Path | None = None) -> DatasetManifest:
    path = Path(path)
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ManifestValidationError(f"Cannot read dataset manifest {path}: {error}") from error
    records = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ManifestValidationError(f"Manifest line {line_number} is invalid JSON: {error}") from error
    if len(records) < 2:
        raise ManifestValidationError("Manifest must contain one dataset header and at least one paper.")
    try:
        header = DatasetHeader.model_validate(records[0])
        if header.record_type != "dataset":
            raise ValueError("First JSONL record must have record_type='dataset'.")
        papers = [DatasetPaper.model_validate(item) for item in records[1:]]
    except (ValidationError, ValueError) as error:
        raise ManifestValidationError(f"Manifest schema error: {error}") from error
    ids = [paper.paper_id for paper in papers]
    if any(paper.record_type != "paper" for paper in papers):
        raise ManifestValidationError("Every dataset record after the header must have record_type='paper'.")
    if len(ids) != len(set(ids)):
        raise ManifestValidationError("Manifest paper_id values must be unique.")
    paths = [paper.relative_path for paper in papers]
    if len(paths) != len(set(paths)):
        raise ManifestValidationError("Manifest relative_path values must be unique.")

    for paper in papers:
        relative = Path(paper.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ManifestValidationError(f"Unsafe paper path for {paper.paper_id}: {relative}")
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ManifestValidationError(f"Paper path escapes repository root: {relative}")
        if not resolved.is_file():
            raise ManifestValidationError(f"PDF for {paper.paper_id} is missing: {relative}")
        if len(paper.sha256) != 64 or any(c not in "0123456789abcdef" for c in paper.sha256.lower()):
            raise ManifestValidationError(f"Invalid SHA-256 for {paper.paper_id}.")
        actual_hash = _sha256(resolved.read_bytes())
        if actual_hash != paper.sha256.lower():
            raise ManifestValidationError(f"PDF hash mismatch for {paper.paper_id}: {relative}")

    return DatasetManifest(
        header=header,
        papers=papers,
        manifest_hash=_sha256(payload),
    )


def load_annotation_records(path: Path) -> list[AnnotationRecord]:
    if not path.exists():
        return []
    output = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if raw.get("record_type") == "annotation_template":
                continue
            output.append(AnnotationRecord.model_validate(raw))
        except (json.JSONDecodeError, ValidationError) as error:
            raise ValueError(f"Invalid annotation line {line_number}: {error}") from error
    return output


def load_chunk_owners(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid citation ownership index: {error}") from error
    if not isinstance(raw, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                        for k, v in raw.items()):
        raise ValueError("Citation ownership index must be a JSON object mapping chunk IDs to paper IDs.")
    return raw


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


def _code_revision(root: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True, timeout=3,
        ).stdout.strip())
        return revision, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _write_run(
    output: EvaluationSystemOutput | None,
    manifest: DatasetManifest,
    *,
    runs_dir: Path,
    system_name: str,
    system_configuration: dict | None = None,
    prompt_bytes: bytes | None = None,
    canonical_prompt_hash: str | None = None,
    protocol_path: Path | None = None,
    annotations_path: Path | None = None,
    chunk_owners: dict[str, str] | None = None,
    status: str = "COMPLETE",
    warnings: list[str] | None = None,
) -> Path:
    root = Path(__file__).resolve().parents[2]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    protocol_path = protocol_path or root / "data" / "evaluation" / "annotation_protocol.md"
    protocol_hash = _sha256(protocol_path.read_bytes()) if protocol_path.is_file() else ""
    annotations = load_annotation_records(annotations_path) if annotations_path else []
    judgments = [judgment for record in annotations
                 if output is not None
                 and record.system_run_id == (output.run_id or "")
                 and record.dataset_version == output.dataset_version
                 and record.research_prompt_id == (output.research_prompt_id or "")
                 for judgment in record.human_judgments]
    metrics: list[MetricScore] = []
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_dir))
    try:
        if output is not None:
            metrics = score_output(output, judgments, chunk_owners)
            _atomic_write(staging_dir / "outputs.json", output.model_dump_json(indent=2))
            _atomic_write(staging_dir / "scores.json", json.dumps(
                [score.model_dump(mode="json") for score in metrics], indent=2,
            ))
        revision, dirty = _code_revision(root)
        run_manifest = EvaluationRunManifest(
            run_id=run_id,
            status=status,
            system_name=system_name,
            dataset_version=manifest.header.dataset_version,
            dataset_hash=manifest.manifest_hash,
            system_configuration=system_configuration or {},
            prompt_hash=_sha256(prompt_bytes) if prompt_bytes is not None else None,
            canonical_prompt_hash=canonical_prompt_hash,
            prompt_hash_status="CANONICAL" if canonical_prompt_hash is not None else "NOT_PROVIDED",
            protocol_hash=protocol_hash,
            code_revision=revision,
            working_tree_dirty=dirty,
            research_prompt_ids=(
                [output.research_prompt_id]
                if output is not None and output.research_prompt_id else []
            ),
            evaluation_split=_run_split(output, manifest),
            paper_ids=output.paper_ids if output else [paper.paper_id for paper in manifest.papers],
            annotation_status="AVAILABLE" if judgments else "MISSING",
            errors=warnings or [],
            metrics=metrics,
        )
        _atomic_write(staging_dir / "run_manifest.json", run_manifest.model_dump_json(indent=2))
        os.replace(staging_dir, run_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return run_dir


def _run_split(output: EvaluationSystemOutput | None,
               manifest: DatasetManifest) -> str | None:
    selected_ids = set(output.paper_ids) if output is not None else {
        paper.paper_id for paper in manifest.papers
    }
    splits = {paper.split for paper in manifest.papers
              if paper.paper_id in selected_ids and paper.split}
    return next(iter(splits)) if len(splits) == 1 else None


def _validate_multi_agent_bundle(bundle: dict, expected_ids: list[str]) -> None:
    evaluation_directory = bundle.get("evaluation_chroma_dir")
    evaluation_collection = bundle.get("evaluation_chroma_collection")
    conventional_directory = (Path(__file__).resolve().parents[2] / "chroma_db").resolve()
    if not isinstance(evaluation_directory, str) or Path(evaluation_directory).resolve() == conventional_directory:
        raise ValueError("Multi-Agent bundle must record an isolated evaluation Chroma directory.")
    if (not isinstance(evaluation_collection, str) or not evaluation_collection
            or evaluation_collection == collection_name_for("BAAI/bge-small-en-v1.5")):
        raise ValueError("Multi-Agent bundle must record an isolated evaluation Chroma collection.")

    papers = bundle.get("papers")
    if not isinstance(papers, list):
        raise ValueError("Multi-Agent evaluation bundle is missing its frozen paper list.")
    validate_exact_paper_ids([
        (paper.get("source_id") or paper.get("doi") or "")
        if isinstance(paper, dict) else ""
        for paper in papers
    ], expected_ids, "Multi-Agent bundle paper list")

    analyses = bundle.get("paper_analyses")
    if not isinstance(analyses, list) or len(analyses) != 6:
        raise ValueError("Multi-Agent evaluation bundle must contain six paper analyses.")
    analysis_ids = []
    for analysis in analyses:
        evidence = analysis.get("evidence") if isinstance(analysis, dict) else None
        evidence_ids = [
            item.get("paper_id", "") if isinstance(item, dict) else ""
            for item in (evidence or [])
        ]
        if not evidence_ids or len(set(evidence_ids)) != 1:
            raise ValueError("Each Multi-Agent paper analysis must cite one paper only.")
        analysis_ids.append(evidence_ids[0])
    validate_exact_paper_ids(analysis_ids, expected_ids, "Multi-Agent bundle analysis list")

    for name in ("comparison", "gap_detection", "evidence_validation"):
        artifact = bundle.get(name)
        paper_ids = artifact.get("paper_ids") if isinstance(artifact, dict) else None
        if not isinstance(paper_ids, list):
            raise ValueError(f"Multi-Agent bundle is missing {name}.paper_ids.")
        validate_exact_paper_ids(paper_ids, expected_ids, f"Multi-Agent bundle {name}")


def create_dry_run(manifest_path: Path, runs_dir: Path,
                   system_name: str = "manifest_validation", *,
                   repo_root: Path | None = None) -> Path:
    """Validate the manifest and record a no-call run.

    ``repo_root`` is the directory manifest ``relative_path`` values resolve
    against; it defaults to this repository.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    manifest = load_dataset_manifest(manifest_path, root)
    return _write_run(
        None, manifest, runs_dir=runs_dir, system_name=system_name,
        status="DRY_RUN_VALIDATED",
        warnings=(
            ["Manifest hashes are valid; research prompts, splits, and inclusion decisions remain unassigned."]
            if manifest.header.curation_status != "CURATED" else []
        ),
    )


def create_imported_run(
    output_path: Path | None,
    manifest_path: Path,
    runs_dir: Path,
    *,
    system_name: str,
    system_configuration: dict | None = None,
    prompt_path: Path | None = None,
    protocol_path: Path | None = None,
    annotations_path: Path | None = None,
    citation_index_path: Path | None = None,
    multi_agent_paths: tuple[Path, Path, Path] | None = None,
    multi_agent_bundle_path: Path | None = None,
    research_prompt_id: str | None = None,
    canonical_prompt_text: str | None = None,
    repo_root: Path | None = None,
) -> Path:
    # ``repo_root``: directory manifest paper paths resolve against (default: this repository).
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    manifest = load_dataset_manifest(manifest_path, root)
    expected_paper_ids = [
        paper.paper_id for paper in manifest.papers
        if isinstance(paper.inclusion_decision, dict)
        and paper.inclusion_decision.get("decision") == "INCLUDE_IN_PRIMARY_EVALUATION"
    ]
    validate_exact_paper_ids(expected_paper_ids, expected_paper_ids, "Frozen evaluation manifest")
    owners = load_chunk_owners(citation_index_path)
    if not research_prompt_id or canonical_prompt_text is None:
        raise ValueError("An explicit research_prompt_id and canonical prompt text are required for every imported run.")
    expected_prompt = canonical_prompt(manifest, research_prompt_id)
    if canonical_prompt_text != expected_prompt:
        raise ValueError("Canonical prompt text does not exactly match the selected finalized manifest prompt.")
    prompt_bytes = canonical_prompt_text.encode("utf-8")
    if prompt_path is not None and prompt_path.read_bytes() != prompt_bytes:
        raise ValueError("Prompt file does not exactly match the selected canonical manifest prompt.")
    if sum(bool(value) for value in (output_path, multi_agent_paths, multi_agent_bundle_path)) != 1:
        raise ValueError("Provide exactly one normalized output, Multi-Agent artifact set, or Multi-Agent run bundle.")
    multi_agent_payloads = None
    if multi_agent_bundle_path is not None:
        bundle = json.loads(Path(multi_agent_bundle_path).read_text(encoding="utf-8"))
        if not isinstance(bundle, dict):
            raise ValueError("Multi-Agent run bundle must be a JSON object.")
        if bundle.get("evaluation_prompt_id") != research_prompt_id:
            raise ValueError("Multi-Agent run bundle prompt ID does not match the requested prompt.")
        if bundle.get("canonical_prompt_text") != canonical_prompt_text:
            raise ValueError("Multi-Agent run bundle does not contain the canonical prompt text.")
        _validate_multi_agent_bundle(bundle, expected_paper_ids)
        multi_agent_payloads = (
            bundle.get("research_questions"), bundle.get("gap_detection"),
            bundle.get("evidence_validation"),
        )
    if multi_agent_paths or multi_agent_payloads:
        if owners is None:
            raise ValueError("Multi-Agent imports require a Chroma-derived citation ownership index.")
        if multi_agent_paths:
            rq_path, gaps_path, validation_path = multi_agent_paths
            rq_bytes, gaps_bytes, validation_bytes = (
                rq_path.read_bytes(), gaps_path.read_bytes(), validation_path.read_bytes(),
            )
            multi_agent_payloads = tuple(json.loads(payload.decode("utf-8")) for payload in (
                rq_bytes, gaps_bytes, validation_bytes,
            ))
        else:
            rq_bytes, gaps_bytes, validation_bytes = (
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                for payload in multi_agent_payloads
            )
        if not all(isinstance(payload, dict) for payload in multi_agent_payloads):
            raise ValueError("Multi-Agent artifacts must each be JSON objects with verifiable prompt topics.")
        for name, artifact in (("Gap Detection", multi_agent_payloads[1]),
                               ("Evidence Validation", multi_agent_payloads[2])):
            paper_ids = artifact.get("paper_ids")
            if not isinstance(paper_ids, list):
                raise ValueError(f"{name} artifact is missing paper_ids.")
            validate_exact_paper_ids(paper_ids, expected_paper_ids, name)
        output = adapt_multi_agent_artifacts(
            multi_agent_payloads[0], multi_agent_payloads[1], multi_agent_payloads[2],
            dataset_version=manifest.header.dataset_version,
            research_prompt_id=research_prompt_id,
            chunk_owners=owners,
        )
        for artifact_name, artifact in (("Research Questions", multi_agent_payloads[0]),
                                        ("Gap Detection", multi_agent_payloads[1]),
                                        ("Evidence Validation", multi_agent_payloads[2])):
            if artifact.get("research_topic") != canonical_prompt_text:
                raise ValueError(f"{artifact_name} artifact was not generated with the canonical evaluation prompt.")
        output.source_provenance.update({
            "research_questions_sha256": _sha256(rq_bytes),
            "gap_detection_sha256": _sha256(gaps_bytes),
            "evidence_validation_sha256": _sha256(validation_bytes),
        })
    else:
        if output_path is None:
            raise ValueError("--input-output is required for a normalized baseline output.")
        output = normalize_common_output(
            json.loads(output_path.read_text(encoding="utf-8")), owners,
        )
        if output.system_name != system_name:
            raise ValueError(
                f"Output system_name {output.system_name!r} does not match requested {system_name!r}."
            )
        if output.dataset_version != manifest.header.dataset_version:
            raise ValueError("Output dataset_version does not match the validated manifest.")
    valid_papers = {paper.paper_id for paper in manifest.papers}
    if output.system_name != system_name:
        raise ValueError(f"Output system_name {output.system_name!r} does not match requested {system_name!r}.")
    unknown = set(output.paper_ids) - valid_papers
    if unknown:
        raise ValueError(f"Output references papers absent from the manifest: {sorted(unknown)}")
    if output.research_prompt_id != research_prompt_id and not (multi_agent_paths or multi_agent_payloads):
        raise ValueError("Output research_prompt_id does not match the explicit selected prompt.")
    if output.research_prompt_id is None and not (multi_agent_paths or multi_agent_payloads):
        raise ValueError("Normalized output is missing its research_prompt_id.")
    output.research_prompt_id = research_prompt_id
    validate_exact_paper_ids(
        output.paper_ids, expected_paper_ids,
        "Imported output",
    )
    status = (
        "PARTIAL"
        if output.completion_status is OutputCompletionStatus.INCOMPLETE
        else "COMPLETE"
    )
    annotations = load_annotation_records(annotations_path) if annotations_path else []
    matching = [record for record in annotations
                if record.dataset_version == output.dataset_version
                and record.research_prompt_id == (output.research_prompt_id or "")
                and record.system_run_id == (output.run_id or "")]
    # Run directory ID is not known until creation; avoid pretending labels are present.
    status_warnings = list(output.warnings)
    if not matching:
        status_warnings.append("No matching human annotation records are available.")
    return _write_run(
        output, manifest, runs_dir=runs_dir, system_name=system_name,
        system_configuration=system_configuration, prompt_bytes=prompt_bytes,
        canonical_prompt_hash=_sha256(prompt_bytes),
        protocol_path=protocol_path, annotations_path=annotations_path,
        chunk_owners=owners, status=status, warnings=status_warnings,
    )
