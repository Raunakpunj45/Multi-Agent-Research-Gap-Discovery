"""Normalize existing pipeline outputs without changing production schemas."""
from collections.abc import Mapping
import json

from pydantic import ValidationError

from .models import (
    CandidateGap,
    Citation,
    Claim,
    EvaluationSystemOutput,
    Hypothesis,
    NoveltyAssessment,
    NoveltyIndication,
    OutputCompletionStatus,
    ResearchQuestion,
)


class AdapterError(ValueError):
    pass


def validate_citation_ownership(
    output: EvaluationSystemOutput,
    chunk_owners: Mapping[str, str],
) -> None:
    """Reject citations to unknown chunks or chunks owned by another paper."""
    for gap in output.candidates:
        citations = list(gap.citations)
        citations.extend(c for claim in gap.claims for c in claim.citations)
        citations.extend(c for question in gap.research_questions for c in question.citations)
        if gap.hypothesis:
            citations.extend(gap.hypothesis.citations)
        for citation in citations:
            actual_owner = chunk_owners.get(citation.chunk_id)
            if actual_owner is None:
                raise AdapterError(f"Unknown citation chunk_id: {citation.chunk_id}")
            if actual_owner != citation.paper_id:
                raise AdapterError(
                    f"Citation ownership mismatch for {citation.chunk_id}: "
                    f"cited as {citation.paper_id}, owned by {actual_owner}."
                )


def normalize_common_output(
    payload: dict,
    chunk_owners: Mapping[str, str] | None = None,
) -> EvaluationSystemOutput:
    """Validate a baseline's already-normalized JSON output."""
    try:
        output = EvaluationSystemOutput.model_validate(payload)
        if _citation_count(output) and chunk_owners is None:
            raise AdapterError(
                "Citation ownership cannot be verified without an explicit ownership index."
            )
        if chunk_owners is not None:
            validate_citation_ownership(output, chunk_owners)
        return output
    except ValidationError as error:
        raise AdapterError(f"Invalid normalized evaluation output: {error}") from error


def _citation_count(output: EvaluationSystemOutput) -> int:
    citations = sum(len(gap.citations) for gap in output.candidates)
    citations += sum(len(c) for gap in output.candidates for c in (claim.citations for claim in gap.claims))
    citations += sum(len(c) for gap in output.candidates for c in (q.citations for q in gap.research_questions))
    citations += sum(len(gap.hypothesis.citations) for gap in output.candidates if gap.hypothesis)
    return citations


def _citations(refs, validation_by_chunk) -> list[Citation]:
    output = []
    for ref in refs or []:
        paper_id = ref["paper_id"] if isinstance(ref, dict) else ref.paper_id
        chunk_ids = ref["chunk_ids"] if isinstance(ref, dict) else ref.chunk_ids
        for chunk_id in chunk_ids:
            item = validation_by_chunk.get(chunk_id)
            if item is None:
                raise AdapterError(f"Research Question artifact cites unvalidated chunk {chunk_id}.")
            item_paper_id = item["paper_id"] if isinstance(item, dict) else item.paper_id
            if item_paper_id != paper_id:
                raise AdapterError(
                    f"Citation ownership mismatch for {chunk_id}: "
                    f"reference says {paper_id}, validation says {item_paper_id}."
                )
            status = item.get("status") if isinstance(item, dict) else item.status
            quote = item.get("supported_contribution_quote") if isinstance(item, dict) else item.supported_contribution_quote
            output.append(Citation(
                paper_id=paper_id,
                chunk_id=chunk_id,
                quote=quote,
                validation_status=(status.value if hasattr(status, "value") else status),
                validation_limitations=[
                    value for value in (
                        (item.get("assessment_error") if isinstance(item, dict) else item.assessment_error),
                        (item.get("retrieval_error") if isinstance(item, dict) else item.retrieval_error),
                        (item.get("overgeneralization") if isinstance(item, dict) else item.overgeneralization),
                    ) if value
                ],
                source_provenance="evidence_validation",
            ))
    return output


def adapt_multi_agent_artifacts(
    research_questions: dict,
    gap_detection: dict,
    evidence_validation: dict,
    *,
    dataset_version: str,
    research_prompt_id: str | None,
    chunk_owners: Mapping[str, str] | None = None,
) -> EvaluationSystemOutput:
    """Adapt current multi-agent JSON artifacts, preserving IDs and caveats."""
    try:
        gaps = {item["gap_id"]: item for item in gap_detection["gaps"]}
        validations = {
            item["gap_id"]: item for item in evidence_validation["gap_validations"]
        }
        validation_by_gap = {}
        for validation in evidence_validation["gap_validations"]:
            validation_by_chunk = validation_by_gap.setdefault(validation["gap_id"], {})
            for item in validation["evidence_validations"]:
                previous = validation_by_chunk.get(item["chunk_id"])
                if previous and previous["paper_id"] != item["paper_id"]:
                    raise AdapterError(f"Conflicting owners for chunk {item['chunk_id']}.")
                validation_by_chunk[item["chunk_id"]] = item

        candidates = []
        for item in research_questions.get("gap_results", []):
            gap_id = item["gap_id"]
            source_gap = gaps.get(gap_id)
            source_validation = validations.get(gap_id)
            if source_gap is None or source_validation is None:
                raise AdapterError(f"No matching gap/evidence validation for {gap_id}.")
            validation_by_chunk = validation_by_gap.get(gap_id, {})
            evidence_facts = []
            for fact in item.get("evidence_backed_facts", []):
                refs = fact.get("source_evidence")
                refs = [refs] if isinstance(refs, dict) else refs or []
                citations = _citations(refs, validation_by_chunk)
                evidence_facts.append(Claim(
                    text=fact["statement"],
                    citations=citations,
                    evidence_support=fact.get("validation_status"),
                    validation_limitations=list(fact.get("validation_limitations", [])),
                ))
            questions = [ResearchQuestion(
                text=q["question"],
                rationale=q.get("rationale", ""),
                citations=_citations(q.get("source_evidence", []), validation_by_chunk),
            ) for q in item.get("research_questions", [])]
            hypothesis_raw = item.get("hypothesis")
            hypothesis = None
            if hypothesis_raw:
                hypothesis = Hypothesis(
                    alternative=hypothesis_raw["alternative_hypothesis"],
                    null=hypothesis_raw["null_hypothesis"],
                    rationale=hypothesis_raw.get("rationale", ""),
                    citations=_citations(hypothesis_raw.get("source_evidence", []), validation_by_chunk),
                    status=item.get("hypothesis_status", "GENERATED"),
                )
            source_refs = source_gap.get("supporting_evidence", [])
            gap_citations = _citations(source_refs, validation_by_chunk)
            candidates.append(CandidateGap(
                gap_id=gap_id,
                title=item.get("gap_title") or source_gap.get("title", ""),
                description=source_gap.get("description", ""),
                claims=evidence_facts,
                citations=gap_citations,
                research_questions=questions,
                hypothesis=hypothesis,
                novelty=NoveltyAssessment(
                    indication=NoveltyIndication.NOT_ASSESSED,
                    corpus_scope="No global novelty inference; assessment requires human review within the frozen dataset.",
                    limitations=["A gap candidate is not proof of global novelty."],
                ),
                source_provenance={
                    "gap_detection": "research_gap_detection.json",
                    "evidence_validation": "evidence_validation.json",
                    "research_questions": "research_questions.json",
                    "gap_detection_inputs": str(
                        json.dumps(gap_detection.get("provenance", {}).get("input_fingerprints", {}), sort_keys=True)
                    ),
                    "evidence_validation_inputs": str(
                        json.dumps(evidence_validation.get("provenance", {}).get("input_fingerprints", {}), sort_keys=True)
                    ),
                },
                limitations=list(item.get("critic_caveats", []))
                + list(source_gap.get("caveats", [])),
            ))

        output = EvaluationSystemOutput(
            system_name="multi_agent",
            dataset_version=dataset_version,
            research_prompt_id=research_prompt_id,
            paper_ids=list(gap_detection.get("paper_ids", [])),
            completion_status=OutputCompletionStatus(
                research_questions.get("generation_status", "INCOMPLETE")
            ),
            candidates=candidates,
            source_provenance={
                "generation_status": research_questions.get("generation_status", "UNKNOWN"),
                "gap_detection_status": "loaded",
                "evidence_validation_status": "loaded",
            },
            warnings=list(research_questions.get("errors", [])),
        )
    except (KeyError, TypeError, ValidationError) as error:
        raise AdapterError(f"Could not adapt multi-agent artifacts: {error}") from error

    if chunk_owners is not None:
        validate_citation_ownership(output, chunk_owners)
    return output
