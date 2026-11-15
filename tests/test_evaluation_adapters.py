import pytest

from src.evaluation.adapters import (
    AdapterError,
    adapt_multi_agent_artifacts,
    normalize_common_output,
    validate_citation_ownership,
)
from src.evaluation.models import (
    CandidateGap,
    Citation,
    EvaluationSystemOutput,
)


PAPER = "https://openalex.org/W1"
CHUNK = f"{PAPER}_chunk_1"


def artifacts():
    return (
        {
            "generation_status": "COMPLETE", "errors": [],
            "gap_results": [{
                "gap_id": "GAP-001", "gap_title": "Title",
                "critic_caveats": ["bounded"],
                "evidence_backed_facts": [{
                    "statement": "The study reports a contribution.",
                    "validation_status": "SUPPORTED",
                    "validation_limitations": ["limited to this chunk"],
                    "source_evidence": {"paper_id": PAPER, "chunk_ids": [CHUNK]},
                }],
                "research_questions": [{
                    "question": "Question?", "rationale": "Motivated by the source.",
                    "source_evidence": [{"paper_id": PAPER, "chunk_ids": [CHUNK]}],
                }],
                "hypothesis": None,
                "hypothesis_status": "INSUFFICIENT_EVIDENCE",
            }],
        },
        {
            "paper_ids": [PAPER],
            "gaps": [{
                "gap_id": "GAP-001", "title": "Title", "description": "A bounded gap.",
                "supporting_evidence": [{"paper_id": PAPER, "chunk_ids": [CHUNK]}],
            }],
            "provenance": {"input_fingerprints": {"comparison": "abc"}},
        },
        {
            "gap_validations": [{"gap_id": "GAP-001", "evidence_validations": [{
                "paper_id": PAPER, "chunk_id": CHUNK, "status": "SUPPORTED",
                "supported_contribution_quote": "Exact quote.",
                "overgeneralization": "Do not generalize beyond the study.",
            }]}],
            "provenance": {"input_fingerprints": {"gap_detection": "def"}},
        },
    )


def test_multi_agent_adapter_preserves_ids_quote_limitations_and_provenance():
    rq, gaps, validation = artifacts()
    output = adapt_multi_agent_artifacts(
        rq, gaps, validation, dataset_version="v1", research_prompt_id="Q1",
    )
    candidate = output.candidates[0]
    assert output.paper_ids == [PAPER]
    assert candidate.citations[0].chunk_id == CHUNK
    assert candidate.claims[0].citations[0].quote == "Exact quote."
    assert "Do not generalize beyond the study." in candidate.claims[0].citations[0].validation_limitations
    assert candidate.research_questions[0].citations[0].paper_id == PAPER
    assert candidate.source_provenance["gap_detection_inputs"] == '{"comparison": "abc"}'
    assert candidate.novelty.indication.value == "NOT_ASSESSED"


def test_adapter_rejects_unknown_or_wrong_owner_evidence():
    rq, gaps, validation = artifacts()
    rq["gap_results"][0]["research_questions"][0]["source_evidence"][0]["paper_id"] = "WRONG"
    with pytest.raises(AdapterError, match="ownership mismatch"):
        adapt_multi_agent_artifacts(
            rq, gaps, validation, dataset_version="v1", research_prompt_id=None,
        )


@pytest.mark.parametrize("owners,match", [
    ({}, "Unknown citation chunk_id"),
    ({CHUNK: "different-paper"}, "ownership mismatch"),
])
def test_citation_index_rejects_unknown_and_mismatched_owners(owners, match):
    output = EvaluationSystemOutput(
        system_name="baseline", dataset_version="v1", paper_ids=[PAPER],
        candidates=[CandidateGap(
            gap_id="G1", title="t", description="d",
            citations=[Citation(paper_id=PAPER, chunk_id=CHUNK)],
        )],
    )
    with pytest.raises(AdapterError, match=match):
        validate_citation_ownership(output, owners)


def test_common_baseline_output_requires_ownership_index_when_cited():
    payload = {
        "system_name": "single_llm", "dataset_version": "v1", "paper_ids": [PAPER],
        "candidates": [{
            "gap_id": "G1", "title": "t", "description": "d",
            "citations": [{"paper_id": PAPER, "chunk_id": CHUNK}],
        }],
    }
    with pytest.raises(AdapterError, match="ownership cannot be verified"):
        normalize_common_output(payload)
    assert normalize_common_output(payload, {CHUNK: PAPER}).candidates[0].citations[0].chunk_id == CHUNK
