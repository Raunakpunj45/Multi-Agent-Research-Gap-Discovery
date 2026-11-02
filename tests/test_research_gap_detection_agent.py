from types import SimpleNamespace

import pytest

from src.agents import (
    ComparativeFinding,
    EvidenceRef,
    MultiPaperComparison,
    UnderexploredDirection,
)
from src.agents.research_gap_detection_agent import (
    GapCategory,
    GapConfidence,
    GapOrigin,
    LLMGap,
    LLMGapDetection,
    LLMGapEvidenceRef,
    ResearchGap,
    ResearchGapDetectionAgent,
    ResearchGapDetectionResult,
)

# IDs deliberately look like real IDs (URLs / DOIs), not P1/P2.
IDS = ["https://openalex.org/W1", "10.1/abc", "paper three"]


def chunk(paper_id, n):
    return f"{paper_id}_chunk_{n}"


def evidence(*pairs):
    return [EvidenceRef(paper_id=p, chunk_ids=[chunk(p, n) for n in ns]) for p, ns in pairs]


def make_comparison(paper_ids=IDS, with_findings=True):
    limitations = []
    directions = []

    if with_findings:
        limitations = [
            ComparativeFinding(
                title="Small evaluations",
                description="Evaluations are small.",
                dimensions=["limitations"],
                evidence=evidence((IDS[0], [1, 2]), (IDS[1], [1])),
            )
        ]
        directions = [
            UnderexploredDirection(
                direction="Expert validation",
                rationale="Papers call for validation.",
                evidence=evidence((IDS[1], [2]), (IDS[2], [1])),
            )
        ]

    return MultiPaperComparison(
        paper_ids=list(paper_ids),
        paper_count=len(paper_ids),
        dimension_summaries=[],
        recurring_patterns=[],
        common_limitations=limitations,
        conflicting_findings=[],
        methodological_differences=[],
        underexplored_directions=directions,
        caveats=[],
    )


def ref(paper, *chunks):
    return LLMGapEvidenceRef(paper_label=paper, evidence_labels=list(chunks))


def llm_gap(**overrides):
    base = dict(
        title="Validation of generated hypotheses",
        description="No shared protocol validates generated hypotheses.",
        gap_category="Methodological Gap",
        reasoning="Two papers rely on informal expert review.",
        evidence=[ref("P1", "P1-E1"), ref("P2", "P2-E1")],
        evidence_summary="P1 and P2 report small evaluations.",
        gap_origin="inferred_cross_paper",
        confidence="medium",
        confidence_reason="Two papers agree.",
        research_opportunity="Tentatively, design a validation protocol.",
    )
    base.update(overrides)
    return LLMGap(**base)


class FakeClient:
    def __init__(self, parsed):
        self.prompts = []
        self.models = SimpleNamespace(generate_content=self._generate)
        self._parsed = parsed

    def _generate(self, model, contents, config):
        self.prompts.append(contents)
        return SimpleNamespace(parsed=self._parsed)


class ExplodingClient:
    def __init__(self):
        self.models = SimpleNamespace(generate_content=self._boom)

    def _boom(self, **kwargs):
        raise AssertionError("LLM must not be called")


def detect(gaps, caveats=None, comparison=None, **agent_kwargs):
    client = FakeClient(LLMGapDetection(gaps=gaps, caveats=caveats or []))
    agent = ResearchGapDetectionAgent(client=client, **agent_kwargs)
    result = agent.detect_gaps(comparison or make_comparison(), research_topic="LLMs")
    return result, client


def test_valid_gap_is_identified_with_real_ids():
    result, _ = detect([llm_gap()])

    assert result.paper_count == 3 and result.paper_ids == IDS
    assert result.research_topic == "LLMs"
    assert len(result.gaps) == 1

    gap = result.gaps[0]
    assert gap.gap_id == "GAP-001"
    assert gap.gap_category is GapCategory.METHODOLOGICAL
    assert gap.gap_origin is GapOrigin.INFERRED
    assert gap.confidence is GapConfidence.MEDIUM
    assert gap.supporting_paper_ids == [IDS[0], IDS[1]]
    assert gap.supporting_chunk_ids == [chunk(IDS[0], 1), chunk(IDS[1], 1)]
    assert result.gap_category_counts == {"Methodological Gap": 1}
    assert any("novelty" in c for c in result.caveats)
    assert any(
        "Evidence Validation Agent" in c and "ID validity only" in c
        for c in result.caveats
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Methodological Gap", GapCategory.METHODOLOGICAL),
        ("empirical", GapCategory.EMPIRICAL),
        ("THEORETICAL GAP", GapCategory.THEORETICAL),
        ("Dataset / Evaluation Gap", GapCategory.DATASET_EVALUATION),
        ("application / domain", GapCategory.APPLICATION_DOMAIN),
        ("Integration Gap", GapCategory.INTEGRATION),
    ],
)
def test_all_six_categories_are_classified(text, expected):
    result, _ = detect([llm_gap(gap_category=text)])
    assert result.gaps[0].gap_category is expected


def test_invalid_category_is_excluded_not_guessed():
    result, _ = detect([llm_gap(gap_category="Vibes Gap")])

    assert result.gaps == []
    assert result.excluded_candidates[0].reason.startswith("Invalid")


def test_single_paper_limitation_does_not_become_a_gap():
    only_p1 = llm_gap(evidence=[ref("P1", "P1-E1", "P1-E2")])
    result, _ = detect([only_p1])

    assert result.gaps == []
    assert "single-paper" in result.excluded_candidates[0].reason
    assert any("No gap was sufficiently supported" in c for c in result.caveats)


def test_insufficient_or_invented_evidence_is_excluded():
    invented_chunk = llm_gap(title="a", evidence=[ref("P1", "P1-E9"), ref("P2", "P2-E1")])
    invented_paper = llm_gap(title="b", evidence=[ref("P9", "P9-E1"), ref("P1", "P1-E1")])
    wrong_owner = llm_gap(title="c", evidence=[ref("P2", "P1-E1"), ref("P3", "P3-E1")])
    no_evidence = llm_gap(title="d", evidence=[])

    result, _ = detect([invented_chunk, invented_paper, wrong_owner, no_evidence])

    assert result.gaps == []
    assert len(result.excluded_candidates) == 4
    assert any("excluded" in c for c in result.caveats)


def test_ids_are_preserved_exactly_and_model_never_sees_them():
    result, client = detect(
        [llm_gap(evidence=[ref("P1", "P1-E2"), ref("P2", "P2-E1", "P2-E2"), ref("P3", "P3-E1")])]
    )

    gap = result.gaps[0]
    assert gap.supporting_paper_ids == IDS
    assert gap.supporting_chunk_ids == [
        chunk(IDS[0], 2), chunk(IDS[1], 1), chunk(IDS[1], 2), chunk(IDS[2], 1)]
    assert all(pid.startswith(("https://", "10.", "paper")) for pid in gap.supporting_paper_ids)

    prompt = client.prompts[0]
    assert "P1-E1" in prompt
    assert "openalex.org" not in prompt and "10.1/abc" not in prompt and "_chunk_" not in prompt


def test_structured_evidence_preserves_paper_to_chunk_mapping():
    result, _ = detect(
        [llm_gap(evidence=[
            ref("P2", "P2-E2", "P2-E1"),
            ref("P1", "P1-E2"),
            ref("P1", "P1-E1", "P1-E2"),  # same paper again: merged, no dupes
            ref("P3", "P3-E1", "P1-E1"),  # P1-E1 does not belong to P3: dropped
        ])]
    )

    gap = result.gaps[0]
    assert all(isinstance(r, EvidenceRef) for r in gap.supporting_evidence)
    assert {r.paper_id: r.chunk_ids for r in gap.supporting_evidence} == {
        IDS[1]: [chunk(IDS[1], 2), chunk(IDS[1], 1)],
        IDS[0]: [chunk(IDS[0], 2), chunk(IDS[0], 1)],
        IDS[2]: [chunk(IDS[2], 1)],
    }
    # Every chunk sits under the paper it belongs to.
    for r in gap.supporting_evidence:
        assert all(c.startswith(r.paper_id + "_chunk_") for c in r.chunk_ids)
    # Flat fields stay for backward compatibility and agree with the mapping.
    assert gap.supporting_paper_ids == [r.paper_id for r in gap.supporting_evidence]
    assert gap.supporting_chunk_ids == [
        c for r in gap.supporting_evidence for c in r.chunk_ids]

    restored = ResearchGapDetectionResult.model_validate_json(result.model_dump_json())
    assert restored.gaps[0].supporting_evidence == gap.supporting_evidence


def test_high_confidence_is_capped_with_two_papers():
    result, _ = detect([llm_gap(confidence="high")])

    gap = result.gaps[0]
    assert gap.confidence is GapConfidence.MEDIUM
    assert "Capped" in gap.confidence_reason

    three = llm_gap(
        confidence="high",
        evidence=[ref("P1", "P1-E1"), ref("P2", "P2-E1"), ref("P3", "P3-E1")],
    )
    result, _ = detect([three])
    assert result.gaps[0].confidence is GapConfidence.HIGH


def test_duplicate_titles_and_missing_fields_are_dropped():
    result, _ = detect([
        llm_gap(),
        llm_gap(title="  validation OF generated hypotheses!  "),
        llm_gap(title="Blank reasoning", reasoning="   "),
    ])

    assert [g.gap_id for g in result.gaps] == ["GAP-001"]
    assert len(result.excluded_candidates) == 2


def test_empty_comparison_returns_no_gaps_without_calling_llm():
    agent = ResearchGapDetectionAgent(client=ExplodingClient())
    result = agent.detect_gaps(make_comparison(with_findings=False))

    assert result.gaps == [] and result.gap_category_counts == {}
    assert result.paper_count == 3
    assert any("no findings" in c for c in result.caveats)


def test_invalid_inputs():
    agent = ResearchGapDetectionAgent(client=FakeClient(None))

    with pytest.raises(TypeError):
        agent.detect_gaps({"not": "a comparison"})
    with pytest.raises(ValueError, match="two"):
        agent.detect_gaps(make_comparison(paper_ids=["only"], with_findings=False))
    with pytest.raises(RuntimeError):
        agent.detect_gaps(make_comparison())  # parsed is None
    with pytest.raises(ValueError):
        ResearchGapDetectionAgent(min_supporting_papers=1)


def test_construction_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = ResearchGapDetectionAgent().detect_gaps(make_comparison(with_findings=False))
    assert result.gaps == []


def test_output_is_structured_pydantic_and_round_trips():
    result, _ = detect([llm_gap()], caveats=["model caveat"])

    assert isinstance(result, ResearchGapDetectionResult)
    assert isinstance(result.gaps[0], ResearchGap)
    assert "model caveat" in result.caveats

    restored = ResearchGapDetectionResult.model_validate_json(result.model_dump_json())
    assert restored == result
