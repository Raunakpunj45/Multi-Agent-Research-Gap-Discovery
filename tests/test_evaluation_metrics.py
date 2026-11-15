from src.evaluation.metrics import (
    score_citation_correctness,
    score_citation_support,
    score_evidence_support,
    score_gap_relevance,
    score_novelty_indication,
)
from src.evaluation.models import (
    CandidateGap,
    Citation,
    EvaluationSystemOutput,
    HumanJudgment,
    NoveltyAssessment,
    NoveltyIndication,
)


def judgment(**overrides):
    values = dict(
        annotation_id="A1", annotator_id="reviewer-1", candidate_gap_id="G1",
    )
    values.update(overrides)
    return HumanJudgment(**values)


def output(indication=NoveltyIndication.INDICATED_WITHIN_SAMPLE):
    return EvaluationSystemOutput(
        system_name="system", dataset_version="v1", paper_ids=["P1"],
        candidates=[CandidateGap(
            gap_id="G1", title="t", description="d",
            citations=[Citation(paper_id="P1", chunk_id="C1")],
            novelty=NoveltyAssessment(indication=indication),
        )],
    )


def test_gap_relevance_is_normalized_from_human_rating_and_missing_is_null():
    score = score_gap_relevance([judgment(gap_relevance=5), judgment(
        annotation_id="A2", gap_relevance=1,
    )])
    assert score.value == 0.5 and score.denominator == 2 and score.method == "HUMAN"
    assert score_gap_relevance([]).value is None


def test_evidence_and_human_citation_support_scores_use_explicit_labels():
    assert score_evidence_support([
        judgment(evidence_support="SUPPORTED"),
        judgment(annotation_id="A2", evidence_support="PARTIAL"),
        judgment(annotation_id="A3", evidence_support="UNSUPPORTED"),
    ]).value == 0.5
    assert score_citation_support([judgment(citation_support="CORRECT")]).value == 1.0


def test_automatic_citation_score_checks_existence_and_ownership_not_semantics():
    score = score_citation_correctness(output(), {"C1": "P1"})
    assert score.value == 1.0 and score.method == "AUTOMATIC"
    assert "does not judge semantic support" in score.notes[0]
    assert score_citation_correctness(output(), None).value is None
    assert score_citation_correctness(output(), {"C1": "P2"}).value == 0.0


def test_novelty_score_is_sample_bounded_and_global_overclaim_is_zero():
    current = output()
    assert score_novelty_indication(
        current, [judgment(novelty_indication="INDICATED_WITHIN_SAMPLE")]
    ).value == 1.0
    assert score_novelty_indication(
        current, [judgment(novelty_indication="GLOBAL_NOVELTY_OVERCLAIM")]
    ).value == 0.0
    abstaining = output(NoveltyIndication.NOT_ASSESSED)
    assert score_novelty_indication(
        abstaining, [judgment(novelty_indication="NOT_INDICATED_WITHIN_SAMPLE")]
    ).value == 1.0
