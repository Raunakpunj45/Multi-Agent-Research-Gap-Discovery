"""Transparent automatic and human-judgment metrics for evaluation outputs."""
from collections.abc import Iterable, Mapping

from .models import EvaluationSystemOutput, HumanJudgment, MetricScore, NoveltyIndication


_SUPPORT = {"SUPPORTED": 1.0, "PARTIAL": 0.5, "PARTIALLY_SUPPORTED": 0.5,
            "UNSUPPORTED": 0.0}
_CITATION_HUMAN = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}


def _mean_score(name: str, values: list[float], method: str,
                missing_reason: str, notes: list[str] | None = None) -> MetricScore:
    if not values:
        return MetricScore(name=name, value=None, denominator=0, method=method,
                           missing_reason=missing_reason, notes=notes or [])
    total = sum(values)
    return MetricScore(name=name, value=total / len(values), numerator=total,
                       denominator=len(values), method=method, notes=notes or [])


def score_gap_relevance(judgments: Iterable[HumanJudgment]) -> MetricScore:
    """Mean human relevance on normalized 0..1 scale from a 1..5 rubric."""
    values = [(item.gap_relevance - 1) / 4 for item in judgments
              if item.gap_relevance is not None]
    return _mean_score(
        "gap_relevance_human", values, "HUMAN",
        "No human gap-relevance ratings are available.",
        ["Human rubric is 1–5; normalized to 0–1. Not an automated semantic score."],
    )


def score_evidence_support(judgments: Iterable[HumanJudgment]) -> MetricScore:
    values = [_SUPPORT[item.evidence_support.strip().upper()]
              for item in judgments
              if item.evidence_support
              and item.evidence_support.strip().upper() in _SUPPORT]
    return _mean_score(
        "evidence_support_human", values, "HUMAN",
        "No human evidence-support labels are available.",
        ["SUPPORTED=1, PARTIAL=0.5, UNSUPPORTED=0."],
    )


def score_citation_correctness(
    output: EvaluationSystemOutput,
    chunk_owners: Mapping[str, str] | None,
) -> MetricScore:
    """Automatic chunk existence and paper-ownership accuracy."""
    if chunk_owners is None:
        return MetricScore(
            name="citation_correctness_automatic", value=None, denominator=0,
            method="AUTOMATIC", missing_reason="No citation ownership index was supplied.",
            notes=["Checks chunk existence and paper ownership only; it does not judge semantic support."],
        )
    citations = []
    for gap in output.candidates:
        citations.extend(gap.citations)
        citations.extend(c for claim in gap.claims for c in claim.citations)
        citations.extend(c for question in gap.research_questions for c in question.citations)
        if gap.hypothesis:
            citations.extend(gap.hypothesis.citations)
    correct = [
        1.0 if chunk_owners.get(c.chunk_id) == c.paper_id else 0.0
        for c in citations
    ]
    return _mean_score(
        "citation_correctness_automatic", correct, "AUTOMATIC",
        "The output contains no citations to check.",
        ["Checks chunk existence and paper ownership only; it does not judge semantic support."],
    )


def score_citation_support(judgments: Iterable[HumanJudgment]) -> MetricScore:
    values = [_CITATION_HUMAN[item.citation_support.strip().upper()]
              for item in judgments
              if item.citation_support
              and item.citation_support.strip().upper() in _CITATION_HUMAN]
    return _mean_score(
        "citation_support_human", values, "HUMAN",
        "No human citation-support ratings are available.",
        ["Human judgment evaluates whether the cited passage supports its claim."],
    )


def score_novelty_indication(
    output: EvaluationSystemOutput,
    judgments: Iterable[HumanJudgment],
) -> MetricScore:
    """Score bounded, sample-local novelty language; never global novelty."""
    by_gap = {candidate.gap_id: candidate for candidate in output.candidates}
    values = []
    for judgment in judgments:
        candidate = by_gap.get(judgment.candidate_gap_id)
        label = (judgment.novelty_indication or "").strip().upper()
        if candidate is None or label not in {
            "INDICATED_WITHIN_SAMPLE", "NOT_INDICATED_WITHIN_SAMPLE",
            "GLOBAL_NOVELTY_OVERCLAIM",
        }:
            continue
        system_label = candidate.novelty.indication.value
        if label == "GLOBAL_NOVELTY_OVERCLAIM":
            values.append(0.0)
        elif system_label == label:
            values.append(1.0)
        elif (system_label == NoveltyIndication.NOT_ASSESSED.value
              and label == NoveltyIndication.NOT_INDICATED_WITHIN_SAMPLE.value):
            # Explicit abstention is acceptable when reviewers find no indication.
            values.append(1.0)
        else:
            values.append(0.0)
    return _mean_score(
        "novelty_indication_human", values, "HUMAN",
        "No human sample-local novelty-indication judgments are available.",
        ["Measures boundedness within the evaluated corpus; it cannot establish global novelty."],
    )


def score_output(output: EvaluationSystemOutput,
                 judgments: Iterable[HumanJudgment],
                 chunk_owners: Mapping[str, str] | None) -> list[MetricScore]:
    judgments = list(judgments)
    return [
        score_gap_relevance(judgments),
        score_evidence_support(judgments),
        score_citation_correctness(output, chunk_owners),
        score_citation_support(judgments),
        score_novelty_indication(output, judgments),
    ]
