"""Paper and gap counts derived from saved workflow state (report and UI).

All counts come from ``paper_selection`` (one entry per retrieved paper) and
the saved gap, validation, critic and question artifacts; nothing is inferred.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

ANALYZED = "analyzed"
FAILED = "failed"
BELOW_RELEVANCE = "not_attempted_below_relevance"
MAX_PAPERS = "not_attempted_max_papers"
QUOTA = "not_attempted_quota"


@dataclass(frozen=True)
class PaperCounts:
    retrieved: int
    relevant: int | None          # at or above the relevance threshold (None: unranked run)
    attempted: int
    downloaded: int               # PDF obtained (analyzed or failed during analysis)
    analyzed: int
    failed: int
    failed_download: int
    failed_analysis: int
    failed_unknown_step: int      # older checkpoints did not record the step
    skipped: int
    skipped_below_relevance: int
    skipped_max_papers: int
    skipped_quota: int

    def as_dict(self) -> dict:
        return asdict(self)


def _get(item, name, default=None):
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def paper_counts(state: dict) -> PaperCounts:
    selection = list(state.get("paper_selection") or [])
    papers = state.get("papers") or []
    analyzed_count = len(state.get("paper_analyses") or [])
    outcomes = [entry.get("outcome") for entry in selection]
    failed = [entry for entry in selection if entry.get("outcome") == FAILED]
    failed_download = sum(1 for e in failed if e.get("failure_step") == "download")
    failed_analysis = sum(1 for e in failed if e.get("failure_step") == "analysis")
    analyzed = outcomes.count(ANALYZED) if selection else analyzed_count
    below = outcomes.count(BELOW_RELEVANCE)
    max_papers = outcomes.count(MAX_PAPERS)
    quota = outcomes.count(QUOTA)
    return PaperCounts(
        retrieved=len(papers),
        relevant=(len(papers) - below) if state.get("relevance_threshold") is not None else None,
        attempted=analyzed + len(failed),
        downloaded=analyzed + failed_analysis,
        analyzed=analyzed,
        failed=len(failed),
        failed_download=failed_download,
        failed_analysis=failed_analysis,
        failed_unknown_step=len(failed) - failed_download - failed_analysis,
        skipped=below + max_papers + quota,
        skipped_below_relevance=below,
        skipped_max_papers=max_papers,
        skipped_quota=quota,
    )


def gap_rows(state: dict) -> list[dict]:
    """One row per candidate gap: validation status, critic decision, questions."""
    detection = state.get("gap_detection")
    if detection is None:
        return []
    validations = {}
    for item in _get(state.get("evidence_validation"), "gap_validations", []) or []:
        validations[_get(item, "gap_id")] = item
    reviews = {}
    for item in _get(state.get("critic_result"), "gap_reviews", []) or []:
        reviews[_get(item, "gap_id")] = item
    questions = {}
    for item in _get(state.get("research_questions"), "gap_results", []) or []:
        questions[_get(item, "gap_id")] = item
    rows = []
    for gap in _get(detection, "gaps", []) or []:
        gap_id = _get(gap, "gap_id")
        validation, review, result = (validations.get(gap_id), reviews.get(gap_id),
                                      questions.get(gap_id))
        status = _get(validation, "status")
        decision = _get(review, "decision")
        hypothesis_status = _get(result, "hypothesis_status")
        rows.append({
            "gap_id": gap_id,
            "title": _get(gap, "title"),
            "papers": len(_get(gap, "supporting_paper_ids", []) or []),
            "validation": getattr(status, "value", status),
            "critic": getattr(decision, "value", decision),
            "questions": len(_get(result, "research_questions", []) or []) if result else 0,
            "hypothesis": getattr(hypothesis_status, "value", hypothesis_status),
        })
    return rows
