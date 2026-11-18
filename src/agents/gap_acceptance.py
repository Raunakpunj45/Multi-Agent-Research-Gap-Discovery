"""The single strict rule for when a candidate research gap may be ACCEPTED.

Shared by the Critic (deterministic pre-check and post-decision guard), the
LangGraph routing gate and the Research Question Agent, so every layer applies
the same integrity rule. A gap is accept-eligible only when:

* its gap-level validation status is FULLY_SUPPORTED;
* its joint assessment exists and is FULLY_SUPPORTED;
* the validated evidence is exactly the evidence the gap cites (when the gap is
  supplied), with no retrieval or assessment error;
* EVERY cited evidence record is SUPPORTED and carries a verified verbatim
  ``supported_contribution_quote`` (a quote is stored only after it was
  located in the chunk's own text);
* SUPPORTED, quote-verified evidence comes from at least two distinct papers;
* no unresolved contradiction remains;
* the deterministic reference checks (when recorded) passed.

PARTIALLY_SUPPORTED and UNSUPPORTED gaps therefore can never be accepted. The
rule only reads saved records; it never changes a validation status, so saved
artifacts and their fingerprints are untouched.
"""
from __future__ import annotations

from .evidence_validation_agent import (
    EvidenceValidationStatus,
    GapSupportStatus,
    GapValidation,
    GapValidationStatus,
)

MIN_ACCEPT_PAPERS = 2


def _status_name(status) -> str:
    return getattr(status, "value", None) or str(status)


def acceptance_blockers(validation: GapValidation | None, gap=None,
                        min_papers: int = MIN_ACCEPT_PAPERS) -> list[str]:
    """Reasons the gap cannot be accepted; an empty list means accept-eligible."""
    if validation is None:
        return ["No evidence validation record exists for this gap."]
    label = validation.gap_id
    blockers: list[str] = []
    if validation.status is not GapValidationStatus.FULLY_SUPPORTED:
        blockers.append(
            f"{label}: gap-level validation is {_status_name(validation.status)}, "
            "not FULLY_SUPPORTED."
        )
    joint = validation.joint_assessment
    if joint is None:
        blockers.append(f"{label}: no joint evidence assessment is recorded.")
    elif joint.status is not GapSupportStatus.FULLY_SUPPORTED:
        blockers.append(
            f"{label}: joint assessment is {_status_name(joint.status)}, not FULLY_SUPPORTED."
        )
    check = getattr(validation, "deterministic_check", None)
    if check is not None and not check.passed:
        blockers.append(f"{label}: deterministic evidence checks failed.")

    records = list(validation.evidence_validations)
    if not records:
        blockers.append(f"{label}: no cited evidence was validated.")
    if gap is not None:
        cited = [(ref.paper_id, chunk_id) for ref in gap.supporting_evidence
                 for chunk_id in ref.chunk_ids]
        validated = [(item.paper_id, item.chunk_id) for item in records]
        if set(cited) != set(validated) or len(validated) != len(set(validated)):
            blockers.append(
                f"{label}: validated evidence does not exactly match the gap's cited evidence."
            )

    supported_papers: set[str] = set()
    for item in records:
        where = f"{item.paper_id}/{item.chunk_id}"
        if item.retrieval_error or item.assessment_error:
            blockers.append(f"{label}: evidence {where} has a retrieval or assessment error.")
            continue
        if item.status is not EvidenceValidationStatus.SUPPORTED:
            blockers.append(
                f"{label}: evidence {where} is {_status_name(item.status)}, not SUPPORTED."
            )
            continue
        if not (item.supported_contribution_quote or "").strip():
            blockers.append(f"{label}: evidence {where} has no verified verbatim quote.")
            continue
        supported_papers.add(item.paper_id)
    if len(supported_papers) < min_papers:
        blockers.append(
            f"{label}: SUPPORTED, quote-verified evidence comes from "
            f"{len(supported_papers)} distinct paper(s); at least {min_papers} are required."
        )

    contradictions = list(getattr(validation, "contradictions", None) or [])
    if joint is not None:
        contradictions += list(joint.contradictions or [])
    if any(item.unresolved for item in contradictions):
        blockers.append(f"{label}: an unresolved contradiction remains.")
    return list(dict.fromkeys(blockers))


def is_accept_eligible(validation: GapValidation | None, gap=None) -> bool:
    return not acceptance_blockers(validation, gap)
