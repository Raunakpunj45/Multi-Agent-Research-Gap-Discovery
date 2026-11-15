"""Offline evaluation utilities for research-gap discovery systems."""

from .models import (
    AnnotationRecord,
    CandidateGap,
    Citation,
    Claim,
    EvaluationRunManifest,
    EvaluationSystemOutput,
    HumanJudgment,
    NoveltyIndication,
    OutputCompletionStatus,
    ReferenceGap,
    ResearchQuestion,
)

__all__ = [
    "AnnotationRecord", "CandidateGap", "Citation", "Claim",
    "EvaluationRunManifest", "EvaluationSystemOutput", "HumanJudgment",
    "NoveltyIndication", "OutputCompletionStatus", "ReferenceGap", "ResearchQuestion",
]
