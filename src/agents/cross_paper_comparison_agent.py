
from pydantic import BaseModel, Field

from .paper_analysis_agent import PaperAnalysisResult


class ComparisonDimension(BaseModel):
    dimension: str = Field(
        description="Aspect being compared, such as methodology or dataset."
    )
    paper_1_observation: str
    paper_2_observation: str
    comparison: str = Field(
        description="Evidence-grounded similarity or difference."
    )
    evidence_chunk_ids: list[str] = Field(
        description="Chunk IDs supporting this comparison."
    )


class CrossPaperComparison(BaseModel):
    paper_1_id: str
    paper_2_id: str
    comparisons: list[ComparisonDimension]
    potential_research_opportunities: list[str] = Field(
        description=(
            "Tentative opportunities suggested by the comparison. "
            "These are not confirmed research gaps."
        )
    )
    caveats: list[str] = Field(
        description="Missing evidence or limitations affecting the comparison."
    )


class CrossPaperComparisonAgent:
    """Compare structured analyses without inventing unsupported gaps."""

    DIMENSIONS = (
        "problem",
        "methodology",
        "dataset",
        "results",
        "limitations",
        "future_work",
    )

    @staticmethod
    def _value(result: PaperAnalysisResult, field: str) -> str:
        return str(getattr(result.analysis, field, "")).strip()

    def compare(
        self,
        paper_1: PaperAnalysisResult,
        paper_2: PaperAnalysisResult,
    ) -> CrossPaperComparison:
        evidence_1 = {
            chunk.chunk_id for chunk in paper_1.evidence
        }
        evidence_2 = {
            chunk.chunk_id for chunk in paper_2.evidence
        }

        comparisons = []

        for dimension in self.DIMENSIONS:
            value_1 = self._value(paper_1, dimension)
            value_2 = self._value(paper_2, dimension)

            if not value_1 or not value_2:
                comparison = (
                    "Insufficient structured information to compare "
                    "this dimension reliably."
                )
            elif value_1 == value_2:
                comparison = (
                    "The structured descriptions match; this does not "
                    "establish that the papers use identical approaches."
                )
            else:
                comparison = (
                    "The structured descriptions differ. Review the cited "
                    "evidence before interpreting this as a substantive "
                    "research difference."
                )

            comparisons.append(
                ComparisonDimension(
                    dimension=dimension,
                    paper_1_observation=value_1 or "Not available",
                    paper_2_observation=value_2 or "Not available",
                    comparison=comparison,
                    evidence_chunk_ids=sorted(evidence_1 | evidence_2),
                )
            )

        return CrossPaperComparison(
            paper_1_id=paper_1.evidence[0].paper_id,
            paper_2_id=paper_2.evidence[0].paper_id,
            comparisons=comparisons,
            potential_research_opportunities=[],
            caveats=[
                "This initial version compares structured descriptions; "
                "it does not independently verify their factual correctness.",
                "A difference or missing detail is not automatically a "
                "research gap.",
                "Evidence chunk IDs identify retrieved source chunks, but "
                "do not yet map each individual claim to its exact support.",
            ],
        )
