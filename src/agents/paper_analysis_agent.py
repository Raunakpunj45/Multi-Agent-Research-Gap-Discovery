import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.rag import RetrievedChunk


load_dotenv()


class PaperAnalysis(BaseModel):
    problem: str = Field(
        description=(
            "The main research problem or objective addressed "
            "by the paper, based only on the provided evidence."
        )
    )

    methodology: str = Field(
        description=(
            "The main methodology, model, algorithm, or approach "
            "used in the paper, based only on the provided evidence."
        )
    )

    dataset: str = Field(
        description=(
            "The datasets, experimental data, benchmarks, or "
            "evaluation resources used in the paper."
        )
    )

    results: str = Field(
        description=(
            "The main experimental or research findings reported "
            "by the paper."
        )
    )

    limitations: str = Field(
        description=(
            "Limitations, weaknesses, constraints, or unresolved "
            "issues supported by the provided evidence."
        )
    )

    future_work: str = Field(
        description=(
            "Future research directions explicitly mentioned "
            "or supported by the provided evidence."
        )
    )


class PaperAnalysisResult(BaseModel):
    analysis: PaperAnalysis = Field(
        description=(
            "Structured analysis of the research paper."
        )
    )

    evidence: list[RetrievedChunk] = Field(
        description=(
            "The retrieved evidence chunks used to produce "
            "the paper analysis."
        )
    )


class PaperAnalysisAgent:
    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = model

    def analyze(
        self,
        evidence_chunks: list[RetrievedChunk],
    ) -> PaperAnalysisResult:

        if not evidence_chunks:
            raise ValueError(
                "Evidence chunks cannot be empty."
            )

        evidence_text = self._build_evidence(
            evidence_chunks
        )

        system_instruction = (
            "You are a scientific Paper Analysis Agent. "
            "Analyze the research paper using only the "
            "provided retrieved evidence. "
            "Do not use outside knowledge. "
            "Do not invent facts, datasets, methods, results, "
            "limitations, or future work. "
            "If a requested aspect is not supported by the "
            "provided evidence, explicitly state that the "
            "information is not available in the provided evidence. "
            "Keep the analysis concise but informative."
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=(
                "Analyze the research paper from the following "
                "retrieved evidence chunks.\n\n"
                f"{evidence_text}"
            ),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=PaperAnalysis,
                temperature=0.2,
            ),
        )

        if response.parsed is None:
            raise RuntimeError(
                "Paper Analysis Agent failed to generate "
                "a structured analysis."
            )

        return PaperAnalysisResult(
            analysis=response.parsed,
            evidence=evidence_chunks,
        )

    @staticmethod
    def _build_evidence(
        evidence_chunks: list[RetrievedChunk],
    ) -> str:

        evidence_parts = []

        for index, chunk in enumerate(
            evidence_chunks,
            start=1,
        ):
            evidence_parts.append(
                f"--- Evidence Chunk {index} ---\n"
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Paper ID: {chunk.paper_id}\n"
                f"Distance: {chunk.distance}\n"
                f"Text:\n{chunk.text}"
            )

        return "\n\n".join(evidence_parts)