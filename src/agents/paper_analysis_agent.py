import os
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


load_dotenv()


class PaperAnalysis(BaseModel):
    problem: str = Field(
        description=(
            "The main research problem or objective addressed "
            "by the paper."
        )
    )

    methodology: str = Field(
        description=(
            "The main methodology, model, algorithm, or approach "
            "used in the paper."
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
            "issues explicitly stated or clearly supported by "
            "the paper."
        )
    )

    future_work: str = Field(
        description=(
            "Future research directions explicitly mentioned "
            "or proposed by the paper."
        )
    )


class PaperAnalysisAgent:
    def __init__(
        self,
        model: str = "gemini-3.5-flash",
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
        paper_text: str,
    ) -> PaperAnalysis:

        if not paper_text.strip():
            raise ValueError(
                "Paper text cannot be empty."
            )

        system_instruction = (
            "You are a scientific Paper Analysis Agent. "
            "Analyze the provided research paper text and "
            "extract only information supported by the text. "
            "Do not invent facts, datasets, methods, results, "
            "limitations, or future work. "
            "If a requested aspect is not clearly available "
            "in the provided text, explicitly state that the "
            "information is not available in the provided text. "
            "Keep the analysis concise but informative."
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=(
                "Analyze the following research paper text "
                "and produce a structured analysis.\n\n"
                f"{paper_text}"
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

        return response.parsed