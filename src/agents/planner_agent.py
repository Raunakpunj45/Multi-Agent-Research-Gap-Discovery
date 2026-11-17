import os
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from src.agents.gemini_config import create_gemini_client


load_dotenv()


class SearchPlan(BaseModel):
    original_topic: str = Field(
        description=(
            "The original broad research topic provided by the user."
        )
    )

    research_domain: str = Field(
        description=(
            "The high-level research domain relevant to the topic."
        )
    )

    search_queries: List[str] = Field(
        description=(
            "Three to five precise academic search queries "
            "suitable for scholarly literature retrieval."
        )
    )

    target_modalities: List[str] = Field(
        description=(
            "Relevant data types, modalities, or research settings "
            "that should be prioritized during literature retrieval."
        )
    )

    evaluation_focus: List[str] = Field(
        description=(
            "Important aspects that should be examined across "
            "the retrieved research papers."
        )
    )


class PlannerAgent:
    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        client=None,
    ):
        # Planning is an upstream analysis-stage Gemini call (ANALYSIS key),
        # created lazily on first use.
        self._client = client

        self.model = model

    @property
    def client(self):
        if self._client is None:
            self._client = create_gemini_client("ANALYSIS")
        return self._client

    @client.setter
    def client(self, value):
        self._client = value

    def create_search_plan(
        self,
        topic: str,
    ) -> SearchPlan:

        if not topic.strip():
            raise ValueError(
                "Research topic cannot be empty."
            )

        system_instruction = (
            "You are an expert Research Planning Agent "
            "for scientific literature discovery. "
            "Your task is to convert a broad research topic "
            "into a precise academic search strategy. "
            "Generate focused search queries suitable for "
            "scholarly literature retrieval systems such as "
            "OpenAlex, Crossref, Semantic Scholar, and arXiv. "
            "Keep the queries concise, relevant, and diverse. "
            "Do not invent specific papers or factual claims."
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=(
                "Generate a structured research search plan "
                f"for the following research topic:\n\n{topic}"
            ),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=SearchPlan,
                temperature=0.2,
            ),
        )

        if response.parsed is None:
            raise RuntimeError(
                "Planner Agent failed to generate "
                "a structured search plan."
            )

        return response.parsed
