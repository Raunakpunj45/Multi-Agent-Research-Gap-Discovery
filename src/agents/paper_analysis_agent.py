import re
import time

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.agents.gemini_config import create_gemini_client
from src.rag import RetrievedChunk


load_dotenv()


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRYABLE_MESSAGE_FRAGMENTS = (
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "rate limit",
    "quota",
)


DAILY_QUOTA_NOT_ATTEMPTED = (
    "Gemini request not attempted: the daily API quota was exhausted earlier "
    "in this run (HTTP 429)."
)


def _error_code(error: Exception) -> int | None:
    """HTTP status from provider error attributes, else from the message text."""
    for candidate in (
        getattr(error, "code", None),
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        try:
            code = int(candidate)
        except (TypeError, ValueError):
            continue
        if 400 <= code <= 599:
            return code
    match = re.search(r"\b(429|5\d{2})\b", str(error))
    return int(match.group(1)) if match else None


def _is_daily_quota_exhaustion(error: Exception) -> bool:
    """True for a 429 caused by a per-day quota (it cannot clear within a run).

    Gemini reports these with quota IDs such as
    ``GenerateRequestsPerDayPerProjectPerModel``; per-minute limits
    (``...PerMinute...``) are transient and do not match.
    """
    message = str(error).casefold()
    return _error_code(error) == 429 and bool(re.search(
        r"daily|per[ _-]?day|day[ _-]?quota|quota.{0,100}(?:day|daily)|"
        r"(?:day|daily).{0,100}quota",
        message,
    ))


def _is_retryable(error: Exception) -> bool:
    """Transient errors only: per-minute rate limits (429) and 500/502/503/504.

    Daily quota exhaustion is never retryable: retrying only spends more
    requests against a quota that will not reset during the run.
    """
    if _is_daily_quota_exhaustion(error):
        return False

    status_code = getattr(error, "code", None)

    if status_code in RETRYABLE_STATUS_CODES:
        return True

    message = str(error)

    if any(re.search(rf"\b{code}\b", message) for code in RETRYABLE_STATUS_CODES):
        return True

    return any(
        fragment.lower() in message.lower()
        for fragment in RETRYABLE_MESSAGE_FRAGMENTS
    )


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
        max_retries: int = 5,
        base_backoff_seconds: float = 2.0,
        max_backoff_seconds: float = 60.0,
        client=None,
    ):
        # Created lazily so constructing the pipeline (e.g. for resume paths
        # that never analyze papers) does not require the ANALYSIS key.
        self._client = client
        # Run-scoped: once a daily quota 429 is seen, later papers in the same
        # run are not sent. Cleared by begin_run() at the start of each run.
        self._daily_quota_exhausted = False

        self.model = model
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds

    @property
    def client(self):
        if self._client is None:
            self._client = create_gemini_client("ANALYSIS")
        return self._client

    def begin_run(self) -> None:
        """Start a new analysis run: clear run-scoped daily-quota state.

        Call once per run (not per paper) so a reused agent can work again
        after the provider quota resets, while a 429 within a run still stops
        the remaining papers of that run.
        """
        self._daily_quota_exhausted = False

    @client.setter
    def client(self, value):
        self._client = value

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

        parsed = self._generate_with_retry(
            system_instruction=system_instruction,
            evidence_text=evidence_text,
        )

        if parsed is None:
            raise RuntimeError(
                "Paper Analysis Agent failed to generate "
                "a structured analysis."
            )

        return PaperAnalysisResult(
            analysis=parsed,
            evidence=evidence_chunks,
        )

    def _generate_with_retry(
        self,
        system_instruction: str,
        evidence_text: str,
    ) -> PaperAnalysis | None:

        if self._daily_quota_exhausted:
            raise RuntimeError(DAILY_QUOTA_NOT_ATTEMPTED)

        for attempt in range(1, self.max_retries + 1):
            try:
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

                return response.parsed

            except Exception as error:

                if _is_daily_quota_exhaustion(error):
                    self._daily_quota_exhausted = True
                    raise

                if attempt == self.max_retries or not _is_retryable(error):
                    raise

                wait_time = min(
                    self.base_backoff_seconds * (2 ** (attempt - 1)),
                    self.max_backoff_seconds,
                )

                print(
                    f"Paper Analysis Agent request failed "
                    f"({error}). Retrying in {wait_time:.0f}s... "
                    f"(attempt {attempt}/{self.max_retries})"
                )

                time.sleep(wait_time)

        return None

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
