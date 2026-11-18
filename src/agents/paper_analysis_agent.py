import re
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import time

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.agents.llm_provider import create_llm_client, resolve_model
from src.agents.output_budget import generation_config
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
    "LLM request not attempted: the daily API quota was exhausted earlier "
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


def _provider_retry_after(error: Exception) -> float | None:
    """Read common HTTP and Google retry-delay fields without requiring a provider type."""
    for name in ("retry_after_seconds", "retry_after"):
        value = getattr(error, name, None)
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                pass

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except (TypeError, ValueError):
                try:
                    target = parsedate_to_datetime(retry_after)
                    if target.tzinfo is None:
                        target = target.replace(tzinfo=timezone.utc)
                    return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass

    details = getattr(error, "details", None)
    if callable(details):
        try:
            details = details()
        except Exception:
            details = None

    def find_retry_delay(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key).casefold().replace("_", "") == "retrydelay":
                    if isinstance(nested, (int, float)):
                        return max(0.0, float(nested))
                    if isinstance(nested, str):
                        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", nested)
                        if match:
                            return float(match.group(1))
                found = find_retry_delay(nested)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for nested in value:
                found = find_retry_delay(nested)
                if found is not None:
                    return found
        return None

    retry_delay = find_retry_delay(details)
    if retry_delay is not None:
        return retry_delay
    return None


def retry_delay_seconds(error: Exception, attempt: int, base_backoff_seconds: float,
                        max_backoff_seconds: float) -> float | None:
    """Delay before retrying a transient error, or ``None`` to fail closed.

    A provider-requested delay (HTTP ``Retry-After``, Groq rate-limit headers,
    Gemini ``retryDelay``) is honoured exactly; it is never shortened. If it is
    longer than ``max_backoff_seconds`` the caller must stop instead of
    retrying early. Without provider guidance, bounded exponential backoff is
    used as before.
    """
    provider_delay = _provider_retry_after(error)
    if provider_delay is not None:
        return provider_delay if provider_delay <= max_backoff_seconds else None
    return min(base_backoff_seconds * (2 ** (attempt - 1)), max_backoff_seconds)


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
        model: str | None = None,
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

        self.model = model or resolve_model("ANALYSIS")
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds

    @property
    def client(self):
        if self._client is None:
            self._client = create_llm_client("ANALYSIS")
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
                    config=generation_config(
                        "paper_analysis",
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

                # Honour a provider Retry-After exactly; fail closed when it
                # exceeds max_backoff_seconds instead of retrying too early.
                wait_time = retry_delay_seconds(
                    error, attempt, self.base_backoff_seconds,
                    self.max_backoff_seconds,
                )
                if wait_time is None:
                    raise

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
            # Only the chunk text is sent. Chunk/paper IDs and retrieval
            # distances are not needed by the model (the analysis cites no
            # IDs; the exact chunks are attached to the result in code), so
            # omitting them saves tokens without removing any evidence text.
            # Whitespace is collapsed; the words are unchanged.
            evidence_parts.append(
                f"--- Evidence Chunk {index} ---\n"
                f"{' '.join(chunk.text.split())}"
            )

        return "\n\n".join(evidence_parts)
