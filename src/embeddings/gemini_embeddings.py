import os
import time

from dotenv import load_dotenv
from google import genai


load_dotenv()


# Status codes / message fragments that are worth retrying on. Gemini
# returns 429 RESOURCE_EXHAUSTED when you hit RPM/TPM/daily quota, and
# 500/502/503/504 on transient server-side issues.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRYABLE_MESSAGE_FRAGMENTS = (
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "rate limit",
    "quota",
)


def _is_retryable(error: Exception) -> bool:
    status_code = getattr(error, "code", None)

    if status_code in RETRYABLE_STATUS_CODES:
        return True

    message = str(error)

    if any(str(code) in message for code in RETRYABLE_STATUS_CODES):
        return True

    return any(
        fragment.lower() in message.lower()
        for fragment in RETRYABLE_MESSAGE_FRAGMENTS
    )


class GeminiEmbeddingModel:
    def __init__(
        self,
        model: str = "gemini-embedding-001",
        max_retries: int = 5,
        base_backoff_seconds: float = 2.0,
        max_backoff_seconds: float = 60.0,
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Text cannot be empty.")

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                )

                return result.embeddings[0].values

            except Exception as error:

                if attempt == self.max_retries or not _is_retryable(error):
                    raise

                wait_time = min(
                    self.base_backoff_seconds * (2 ** (attempt - 1)),
                    self.max_backoff_seconds,
                )

                print(
                    f"Gemini embedding request failed "
                    f"({error}). Retrying in {wait_time:.0f}s... "
                    f"(attempt {attempt}/{self.max_retries})"
                )

                time.sleep(wait_time)

        raise RuntimeError(
            "Gemini embedding request failed after maximum retries."
        )