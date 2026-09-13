import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


class GeminiEmbeddingModel:
    def __init__(
        self,
        model: str = "gemini-embedding-001",
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Text cannot be empty.")

        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )

        return result.embeddings[0].values