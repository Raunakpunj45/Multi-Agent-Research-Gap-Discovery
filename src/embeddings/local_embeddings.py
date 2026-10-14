"""Local sentence-transformers embeddings: the project's only embedding backend."""

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIMENSION = 384

# bge v1.5 retrieval instruction. Applied to queries only, never to documents.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# One loaded model per (name, device) per process, so creating several
# LocalEmbeddingModel objects (e.g. one per paper in the demo) loads it once.
_MODEL_CACHE: dict = {}


class LocalEmbeddingModel:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        batch_size: int = 32,
        query_prefix: str = BGE_QUERY_PREFIX,
        device: str | None = None,
        model=None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.device = device
        self._model = model  # injectable so unit tests never load a real model

    def _load(self):
        if self._model is None:
            key = (self.model_name, self.device)

            if key not in _MODEL_CACHE:
                # Imported lazily: importing this module must not need torch.
                from sentence_transformers import SentenceTransformer

                _MODEL_CACHE[key] = SentenceTransformer(
                    self.model_name, device=self.device
                )

            self._model = _MODEL_CACHE[key]

        return self._model

    @staticmethod
    def _check_text(text) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        vectors = (
            vectors.tolist()
            if hasattr(vectors, "tolist")
            else [list(v) for v in vectors]
        )

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding model returned {len(vectors)} vectors "
                f"for {len(texts)} texts."
            )

        for vector in vectors:
            if len(vector) != self.dimension:
                raise RuntimeError(
                    f"Embedding model '{self.model_name}' returned "
                    f"{len(vector)} dimensions, expected {self.dimension}."
                )

        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            raise TypeError("embed_documents expects a list of strings.")

        if not texts:
            return []

        for text in texts:
            self._check_text(text)

        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        self._check_text(text)

        return self._encode([self.query_prefix + text])[0]
