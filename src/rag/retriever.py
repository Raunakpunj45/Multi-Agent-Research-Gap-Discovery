from src.embeddings import GeminiEmbeddingModel
from src.vectorstore import ChromaVectorStore

from .models import RetrievedChunk


class RAGRetriever:
    def __init__(
        self,
        embedding_model: GeminiEmbeddingModel | None = None,
        vector_store: ChromaVectorStore | None = None,
    ):
        self.embedding_model = (
            embedding_model
            or GeminiEmbeddingModel()
        )

        self.vector_store = (
            vector_store
            or ChromaVectorStore()
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        paper_id: str | None = None,
    ) -> list[RetrievedChunk]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_embedding = self.embedding_model.embed(
            query
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=top_k,
            paper_id=paper_id,
        )

        return self._convert_results(
            results
        )

    @staticmethod
    def _convert_results(
        results: dict,
    ) -> list[RetrievedChunk]:

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_chunks = []

        for index in range(len(ids)):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            chunk = RetrievedChunk(
                chunk_id=metadata.get(
                    "chunk_id",
                    ids[index],
                ),
                paper_id=metadata.get(
                    "paper_id",
                    "unknown",
                ),
                text=documents[index],
                distance=distance,
            )

            retrieved_chunks.append(chunk)

        return retrieved_chunks