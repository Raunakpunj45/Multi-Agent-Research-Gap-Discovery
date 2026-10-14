import re
from pathlib import Path

import chromadb


BASE_COLLECTION_NAME = "research_papers"


def collection_name_for(
    embedding_model: str,
    base_name: str = BASE_COLLECTION_NAME,
) -> str:
    """Deterministic per-model name, e.g. research_papers__bge_small_en_v1_5."""
    slug = re.sub(
        r"[^a-z0-9]+", "_", embedding_model.rsplit("/", 1)[-1].lower()
    ).strip("_")

    return f"{base_name}__{slug}"


class ChromaVectorStore:
    """One collection per embedding model.

    The model name and dimension are required and checked against the
    collection's stored metadata, so vectors from different models can never
    share a collection. The legacy "research_papers" collection (Gemini
    vectors, no metadata) is never opened or modified by this class unless
    explicitly requested, and then it is rejected by the identity check.
    """

    def __init__(
        self,
        embedding_model: str,
        embedding_dimension: int,
        persist_directory: str = "chroma_db",
        collection_name: str | None = None,
    ):
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.collection_name = (
            collection_name or collection_name_for(embedding_model)
        )

        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
            },
        )

        self._check_identity()

    @classmethod
    def for_embedder(
        cls,
        embedder,
        persist_directory: str = "chroma_db",
    ) -> "ChromaVectorStore":
        return cls(
            embedding_model=embedder.model_name,
            embedding_dimension=embedder.dimension,
            persist_directory=persist_directory,
        )

    def _check_identity(self) -> None:
        metadata = self.collection.metadata or {}

        if (
            metadata.get("embedding_model") != self.embedding_model
            or metadata.get("embedding_dimension")
            != self.embedding_dimension
        ):
            raise ValueError(
                f"Collection '{self.collection_name}' was not created for "
                f"model '{self.embedding_model}' "
                f"({self.embedding_dimension}-d). Its metadata: "
                f"model={metadata.get('embedding_model')!r}, "
                f"dimension={metadata.get('embedding_dimension')!r}. "
                f"Use the per-model collection instead of mixing vectors."
            )

    def _check_dimension(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            if len(vector) != self.embedding_dimension:
                raise ValueError(
                    f"Expected {self.embedding_dimension}-d vectors, "
                    f"got {len(vector)}-d."
                )

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:

        self._check_dimension(embeddings)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 3,
        paper_id: str | None = None,
    ) -> dict:

        self._check_dimension([query_embedding])

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }

        if paper_id:
            query_kwargs["where"] = {
                "paper_id": paper_id
            }

        return self.collection.query(
            **query_kwargs
        )

    def get_chunk_hashes(self, paper_id: str) -> dict[str, str | None]:
        """chunk_id -> text_hash for every stored chunk of a paper."""
        result = self.collection.get(
            where={"paper_id": paper_id},
            include=["metadatas"],
        )

        return {
            chunk_id: (metadata or {}).get("text_hash")
            for chunk_id, metadata in zip(
                result["ids"], result["metadatas"]
            )
        }


    def delete_stale_chunks(
        self,
        paper_id: str,
        keep_chunk_ids: list[str],
    ) -> None:
        """Delete outdated chunks while preserving the supplied chunk IDs."""
        result = self.collection.get(
            where={"paper_id": paper_id},
            include=[],
        )

        stale_ids = [
            chunk_id
            for chunk_id in result["ids"]
            if chunk_id not in set(keep_chunk_ids)
        ]

        if stale_ids:
            self.collection.delete(ids=stale_ids)



    def delete_paper(self, paper_id: str) -> None:
        self.collection.delete(
            where={"paper_id": paper_id}
        )

    def count(self) -> int:
        return self.collection.count()
