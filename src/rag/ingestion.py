import hashlib
import time

from src.embeddings import LocalEmbeddingModel
from src.processing import PDFExtractor, TextCleaner, TextChunker
from src.vectorstore import ChromaVectorStore


class PaperIngestionPipeline:
    def __init__(
        self,
        extractor: PDFExtractor | None = None,
        cleaner: TextCleaner | None = None,
        chunker: TextChunker | None = None,
        embedding_model: LocalEmbeddingModel | None = None,
        vector_store: ChromaVectorStore | None = None,
    ):
        self.extractor = extractor or PDFExtractor()
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or TextChunker()
        self.embedding_model = (
            embedding_model
            or LocalEmbeddingModel()
        )
        self.vector_store = (
            vector_store
            or ChromaVectorStore.for_embedder(self.embedding_model)
        )

    def ingest(
        self,
        pdf_path: str,
        paper_id: str,
    ) -> int:

        if not pdf_path.strip():
            raise ValueError(
                "PDF path cannot be empty."
            )

        if not paper_id.strip():
            raise ValueError(
                "Paper ID cannot be empty."
            )

        text = self.extractor.extract_text(
            pdf_path
        )

        cleaned_text = self.cleaner.clean(
            text
        )

        chunks = self.chunker.chunk(
            text=cleaned_text,
            paper_id=paper_id,
        )

        if not chunks:
            raise ValueError(
                "No chunks were created from the PDF."
            )

        hashes = {
            chunk.chunk_id: hashlib.sha256(
                chunk.text.encode("utf-8")
            ).hexdigest()
            for chunk in chunks
        }

        # Unchanged paper already stored: skip embedding entirely.
        if self.vector_store.get_chunk_hashes(paper_id) == hashes:
            print(
                f"Paper '{paper_id}' already ingested unchanged "
                f"({len(chunks)} chunks); embedding skipped."
            )
            return len(chunks)

        # Changed or partial earlier ingest: drop its chunks so no stale
        # ones survive, then store the fresh set (stable IDs + upsert).
        self.vector_store.delete_paper(paper_id)

        started = time.perf_counter()

        embeddings = self.embedding_model.embed_documents(
            [chunk.text for chunk in chunks]
        )

        print(
            f"Embedded {len(chunks)} chunks locally in "
            f"{time.perf_counter() - started:.1f}s (batched)."
        )

        self.vector_store.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "paper_id": chunk.paper_id,
                    "chunk_id": chunk.chunk_id,
                    "text_hash": hashes[chunk.chunk_id],
                }
                for chunk in chunks
            ],
        )

        return len(chunks)
