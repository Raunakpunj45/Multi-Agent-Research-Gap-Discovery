import time

from src.embeddings import GeminiEmbeddingModel
from src.processing import PDFExtractor, TextCleaner, TextChunker
from src.vectorstore import ChromaVectorStore


class PaperIngestionPipeline:
    def __init__(
        self,
        extractor: PDFExtractor | None = None,
        cleaner: TextCleaner | None = None,
        chunker: TextChunker | None = None,
        embedding_model: GeminiEmbeddingModel | None = None,
        vector_store: ChromaVectorStore | None = None,
        embed_call_delay_seconds: float = 0.3,
    ):
        self.extractor = extractor or PDFExtractor()
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or TextChunker()
        self.embedding_model = (
            embedding_model
            or GeminiEmbeddingModel()
        )
        self.vector_store = (
            vector_store
            or ChromaVectorStore()
        )
        # Small pacing delay between successive embedding API calls.
        # This proactively spreads out requests to reduce how often the
        # free-tier RPM limit gets hit in the first place (in addition to
        # the reactive retry/backoff already inside GeminiEmbeddingModel).
        # Set to 0 to disable.
        self.embed_call_delay_seconds = embed_call_delay_seconds

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

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for index, chunk in enumerate(chunks):

            embedding = self.embedding_model.embed(
                chunk.text
            )

            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            embeddings.append(embedding)

            metadatas.append(
                {
                    "paper_id": chunk.paper_id,
                    "chunk_id": chunk.chunk_id,
                }
            )

            is_last_chunk = index == len(chunks) - 1

            if self.embed_call_delay_seconds > 0 and not is_last_chunk:
                time.sleep(self.embed_call_delay_seconds)

        self.vector_store.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)