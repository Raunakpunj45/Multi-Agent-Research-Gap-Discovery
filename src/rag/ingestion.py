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

        for chunk in chunks:

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

        self.vector_store.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)