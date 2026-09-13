from src.processing import (
    PDFExtractor,
    TextCleaner,
    TextChunker,
)

from src.embeddings.gemini_embeddings import GeminiEmbeddingModel
from src.vectorstore.chroma_store import ChromaVectorStore


PDF_PATH = "data/papers/test_paper.pdf"
PAPER_ID = "test_paper"


extractor = PDFExtractor()
cleaner = TextCleaner()
chunker = TextChunker(
    chunk_size=300,
    overlap=50,
)

embedding_model = GeminiEmbeddingModel()

vector_store = ChromaVectorStore(
    persist_directory="chroma_db",
    collection_name="research_papers",
)


# 1. Extract PDF text
raw_text = extractor.extract_text(PDF_PATH)

# 2. Clean text
cleaned_text = cleaner.clean(raw_text)

# 3. Create chunks
chunks = chunker.chunk(
    text=cleaned_text,
    paper_id=PAPER_ID,
)

print(f"Created {len(chunks)} chunks.")

# 4. Generate embeddings
ids = []
documents = []
embeddings = []
metadatas = []

for chunk in chunks:
    print(f"Embedding {chunk.chunk_id}...")

    ids.append(chunk.chunk_id)
    documents.append(chunk.text)

    embeddings.append(
        embedding_model.embed(chunk.text)
    )

    metadatas.append(
        {
            "paper_id": chunk.paper_id,
            "chunk_id": chunk.chunk_id,
        }
    )


# 5. Store in ChromaDB
vector_store.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
)

print(
    f"\nSuccessfully stored "
    f"{vector_store.count()} chunks in ChromaDB."
)