
import sys
from pathlib import Path

# Make the project root importable when running this script directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.embeddings import LocalEmbeddingModel
from src.processing import PDFExtractor, TextCleaner, TextChunker


PDF_PATH = (
    ROOT_DIR
    / "data"
    / "papers"
    / "10_1038_s41746_023_00811_0.pdf"
)


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Run: Get-ChildItem data\\papers"
        )

    print("Extracting PDF text...")

    extractor = PDFExtractor()
    cleaner = TextCleaner()
    chunker = TextChunker()

    text = extractor.extract_text(str(PDF_PATH))
    cleaned_text = cleaner.clean(text)
    chunks = chunker.chunk(
        text=cleaned_text,
        paper_id="token_check",
    )

    if not chunks:
        print("No chunks generated.")
        return

    print("Loading local embedding model...")

    embedding_model = LocalEmbeddingModel()

    # Use the model loader defined by LocalEmbeddingModel.
    sentence_model = embedding_model._load()
    tokenizer = sentence_model.tokenizer
    max_tokens = sentence_model.max_seq_length

    token_lengths = []

    for chunk in chunks:
        token_ids = tokenizer.encode(
            chunk.text,
            add_special_tokens=True,
            truncation=False,
        )
        token_lengths.append(len(token_ids))

    exceeding = [
        (index, length)
        for index, length in enumerate(token_lengths, start=1)
        if length > max_tokens
    ]

    print("\n========== TOKEN LENGTH REPORT ==========")
    print(f"PDF: {PDF_PATH.name}")
    print(f"Embedding model: {embedding_model.model_name}")
    print(f"Maximum sequence length: {max_tokens} tokens")
    print(f"Chunks checked: {len(token_lengths)}")
    print(f"Shortest chunk: {min(token_lengths)} tokens")
    print(f"Longest chunk: {max(token_lengths)} tokens")
    print(
        f"Average chunk length: "
        f"{sum(token_lengths) / len(token_lengths):.1f} tokens"
    )
    print(f"Chunks exceeding limit: {len(exceeding)}")
    print(
        f"Chunks within limit: "
        f"{len(token_lengths) - len(exceeding)}"
    )

    print("\nFirst 10 chunk lengths:")
    for index, length in enumerate(token_lengths[:10], start=1):
        status = "OVER LIMIT" if length > max_tokens else "OK"
        print(f"Chunk {index}: {length} tokens [{status}]")

    if exceeding:
        print("\nChunks exceeding the limit:")
        for index, length in exceeding:
            print(f"Chunk {index}: {length} tokens (limit: {max_tokens})")
    else:
        print("\nAll chunks are within the reported sequence-length limit.")

    print("=========================================")


if __name__ == "__main__":
    main()
