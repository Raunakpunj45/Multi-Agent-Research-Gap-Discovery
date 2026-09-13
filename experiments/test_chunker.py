from src.processing.pdf_extractor import PDFExtractor
from src.processing.text_cleaner import TextCleaner
from src.processing.chunker import TextChunker


extractor = PDFExtractor()
cleaner = TextCleaner()
chunker = TextChunker(
    chunk_size=300,
    overlap=50,
)

pdf_path = "data/papers/test_paper.pdf"
paper_id = "test_paper"

raw_text = extractor.extract_text(pdf_path)
cleaned_text = cleaner.clean(raw_text)

chunks = chunker.chunk(
    text=cleaned_text,
    paper_id=paper_id,
)

print(f"Total chunks: {len(chunks)}")

for chunk in chunks[:3]:
    print(f"\n--- {chunk.chunk_id} ---")
    print(f"Paper ID: {chunk.paper_id}")
    print(f"Words: {len(chunk.text.split())}")
    print(chunk.text[:500])