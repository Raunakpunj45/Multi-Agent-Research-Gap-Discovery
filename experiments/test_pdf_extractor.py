from src.processing.pdf_extractor import PDFExtractor


extractor = PDFExtractor()

pdf_path = "data/papers/test_paper.pdf"

text = extractor.extract_text(pdf_path)

print(f"Extracted characters: {len(text)}")
print("\nFirst 1000 characters:\n")
print(text[:1000])