from src.processing.pdf_extractor import PDFExtractor
from src.processing.text_cleaner import TextCleaner


extractor = PDFExtractor()
cleaner = TextCleaner()

pdf_path = "data/papers/test_paper.pdf"

raw_text = extractor.extract_text(pdf_path)
cleaned_text = cleaner.clean(raw_text)

print(f"Raw characters: {len(raw_text)}")
print(f"Cleaned characters: {len(cleaned_text)}")

print("\nFirst 1000 cleaned characters:\n")
print(cleaned_text[:1000])