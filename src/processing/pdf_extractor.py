from pathlib import Path

import pymupdf


class PDFExtractor:
    def extract_text(self, pdf_path: str | Path) -> str:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        document = pymupdf.open(pdf_path)

        pages = []

        for page in document:
            text = page.get_text()
            pages.append(text)

        document.close()

        return "\n".join(pages)