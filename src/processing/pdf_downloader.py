from pathlib import Path

import requests


class PDFDownloader:
    def __init__(self, output_dir: str = "data/papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        pdf_url: str,
        paper_id: str,
    ) -> Path:

        if not pdf_url:
            raise ValueError("PDF URL is required.")

        output_path = self.output_dir / f"{paper_id}.pdf"

        response = requests.get(
            pdf_url,
            timeout=30,
            headers={
                "User-Agent": "Multi-Agent-Research-Gap-Discovery/1.0"
            },
        )

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" not in content_type:
            raise ValueError(
                f"URL did not return a PDF. "
                f"Content-Type: {content_type}"
            )

        output_path.write_bytes(response.content)

        return output_path