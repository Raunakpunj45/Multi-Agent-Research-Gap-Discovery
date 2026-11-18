import os
import tempfile
from pathlib import Path

import requests

PDF_SIGNATURE = b"%PDF-"


class PDFDownloadError(ValueError):
    """A PDF could not be obtained (HTTP error, not a PDF, empty body)."""


class PDFDownloader:
    def __init__(self, output_dir: str = "data/papers", timeout: int = 30,
                 session=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._session = session  # injectable for offline tests

    def download(
        self,
        pdf_url: str,
        paper_id: str,
    ) -> Path:
        """Download one PDF and store it atomically under ``output_dir``.

        A response counts as a PDF when its body starts with the PDF signature
        (``%PDF-``), whatever its Content-Type: repositories often serve real
        PDFs as ``application/octet-stream``, and HTML landing pages are
        sometimes labelled ``application/pdf``. Nothing is written unless the
        whole body was received and is a PDF, so a failed or partial download
        never leaves a file that later runs would reuse.
        """

        if not pdf_url:
            raise ValueError("PDF URL is required.")

        output_path = self.output_dir / f"{paper_id}.pdf"

        response = (self._session or requests).get(
            pdf_url,
            timeout=self.timeout,
            headers={
                "User-Agent": "Multi-Agent-Research-Gap-Discovery/1.0"
            },
        )

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        content = response.content or b""

        if not content:
            raise PDFDownloadError(
                f"URL returned an empty body. Content-Type: {content_type or 'unknown'}"
            )

        if not content.lstrip()[:len(PDF_SIGNATURE)] == PDF_SIGNATURE:
            raise PDFDownloadError(
                f"URL did not return a PDF. "
                f"Content-Type: {content_type or 'unknown'}"
            )

        handle, temporary = tempfile.mkstemp(
            dir=self.output_dir, prefix=f".{paper_id}.", suffix=".part")
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(content)
            os.replace(temporary, output_path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

        return output_path
