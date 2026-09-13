from dataclasses import dataclass
from typing import Optional


@dataclass
class PaperRecord:
    title: str
    authors: list[str]
    year: Optional[int]
    doi: Optional[str]
    abstract: Optional[str]
    source: str
    source_id: Optional[str]
    landing_url: Optional[str]
    pdf_url: Optional[str]