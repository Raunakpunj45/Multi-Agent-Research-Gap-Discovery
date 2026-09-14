from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievedChunk:
    chunk_id: str
    paper_id: str
    text: str
    distance: Optional[float] = None    