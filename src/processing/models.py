from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    text: str