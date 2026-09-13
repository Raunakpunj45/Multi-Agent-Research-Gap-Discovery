from .models import Chunk


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 300,
        overlap: int = 50,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                "Overlap must be smaller than chunk size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        text: str,
        paper_id: str,
    ) -> list[Chunk]:

        words = text.split()
        chunks = []

        step = self.chunk_size - self.overlap

        for start in range(0, len(words), step):
            chunk_words = words[
                start:start + self.chunk_size
            ]

            if not chunk_words:
                continue

            chunk_number = len(chunks) + 1

            chunks.append(
                Chunk(
                    chunk_id=f"{paper_id}_chunk_{chunk_number}",
                    paper_id=paper_id,
                    text=" ".join(chunk_words),
                )
            )

        return chunks