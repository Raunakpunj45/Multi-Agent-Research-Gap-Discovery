import dataclasses
import re

from .models import PaperRecord


class PaperDeduplicator:
    def deduplicate(
        self,
        papers: list[PaperRecord],
    ) -> list[PaperRecord]:

        unique_papers = []
        by_doi = {}
        by_title = {}

        for paper in papers:
            doi = (
                self._normalize_doi(paper.doi)
                if paper.doi
                else None
            )
            title = self._normalize_title(paper.title)

            kept = (
                (by_doi.get(doi) if doi else None)
                or (by_title.get(title) if title else None)
            )

            if kept is None:
                # Copy, so merging never mutates the caller's records.
                kept = dataclasses.replace(paper)
                unique_papers.append(kept)
            else:
                self._merge(kept, paper)

            # Re-register after merge: the kept record may have gained a DOI.
            if kept.doi:
                by_doi.setdefault(
                    self._normalize_doi(kept.doi), kept
                )

            if title:
                by_title.setdefault(title, kept)

        return unique_papers

    @staticmethod
    def _merge(kept: PaperRecord, other: PaperRecord) -> None:
        """Fill fields the kept record lacks; never overwrite existing ones."""
        for name in (
            "doi",
            "year",
            "abstract",
            "landing_url",
            "pdf_url",
            "authors",
        ):
            if not getattr(kept, name) and getattr(other, name):
                setattr(kept, name, getattr(other, name))

    @staticmethod
    def _normalize_doi(doi: str) -> str:
        doi = doi.lower().strip()

        doi = doi.replace("https://doi.org/", "")
        doi = doi.replace("http://doi.org/", "")
        doi = doi.replace("doi:", "")

        return doi.strip()

    @staticmethod
    def _normalize_title(title: str) -> str:
        title = title.lower().strip()

        title = re.sub(r"[^a-z0-9\s]", "", title)
        title = re.sub(r"\s+", " ", title)

        return title
