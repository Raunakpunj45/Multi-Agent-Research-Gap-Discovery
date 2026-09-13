import re

from .models import PaperRecord


class PaperDeduplicator:
    def deduplicate(
        self,
        papers: list[PaperRecord],
    ) -> list[PaperRecord]:

        unique_papers = []
        seen_dois = set()
        seen_titles = set()

        for paper in papers:

            # DOI-based deduplication
            if paper.doi:
                doi = self._normalize_doi(paper.doi)

                if doi in seen_dois:
                    continue

                seen_dois.add(doi)

            # Title-based fallback
            title = self._normalize_title(paper.title)

            if title and title in seen_titles:
                continue

            if title:
                seen_titles.add(title)

            unique_papers.append(paper)

        return unique_papers

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