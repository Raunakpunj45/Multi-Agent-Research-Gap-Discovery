from .models import PaperRecord


class RelevanceFilter:
    def filter(
        self,
        papers: list[PaperRecord],
        query: str,
    ) -> list[PaperRecord]:

        query_terms = set(query.lower().split())

        relevant_papers = []

        for paper in papers:
            text = " ".join(
                [
                    paper.title or "",
                    paper.abstract or "",
                ]
            ).lower()

            matches = sum(
                1 for term in query_terms
                if term in text
            )

            if matches > 0:
                relevant_papers.append(paper)

        return relevant_papers