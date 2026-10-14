import time
from typing import Optional

import requests

from .models import PaperRecord


class OpenAlexRetriever:
    BASE_URL = "https://api.openalex.org/works"

    # Valid OpenAlex work types for journal articles and conference papers
    # (listed by https://api.openalex.org/work-types). The Crossref names
    # "journal-article" / "proceedings-article" are not OpenAlex type values
    # and match nothing.
    PEER_REVIEWED_TYPES = "article|conference-paper"

    def __init__(self, timeout: int = 20, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries

    def search(
        self,
        query: str,
        max_results: int = 10,
        only_journal_conference: bool = True,
    ) -> list[PaperRecord]:

        params = {
            "search": query,
            "per-page": max_results,
        }

        if only_journal_conference:
            # OR filter: journal articles OR conference papers.
            # Excludes preprints, dissertations, books, etc.
            params["filter"] = f"type:{self.PEER_REVIEWED_TYPES}"

        response = self._request(params)
        data = response.json()

        return [
            self._parse_paper(paper)
            for paper in data.get("results", [])
        ]

    def _request(self, params: dict) -> requests.Response:

        for attempt in range(1, self.max_retries + 1):

            try:
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response

                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(
                        f"OpenAlex rate limit reached. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue

                if response.status_code >= 500:
                    wait_time = 2 ** attempt
                    print(
                        f"OpenAlex server error "
                        f"({response.status_code}). "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

            except requests.RequestException as error:

                if attempt == self.max_retries:
                    raise RuntimeError(
                        "OpenAlex request failed after "
                        f"{self.max_retries} attempts."
                    ) from error

                wait_time = 2 ** attempt
                print(
                    f"Request failed. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        raise RuntimeError(
            "OpenAlex request failed after maximum retries."
        )

    def _parse_paper(self, paper: dict) -> PaperRecord:

        authors = []

        for authorship in paper.get("authorships", []):
            author = authorship.get("author")

            if author and author.get("display_name"):
                authors.append(author["display_name"])

        return PaperRecord(
            title=paper.get("display_name", ""),
            authors=authors,
            year=paper.get("publication_year"),
            doi=paper.get("doi"),
            abstract=self._reconstruct_abstract(
                paper.get("abstract_inverted_index")
            ),
            source="openalex",
            source_id=paper.get("id"),
            landing_url=paper.get("doi"),
            pdf_url=self._extract_pdf_url(paper),
        )

    @staticmethod
    def _reconstruct_abstract(
        inverted_index: Optional[dict]
    ) -> Optional[str]:

        if not inverted_index:
            return None

        words = []

        for word, positions in inverted_index.items():
            for position in positions:
                words.append((position, word))

        words.sort(key=lambda item: item[0])

        return " ".join(
            word for _, word in words
        )

    @staticmethod
    def _extract_pdf_url(paper: dict) -> Optional[str]:

        best_location = paper.get("best_oa_location")

        if not best_location:
            return None

        return best_location.get("pdf_url")
