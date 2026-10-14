import itertools
import time
from typing import Optional, Union

import requests

from .models import PaperRecord


class CrossrefRetriever:
    BASE_URL = "https://api.crossref.org/works"

    # Crossref answers repeated "filter" params with HTTP 400
    # (pair-list-form-invalid), so each type gets its own single-filter
    # request and the results are merged in search().
    PEER_REVIEWED_TYPES = ("journal-article", "proceedings-article")

    def __init__(self, timeout: int = 20, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries

    def search(
        self,
        query: str,
        max_results: int = 10,
        only_journal_conference: bool = True,
    ) -> list[PaperRecord]:

        types = (
            self.PEER_REVIEWED_TYPES
            if only_journal_conference
            else (None,)
        )

        # A failed request raises, so the scope is never silently narrowed
        # to a single publication type.
        batches = []

        for work_type in types:
            params = {
                "query": query,
                "rows": max_results,
            }

            if work_type:
                params["filter"] = f"type:{work_type}"

            data = self._request(params).json()

            batches.append(
                data.get("message", {}).get("items", [])
            )

        papers = []
        seen_dois = set()

        # Round-robin across types so both stay represented under the cap.
        for round_items in itertools.zip_longest(*batches):
            for item in round_items:
                if item is None or len(papers) >= max_results:
                    continue

                title = (item.get("title") or [""])[0].lower()

                if "decision letter" in title:
                    continue

                doi = (item.get("DOI") or "").lower()

                if doi:
                    if doi in seen_dois:
                        continue

                    seen_dois.add(doi)

                papers.append(self._parse_paper(item))

        return papers

    def _request(
        self,
        params: Union[dict, list[tuple[str, Union[str, int]]]],
    ) -> requests.Response:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    return response

                if response.status_code == 429 or response.status_code >= 500:
                    wait_time = 2 ** attempt
                    print(
                        f"Crossref request error ({response.status_code}). "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

            except requests.RequestException as error:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Crossref request failed after "
                        f"{self.max_retries} attempts."
                    ) from error

                wait_time = 2 ** attempt
                print(f"Request failed. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise RuntimeError("Crossref request failed after maximum retries.")

    @staticmethod
    def _parse_paper(paper: dict) -> PaperRecord:
        authors = []

        for author in paper.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")

            full_name = f"{given} {family}".strip()

            if full_name:
                authors.append(full_name)

        year = CrossrefRetriever._extract_year(paper)

        doi = paper.get("DOI")

        landing_url = (
            f"https://doi.org/{doi}"
            if doi
            else paper.get("URL")
        )

        return PaperRecord(
            title=(paper.get("title") or [""])[0],
            authors=authors,
            year=year,
            doi=doi,
            abstract=paper.get("abstract"),
            source="crossref",
            source_id=paper.get("DOI"),
            landing_url=landing_url,
            pdf_url=None,
        )

    @staticmethod
    def _extract_year(paper: dict) -> Optional[int]:
        date_info = (
            paper.get("published-print")
            or paper.get("published-online")
            or paper.get("issued")
        )

        if not date_info:
            return None

        date_parts = date_info.get("date-parts", [])

        if not date_parts or not date_parts[0]:
            return None

        return date_parts[0][0]
