import time
from typing import Optional

import requests

from .models import PaperRecord


class CrossrefRetriever:
    BASE_URL = "https://api.crossref.org/works"

    def __init__(self, timeout: int = 20, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries

    def search(self, query: str, max_results: int = 10) -> list[PaperRecord]:
        params = {
            "query": query,
            "rows": max_results,
        }

        response = self._request(params)
        data = response.json()

        items = data.get("message", {}).get("items", [])

        papers = []

        for item in items:
            title = (item.get("title") or [""])[0].lower()

            if "decision letter" in title:
                continue

            papers.append(self._parse_paper(item))

        return papers

    def _request(self, params: dict) -> requests.Response:
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