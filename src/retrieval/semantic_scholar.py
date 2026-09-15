import os
import time
from typing import Optional
 
import requests
 
from .models import PaperRecord
 
 
class SemanticScholarRetriever:
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
 
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
            "query": query,
            "limit": max_results,
            "fields": (
                "paperId,title,authors,year,abstract,externalIds,"
                "url,openAccessPdf,publicationTypes"
            ),
        }
 
        if only_journal_conference:
            # Comma-separated values are OR'd by the Semantic Scholar API.
            params["publicationTypes"] = "JournalArticle,Conference"
 
        response = self._request(params)
        data = response.json()
 
        papers = []
 
        for item in data.get("data", []):
            papers.append(self._parse_paper(item))
 
        return papers
 
    def _request(self, params: dict) -> requests.Response:
        headers = {}
 
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
 
        if api_key:
            headers["x-api-key"] = api_key
 
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
 
                if response.status_code == 200:
                    return response
 
                if response.status_code == 429 or response.status_code >= 500:
                    wait_time = 2 ** attempt
 
                    print(
                        f"Semantic Scholar request error "
                        f"({response.status_code}). "
                        f"Retrying in {wait_time}s..."
                    )
 
                    time.sleep(wait_time)
                    continue
 
                response.raise_for_status()
 
            except requests.RequestException as error:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Semantic Scholar request failed after "
                        f"{self.max_retries} attempts."
                    ) from error
 
                wait_time = 2 ** attempt
 
                print(
                    f"Request failed. Retrying in {wait_time}s..."
                )
 
                time.sleep(wait_time)
 
        raise RuntimeError(
            "Semantic Scholar request failed after maximum retries."
        )
 
    @staticmethod
    def _parse_paper(paper: dict) -> PaperRecord:
        authors = []
 
        for author in paper.get("authors") or []:
            name = author.get("name")
 
            if name:
                authors.append(name)
 
        external_ids = paper.get("externalIds") or {}
 
        doi = external_ids.get("DOI")
 
        open_access_pdf = paper.get("openAccessPdf") or {}
 
        pdf_url = open_access_pdf.get("url")
 
        return PaperRecord(
            title=paper.get("title") or "",
            authors=authors,
            year=paper.get("year"),
            doi=doi,
            abstract=paper.get("abstract"),
            source="semantic_scholar",
            source_id=paper.get("paperId"),
            landing_url=paper.get("url"),
            pdf_url=pdf_url,
        )
 