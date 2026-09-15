import time
import xml.etree.ElementTree as ET
 
import requests
 
from .models import PaperRecord
 
 
class ArxivRetriever:
    BASE_URL = "https://export.arxiv.org/api/query"
 
    def __init__(self, timeout: int = 20, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
 
    def search(
        self,
        query: str,
        max_results: int = 10,
        only_journal_conference: bool = True,
    ) -> list[PaperRecord]:
        # NOTE: arXiv is a preprint server. It has no journal/conference
        # "type" field to filter on, so only_journal_conference is accepted
        # here only for interface consistency with the other retrievers
        # (so MultiSourceRetriever can call every retriever the same way)
        # and has no effect on the results. In practice, MultiSourceRetriever
        # excludes ArxivRetriever entirely when peer_reviewed_only=True.
        params = {
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": max_results,
        }
 
        try:
            response = self._request(params)
        except RuntimeError as error:
            print(f"arXiv unavailable: {error}")
            return []
 
        root = ET.fromstring(response.text)
 
        namespace = {
            "atom": "http://www.w3.org/2005/Atom"
        }
 
        papers = []
 
        for entry in root.findall("atom:entry", namespace):
            papers.append(
                self._parse_paper(entry, namespace)
            )
 
        return papers
 
    def _request(self, params: dict) -> requests.Response:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.timeout,
                    headers={
                        "User-Agent": (
                            "Multi-Agent-Research-Gap-Discovery/1.0"
                        )
                    },
                )
 
                if response.status_code == 200:
                    return response
 
                if response.status_code == 429 or response.status_code >= 500:
                    wait_time = 10 * attempt
 
                    print(
                        f"arXiv request error ({response.status_code}). "
                        f"Retrying in {wait_time}s..."
                    )
 
                    time.sleep(wait_time)
                    continue
 
                response.raise_for_status()
 
            except requests.RequestException as error:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"arXiv request failed after "
                        f"{self.max_retries} attempts."
                    ) from error
 
                wait_time = 10 * attempt
 
                print(
                    f"Request failed. Retrying in {wait_time}s..."
                )
 
                time.sleep(wait_time)
 
        raise RuntimeError(
            "arXiv request failed after maximum retries."
        )
 
    @staticmethod
    def _parse_paper(
        entry: ET.Element,
        namespace: dict
    ) -> PaperRecord:
 
        title = entry.findtext(
            "atom:title",
            default="",
            namespaces=namespace
        ).strip()
 
        abstract = entry.findtext(
            "atom:summary",
            default="",
            namespaces=namespace
        ).strip()
 
        published = entry.findtext(
            "atom:published",
            default="",
            namespaces=namespace
        )
 
        year = int(published[:4]) if published else None
 
        authors = []
 
        for author in entry.findall("atom:author", namespace):
            name = author.findtext(
                "atom:name",
                default="",
                namespaces=namespace
            ).strip()
 
            if name:
                authors.append(name)
 
        entry_id = entry.findtext(
            "atom:id",
            default="",
            namespaces=namespace
        ).strip()
 
        arxiv_id = entry_id.rsplit("/", 1)[-1]
 
        pdf_url = None
 
        for link in entry.findall("atom:link", namespace):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break
 
        return PaperRecord(
            title=title,
            authors=authors,
            year=year,
            doi=None,
            abstract=abstract or None,
            source="arxiv",
            source_id=arxiv_id,
            landing_url=entry_id or None,
            pdf_url=pdf_url,
        )
 