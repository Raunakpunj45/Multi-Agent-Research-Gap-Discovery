"""Offline guard and fresh-clone fixtures for the test suite.

Every test runs with outbound network connections disabled, so a test that
accidentally reaches Gemini, OpenAlex, Crossref, Semantic Scholar, arXiv or
Hugging Face fails loudly instead of silently consuming quota. Local Unix
sockets remain available.
"""
import hashlib
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest


class NetworkAccessBlocked(RuntimeError):
    pass


_ALLOWED_FAMILIES = {getattr(socket, "AF_UNIX", None)} - {None}


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_connect(self, address):
        if self.family in _ALLOWED_FAMILIES:
            return original_connect(self, address)
        raise NetworkAccessBlocked(f"Offline test attempted a network connection to {address!r}.")

    def guarded_connect_ex(self, address):
        if self.family in _ALLOWED_FAMILIES:
            return original_connect_ex(self, address)
        raise NetworkAccessBlocked(f"Offline test attempted a network connection to {address!r}.")

    def guarded_create_connection(address, *args, **kwargs):
        raise NetworkAccessBlocked(f"Offline test attempted a network connection to {address!r}.")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    yield


_PROVIDER_ENV_PREFIXES = ("GROQ_", "LLM_PROVIDER", "GEMINI_MODEL")


@pytest.fixture(autouse=True)
def _hermetic_llm_provider_env(monkeypatch):
    """Tests start from the documented defaults, not a developer's local .env.

    Provider selection, Groq keys and model overrides are cleared, and the
    provider module does not reload .env. Tests that need a setting set it
    explicitly with monkeypatch.
    """
    import os

    for name in list(os.environ):
        if name.startswith(_PROVIDER_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("src.agents.llm_provider._load_env", lambda: None)
    # Each test starts with an empty Groq tokens-per-minute window.
    monkeypatch.setattr("src.agents.groq_client._WINDOWS", {})
    yield


# ---------------------------------------------------------------------------
# Fresh-clone fixtures. The suite never reads the git-ignored local corpus
# (data/papers, data/processed, chroma_db).
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_MANIFEST = REPO_ROOT / "data" / "evaluation" / "dataset_manifest.jsonl"
LANGGRAPH_RUN_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "langgraph_runs"

_WORDS = (
    "retrieval augmented generation grounds model answers in cited source chunks "
    "while evaluation compares single model baselines against multi agent review "
    "of evidence ownership limitations datasets methods and reported results"
).split()


def _synthetic_pdf(path: Path, paper_id: str, word_count: int = 420) -> None:
    """Write a small real PDF whose extracted text yields several chunks."""
    import pymupdf

    words = [f"Synthetic offline fixture text for {paper_id}."]
    words += [_WORDS[(index * 7 + len(paper_id)) % len(_WORDS)] for index in range(word_count)]
    lines = [" ".join(words[start:start + 12]) for start in range(0, len(words), 12)]
    document = pymupdf.open()
    page = document.new_page()
    for number, line in enumerate(lines):
        page.insert_text((36, 40 + number * 11), line, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    document.close()


@pytest.fixture(scope="session")
def evaluation_corpus(tmp_path_factory):
    """Isolated copy of the frozen evaluation corpus with synthetic PDFs.

    The tracked manifest is copied unchanged except that each ``sha256`` is
    replaced by the hash of a generated stand-in PDF written at the record's
    ``relative_path`` under a temporary root. Paper IDs, prompts, inclusion
    decisions and all other metadata come from the real manifest, so manifest,
    prompt, hash and path validation run exactly as in production.
    """
    root = tmp_path_factory.mktemp("evaluation_repo")
    records = [json.loads(line) for line in
               TRACKED_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records[1:]:
        pdf_path = root / record["relative_path"]
        _synthetic_pdf(pdf_path, record["paper_id"])
        record["sha256"] = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    manifest = root / "data" / "evaluation" / "dataset_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return SimpleNamespace(root=root, manifest=manifest)


@pytest.fixture
def langgraph_run_fixture(tmp_path):
    """Copy a sanitized saved LangGraph run into tmp_path and return (path, bytes)."""
    def copy(name: str):
        content = (LANGGRAPH_RUN_FIXTURES / name).read_bytes()
        path = tmp_path / name
        path.write_bytes(content)
        return path, content
    return copy
