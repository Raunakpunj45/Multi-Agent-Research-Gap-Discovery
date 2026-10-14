import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("GeminiEmbeddingModel", "gemini_embeddings", "gemini-embedding", "embed_content")


def python_files():
    for folder in ("src", "experiments"):
        yield from (ROOT / folder).rglob("*.py")
    yield from ROOT.glob("demo*.py")


def test_no_gemini_embedding_references_in_code():
    hits = [
        f"{path.relative_to(ROOT)}: {word}"
        for path in python_files()
        for word in FORBIDDEN
        if word in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert hits == []


def test_embedding_and_rag_imports_do_not_load_gemini_sdk():
    code = (
        "import sys, src.embeddings, src.vectorstore, src.rag; "
        "sys.exit(1 if 'google.genai' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
