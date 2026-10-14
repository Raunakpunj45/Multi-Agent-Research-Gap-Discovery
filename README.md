# Multi-Agent Research Gap Discovery

## Project

Evidence-Grounded Multi-Agent Framework for Automated Research Gap Discovery with Cross-Paper Analysis and Human-in-the-Loop Validation.

## Current Development Scope

- Research paper retrieval (OpenAlex, Crossref, Semantic Scholar)
- Planner/Retrieval Agent
- PDF extraction, text cleaning and chunking
- Local embeddings and ChromaDB vector storage
- Semantic retrieval and basic RAG
- Paper Analysis Agent

## Embeddings (local only)

Embeddings are generated locally with `sentence-transformers` using
`BAAI/bge-small-en-v1.5` (384 dimensions). Gemini embeddings have been
removed; **Gemini is still used as an LLM** by the Planner and Paper Analysis
agents, so `GEMINI_API_KEY` is still needed for those.

- The model is downloaded from Hugging Face on first use and cached locally.
  Installing `sentence-transformers` also installs PyTorch, which is a large
  download. Check disk space and RAM on your machine before installing.
- Documents are embedded in batches; queries use the model's retrieval prefix.
- Vectors are stored in a per-model ChromaDB collection named
  `research_papers__bge_small_en_v1_5` (cosine distance). The collection records
  its model and dimension and refuses to open with a different one.
- The old `research_papers` collection (Gemini vectors) is not touched or reused.
- Ingestion is idempotent: re-ingesting an unchanged paper skips embedding; a
  changed paper replaces its old chunks.

## Usage

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -v          # always pass "tests": experiments/ run live code
python experiments/smoke_local_embeddings.py data\papers\test_paper.pdf
```

## Future Scope

- Cross-paper comparison
- Research gap detection
- Evidence validation
- Critic agent
- Human-in-the-loop validation
- Research question/hypothesis generation
- Evaluation
