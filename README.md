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
- Cross-Paper Comparison Agent
- Research Gap Detection Agent
- Evidence Validation Agent
- Critic Agent
- Research Question and Hypothesis Generation Agent

## Embeddings (local only)

Embeddings are generated locally with `sentence-transformers` using
`BAAI/bge-small-en-v1.5` (384 dimensions). Gemini embeddings have been
removed; **Gemini remains the LLM provider for model-backed agent steps**, so
`GEMINI_API_KEY` is needed when running them.

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
python experiments/run_critic.py
python experiments/run_research_questions.py
```

The Research Question and Hypothesis Generation Agent reads the saved critic,
gap-detection, and evidence-validation JSON files. It generates questions only
for gaps with a complete `ACCEPT` critic review, maps evidence labels back to
the exact validated paper/chunk references, and keeps evidence-backed
contributions separate from proposed questions and hypotheses. It withholds a
hypothesis unless the complete gap is fully supported and supported findings
come from at least two papers. Generated questions and hypotheses are proposals,
not claims of global novelty.
The runner makes one Gemini request for each eligible accepted gap.
Critic reviews record SHA-256 fingerprints of the exact comparison, gap-detection,
and evidence-validation input files. The Research Question runner rejects reviews
whose fingerprints are missing or no longer match, so rerun the critic after any
of those inputs changes. Critic review output is saved even when incomplete, and
the critic runner exits non-zero for incomplete results.

## LangGraph Workflow

The LangGraph runner orchestrates the existing agents without replacing their
implementations. It retrieves papers, analyzes them, compares the analyses,
detects and validates candidate gaps, applies bounded critic revisions, and
generates research questions only after complete evidence validation and a
critic `ACCEPT` decision. It writes a separate run artifact and does not replace
the existing stage runners or their outputs.

```powershell
python experiments/run_langgraph.py --topic "your research topic" --max-revisions 2
```

The runner performs live retrieval and model-backed work. Use mocked
`WorkflowAgents` in tests; the graph tests do not call Gemini or retrieval APIs.
For a prompt-specific evaluation workflow, select a finalized prompt ID. The
runner passes its exact manifest text as both the retrieval topic and paper
analysis query, records the ID and text in the artifact, and writes a distinct
`langgraph_run-PROMPT-XXX.json` file:

```powershell
python experiments/run_langgraph.py --evaluation-prompt-id PROMPT-001
```

## Future Scope

- Human-in-the-loop validation

## Offline Evaluation Framework

`data/evaluation/dataset_manifest.jsonl` defines the frozen six-paper primary
evaluation corpus, excludes the two medical-imaging papers, and provides three
shared finalized prompts for Single LLM, Conventional RAG, and Multi-Agent
comparisons. This evaluation/testing corpus does not change the main project
scope or objectives. No train/dev/test splits are assigned. The annotation JSONL
contains a template row only, not gold annotations or evaluation judgments.

Run either independent baseline for one prompt with:

```powershell
python experiments/run_evaluation_baseline.py --system single_llm --prompt-id PROMPT-001
python experiments/run_evaluation_baseline.py --system conventional_rag --prompt-id PROMPT-001
```

Single LLM makes one generation call over all chunks extracted from the six
local PDFs. Conventional RAG retrieves per-paper chunks from the existing
ChromaDB collection, then makes one generation call. Both import normalized
outputs through the evaluation runner and preserve the same paper IDs, prompt
IDs, and source chunk ownership checks. These commands make a live Gemini call;
the baseline tests use mocked generation and retrieval.

Validate the corpus and write an isolated no-call run record with:

```powershell
python experiments/run_evaluation.py --dry-run
```

The evaluation runner does not generate model outputs. It can import a
normalized output or existing multi-agent artifacts into a separate run
directory. `experiments/compare_evaluation_runs.py` aggregates only runs with
matching dataset, canonical prompt, and annotation protocol fingerprints.
Missing human scores remain missing, and novelty scores describe only the
frozen sample, never proof of global novelty. Imported runs require the prompt
ID and exact canonical text from the manifest. For Multi-Agent imports, use the
prompt-bound LangGraph artifact plus a Chroma-derived ownership index:

```powershell
python experiments/run_evaluation.py --export-citation-index data\evaluation\chunk_owners.json
python experiments/run_evaluation.py --multi-agent-run data\processed\langgraph_run-PROMPT-001.json --research-prompt-id PROMPT-001 --prompt path\to\PROMPT-001.txt --citation-index data\evaluation\chunk_owners.json
```

The prompt file must contain exactly the selected manifest prompt text. The
ownership-index command reads the existing indexed collection and rejects
missing or inconsistent paper/chunk metadata; it does not create records. Runs
remain unsplit, and comparison reports preserve `evaluation_split: null`. See
`data/evaluation/annotation_protocol.md` before adding human judgments.
