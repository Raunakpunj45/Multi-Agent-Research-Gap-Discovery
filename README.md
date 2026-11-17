# Multi-Agent Research Gap Discovery

**Multi-Agent Framework for Automated Research Gap Discovery and Evidence-Grounded
Hypothesis Generation.**

The system retrieves research papers, extracts paper-level information, compares
papers, proposes candidate research gaps, validates each gap against the exact
source chunks it cites, has a Critic Agent review (and automatically revise) the
reasoning chain, and generates research questions — plus hypotheses only when the
evidence is sufficient. Citations, provenance, and evidence statuses are preserved
end to end. Comparative evaluation against a single-LLM and a conventional-RAG
baseline is one supported objective (see *Offline Evaluation Framework*), not the
whole project.

## Architecture

```
src/
  agents/        the agents, artifact provenance, Gemini key configuration, artifact I/O
  workflow/      LangGraph orchestration (routing and provenance handoffs only)
  retrieval/     OpenAlex, Crossref, Semantic Scholar, arXiv clients + deduplication
  processing/    PDF download/extraction, cleaning, chunking
  embeddings/    local sentence-transformers embeddings (BAAI/bge-small-en-v1.5)
  vectorstore/   ChromaDB store (per-model collection)
  rag/           ingestion and retrieval of paper chunks
  evaluation/    frozen-corpus evaluation, baselines, metrics, citation checks
experiments/     command-line runners (these make LIVE API calls)
tests/           offline pytest suite (network is blocked)
data/            papers, processed artifacts, evaluation corpus, Chroma stores
```

### Agents and Gemini key ownership

| Agent | Responsibility | Key group |
|---|---|---|
| Planner / Retrieval Agent | Turns a topic into search queries; retrieves and de-duplicates papers | `ANALYSIS` |
| Paper Analysis Agent | Extracts problem, methodology, dataset, results, limitations, future work from retrieved chunks of one paper | `ANALYSIS` |
| Cross-Paper Comparison Agent | Recurring patterns, common limitations, methodological differences, conflicts, underexplored directions — each with paper/chunk evidence | `COMPARISON` |
| Research Gap Detection Agent | Candidate gaps with exact evidence; a gap needs evidence from at least two papers | `GAP` |
| Evidence Validation Agent | Re-reads every cited chunk by exact ID, checks chunk ownership, assesses each paper's contribution and the complete gap jointly | `VALIDATION` |
| Critic Agent | ACCEPT / REVISE / REJECT per gap, origin assessment, evidence-labelled criticisms and revision suggestions | `CRITIC` |
| Research Question / Hypothesis Agent | Questions for critic-accepted gaps; hypotheses only when evidence is sufficient | `QUESTION` |

Every model sees short labels (`P1`, `P1-E2`) instead of real IDs; labels are
mapped back to exact paper/chunk IDs in code, so a model cannot invent or alter a
citation. Each agent owns its own Gemini client, created lazily on first use.

## Workflow

```
Retrieval → Paper Analysis → Cross-Paper Comparison → Gap Detection
  → Evidence Validation → Critic
      ├─ REVISE (within --max-revisions) → apply safe revisions → revalidate revised gaps → Critic
      ├─ all REJECT → stop
      └─ otherwise → Research Questions (accepted gaps) → Hypotheses (only when eligible)
```

* **Stops** are explicit and recorded in `workflow_status` / `stop_reason`:
  too few papers retrieved or analyzed, no candidate gaps, incomplete evidence
  validation, an incomplete Critic review, revision limit reached, all gaps
  rejected, no safe revision applicable, or incomplete question generation.
* **Automatic revision.** For a `REVISE` decision the graph applies only safe
  text/origin/evidence suggestions whose citations exist in the comparison, never
  reduces a gap to fewer than two supporting papers, then revalidates only the
  revised gaps and asks the Critic again. No manual gap editing is needed.
* **Rejected gaps** (complete Critic `REJECT`) are recorded in
  `research_questions.excluded_gaps` and do not block questions for accepted gaps.
* **Provenance.** Each artifact records SHA-256 fingerprints of its inputs.
  Downstream stages and resume paths refuse stale or unfingerprinted inputs.

## Evidence, question and hypothesis rules

* A single paper's limitation is never a gap: gaps (and Critic revisions) need
  evidence from at least two papers.
* Every evidence reference must resolve to the exact stored chunk **and** that
  chunk's stored `paper_id` must match the cited paper; analyses whose evidence
  mixes papers are excluded.
* Per-chunk statuses (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`,
  `NOT_FOUND`, `MISMATCHED`) are preserved and never upgraded.
* A gap is `FULLY_SUPPORTED` only when the joint assessment says so, no unresolved
  contradiction remains, at least one cited chunk is `SUPPORTED`, and no cited
  chunk is `UNSUPPORTED`. Partially supported gaps are kept with caveats.
* Research questions are generated only for gaps with a complete Critic `ACCEPT`,
  complete evidence validation, and at least one supported/partially supported
  chunk. Each question cites exact validated chunks; author-stated contributions
  (`evidence_backed_facts`) are kept separate from proposed questions.
* A **hypothesis** is generated only if the gap and its joint assessment are
  `FULLY_SUPPORTED`, no cited chunk is `UNSUPPORTED`, `SUPPORTED` evidence comes
  from at least two papers, the hypothesis cites only `SUPPORTED` chunks from two
  papers, and it names an independent variable, dependent variable, comparator,
  and measurable outcome. Otherwise it is withheld with
  `hypothesis_status: INSUFFICIENT_EVIDENCE` and a stated limitation.
* Outputs are proposals bounded to the analyzed papers; absence from this sample
  is never presented as global novelty.

## Setup

```powershell
python -m pip install -r requirements.txt       # run the agents
python -m pip install -r requirements-dev.txt   # run the agents and the offline tests
```

Runtime dependencies (`requirements.txt`): `google-genai`, `python-dotenv`,
`requests`, `pymupdf`, `chromadb`, `sentence-transformers`,
`langgraph>=1.0,<2.0`. `requirements-dev.txt` includes these and adds `pytest`.
Embeddings are local (`BAAI/bge-small-en-v1.5`, 384 dimensions); the model is
downloaded from Hugging Face on first use, and `sentence-transformers` installs
PyTorch, which is a large download. Vectors are stored in the per-model
collection `research_papers__bge_small_en_v1_5` (cosine distance); it refuses to
open with a different model, and re-ingesting an unchanged paper is skipped.

### Gemini configuration (six keys)

Copy `.env.example` to `.env` and fill in one key per agent group:

```
GEMINI_API_KEY_ANALYSIS=      # Paper Analysis, Planner, single-LLM baseline
GEMINI_API_KEY_COMPARISON=    # Cross-Paper Comparison
GEMINI_API_KEY_GAP=           # Research Gap Detection
GEMINI_API_KEY_VALIDATION=    # Evidence Validation
GEMINI_API_KEY_CRITIC=        # Critic
GEMINI_API_KEY_QUESTION=      # Research Question / Hypothesis
```

Precedence per group: its six-key variable → the previous four-key variable
(`GEMINI_API_KEY_COMPARISON_GAP` for comparison/gap,
`GEMINI_API_KEY_VALIDATION_CRITIC` for validation/critic) → `GEMINI_API_KEY`.
Error messages name the variables, never their values. `.env` is git-ignored.

## Running the workflow

All commands below make **live** retrieval and/or Gemini calls.

```powershell
# Full LangGraph workflow for a topic
python experiments/run_langgraph.py --topic "your research topic" --max-revisions 2

# Frozen six-paper evaluation corpus with a finalized prompt
python experiments/run_langgraph.py --evaluation-prompt-id PROMPT-001
```

Evaluation-mode runs write a new uniquely named
`data/processed/langgraph_run-PROMPT-XXX-<timestamp>-<token>.json` and never
overwrite an existing artifact. A normal-topic run writes
`data/processed/langgraph_run.json` (or `--output`); an existing file is first
preserved under `data/processed/archive/`.

Stage-by-stage runners (each reads the previous stage's file in `data/processed/`):

```powershell
python experiments/run_multi_paper_comparison.py
python experiments/run_research_gap_detection.py
python experiments/run_evidence_validation.py
python experiments/run_critic.py
python experiments/run_research_questions.py
```

These runners replace their canonical output file atomically, after first
preserving the previous version in `data/processed/archive/`, so an incomplete
re-run (for example after HTTP 429) cannot destroy an earlier complete result.
Each exits non-zero when its output is incomplete.

## Resuming from saved artifacts

```powershell
# Continue a saved run from its Critic result (re-runs only incomplete reviews,
# then revision / question generation as routed)
python experiments/run_langgraph.py --resume-from-critic data\processed\<run>.json

# Resume at the Critic with a replacement evidence-validation artifact
python experiments/run_langgraph.py --resume-from-evidence data\processed\<run>.json `
    --resume-evidence-validation data\processed\<evidence>.json
```

Resume validates every fingerprint, paper-ID set and evidence reference first,
reuses only complete Critic reviews whose three input fingerprints all match, drops stale
downstream results (old Critic output / research questions) instead of carrying
them into the new artifact, and writes a new `...-resumed-...json` artifact. The
new artifact's `run_lineage` records the source artifact name and SHA-256.

Targeted evidence retries (no full re-validation):

```powershell
python experiments/run_evidence_validation.py --retry-joint GAP-003
python experiments/run_evidence_validation.py --input data\processed\<run>.json --retry-evidence GAP-003
```

## Targeted research-question recovery

When question generation for one gap failed (for example HTTP 429), regenerate
only that gap from a saved LangGraph artifact:

```powershell
python experiments/run_research_questions.py `
    --resume-from-langgraph data\processed\<run>.json `
    --resume-research-questions GAP-004
```

Before any model call the runner verifies the artifact's provenance and Critic
fingerprints, that the Critic review is complete, that the target gap has a
complete `ACCEPT`, is evidence-eligible, and is currently recorded as skipped
(not completed or excluded). Only that gap is sent to the model. Previously
generated gaps are carried over unchanged (enforced in code), the source artifact
is never modified, and the merged result is written to a new uniquely named
artifact. The status becomes `COMPLETE` only when every non-excluded gap has
results; otherwise it stays `INCOMPLETE`/`STOPPED` and the command exits 1. Each
attempt — successful or not — appends a `run_lineage` entry (`gap_id`, `outcome`,
`reason`, source artifact SHA-256), and recovery can be chained from a previous
recovery artifact.

## Known limitations

* **API quota.** Gemini free-tier quotas can return HTTP 429. Every agent uses
  one shared classifier (`src/agents/paper_analysis_agent.py`): per-minute rate
  limits (429) and HTTP 500/502/503/504 are retried with bounded exponential
  backoff; daily-quota exhaustion (a 429 whose quota is per day, e.g.
  `GenerateRequestsPerDayPerProjectPerModel`) and other errors are never retried.
  After a daily-quota 429 the Paper Analysis, Evidence Validation, Critic and
  Question agents send no further requests in that run and record the remaining
  items as incomplete/not attempted. This state is run-scoped: it is cleared at
  the start of each run (`PaperAnalysisAgent.begin_run()`, called by the
  pipeline and the LangGraph analysis node; `validate()`; `critique()`;
  `generate_questions()`), never between papers or gaps within a run, so a
  reused agent works again once the quota resets. The Critic and Question agents also honour
  the provider's retry delay (failing closed if it exceeds `max_backoff_seconds`). Code cannot remove a quota limit: wait
  for the quota to reset (or use a key with quota), then use the resume or
  targeted-recovery commands above.
* All judgments are limited to the analyzed papers and retrieved chunks, not
  full texts, and do not establish novelty in the wider literature.
* `gap_origin` (author-stated vs. inferred) is a model assessment reviewed by the
  Critic and the joint evidence assessment; it is not independently verified.
* Saved artifacts written before the stricter `FULLY_SUPPORTED` rule keep their
  recorded gap status; hypothesis eligibility re-checks the per-chunk statuses
  so an `UNSUPPORTED` cited chunk still blocks a hypothesis.

## Offline tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest            # pytest.ini limits collection to tests/
```

The suite uses mocked agents and Gemini clients; `tests/conftest.py` blocks all
outbound network connections, so a test that tries to reach an API fails instead
of consuming quota. `experiments/test_*.py` are live scripts and are never
collected. The suite does not depend on the git-ignored local corpus
(`data/papers/`, `data/processed/`, `chroma_db/`), so it runs from a fresh clone.
Resume and recovery tests copy sanitized saved runs from `tests/fixtures/` into a
temporary directory, and evaluation tests build a temporary corpus with synthetic
PDFs. `tests/fixtures/README.md` lists the source papers quoted in those fixtures
and their verified licenses, including one paper whose license restricts reuse.

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
python experiments/run_evaluation.py --multi-agent-run data\processed\langgraph_run-PROMPT-001-<timestamp>-<token>.json --research-prompt-id PROMPT-001 --prompt path\to\PROMPT-001.txt --citation-index data\evaluation\chunk_owners.json
```

The prompt file must contain exactly the selected manifest prompt text. The
ownership-index command reads the existing indexed collection and rejects
missing or inconsistent paper/chunk metadata; it does not create records. Runs
remain unsplit, and comparison reports preserve `evaluation_split: null`. See
`data/evaluation/annotation_protocol.md` before adding human judgments.

## Future scope

* Human-in-the-loop validation.
