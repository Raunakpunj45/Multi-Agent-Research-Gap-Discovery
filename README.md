# Multi-Agent Research Gap Discovery

**Multi-Agent Framework for Automated Research Gap Discovery and Evidence-Grounded
Hypothesis Generation.**

The system retrieves research papers, extracts paper-level information, compares
papers in a deterministic evidence matrix, proposes candidate research gaps,
validates each gap against the exact source chunks it cites, has a Critic Agent
accept or reject each gap, and generates research questions — plus hypotheses
only when the evidence is sufficient — and a Markdown report. Citations,
provenance, and evidence statuses are preserved end to end. The default pipeline
is sized to run within one Groq free-tier account (8,000 TPM per model). Comparative evaluation against a single-LLM and a conventional-RAG
baseline is one supported objective (see *Offline Evaluation Framework*), not the
whole project.

## Architecture

```
src/
  agents/        the agents, artifact provenance, LLM provider configuration, artifact I/O
  workflow/      LangGraph orchestration (routing and provenance handoffs only)
  retrieval/     OpenAlex, Crossref, Semantic Scholar, arXiv clients + deduplication
  processing/    PDF download/extraction, cleaning, chunking
  embeddings/    local sentence-transformers embeddings (BAAI/bge-small-en-v1.5)
  vectorstore/   ChromaDB store (per-model collection)
  rag/           ingestion and retrieval of paper chunks
  evaluation/    frozen-corpus evaluation, baselines, metrics, citation checks
experiments/     command-line runners (these make LIVE API calls)
tests/           offline pytest suite (network is blocked)
data/            evaluation corpus metadata and the sample run report (PDFs,
                 processed artifacts and Chroma stores are local and git-ignored)
streamlit_app.py Streamlit demo front end over src/workflow/demo.py
```

### Agents, models and key groups

| Agent | Responsibility | Key group | Default Groq model (role) |
|---|---|---|---|
| Planner / Retrieval Agent | Turns a topic into search queries; retrieves and de-duplicates papers | `ANALYSIS` | `openai/gpt-oss-20b` (`PLANNER`) |
| Paper Analysis Agent | Extracts problem, methodology, dataset, results, limitations, future work from retrieved chunks of one paper | `ANALYSIS` | `openai/gpt-oss-20b` (`ANALYSIS`) |
| Evidence Matrix (deterministic, no LLM) | Paper × research-dimension matrix; grounds each analysis statement in the paper's own chunk (exact quote, exact chunk ID) and clusters related statements across papers with local embeddings; keeps only clusters supported by ≥ 2 distinct papers | — | none (`BAAI/bge-small-en-v1.5`, local) |
| Research Gap Detection Agent | At most 3 candidate gaps from the compact matrix, in one bounded request; a gap needs evidence from at least two papers | `GAP` | `openai/gpt-oss-120b` (`GAP`) |
| Evidence Validation Agent | Deterministic checks first (chunk exists, paper ownership, exact matrix quotes, completeness), then one request per gap assessing all its evidence jointly | `VALIDATION` | `openai/gpt-oss-20b` (`VALIDATION`) |
| Critic Agent | One ACCEPT / REJECT request per validated gap, with evidence-grounded reasoning and origin assessment; never rewrites a gap | `CRITIC` | `openai/gpt-oss-120b` (`CRITIC`) |
| Research Question / Hypothesis Agent | Questions for critic-accepted gaps; hypotheses only when evidence is sufficient | `QUESTION` | `openai/gpt-oss-120b` (`QUESTION`) |
| Single-LLM / Conventional-RAG baseline (evaluation only) | One structured generation over the frozen corpus | `ANALYSIS` | `openai/gpt-oss-120b` (`BASELINE`) |

Every model sees short labels (`P1`, `P1-E2`) instead of real IDs; labels are
mapped back to exact paper/chunk IDs in code, so a model cannot invent or alter a
citation. Each agent owns its own LLM client (Groq by default, Gemini optional),
created lazily on first use.

## Workflow

```
Retrieval → Paper Analysis (20B, ≤ 6 papers) → Evidence Matrix (no LLM)
  → Gap Detection (one 120B request, ≤ 3 gaps)
  → Evidence Validation (deterministic checks, then one 20B request per gap)
  → Critic (one 120B ACCEPT/REJECT request per validated gap)
      ├─ all REJECT → stop: insufficient evidence, no hypothesis
      └─ otherwise → Research Questions (accepted gaps) → Hypotheses (only when eligible)
  → report.md (deterministic Markdown, written for every finished run)
```

* **Paper selection.** The planner path applies no lexical filter per
  sub-query: the results of all planner queries are aggregated and
  deduplicated (duplicates fill missing abstracts / PDF URLs), and every
  candidate is then ranked by relevance to the *original* research topic
  (cosine similarity of the topic and each paper's markup-free title + abstract
  with the local `bge-small` model; no API call), ties keeping retrieval order.
  The method is recorded as `relevance_method`. Standalone `MultiSourceRetriever`
  calls still filter lexically by default, with acronym/phrase equivalences
  (`LLM` ↔ `large language model`, `multi-agent` ↔ `multiagent`, `RAG`, ...)
  and one matching word sufficing for title-only records. Analysis walks this ranked list: papers below the relevance
  threshold (default `0.75`, `PAPER_RELEVANCE_THRESHOLD`) are never analyzed,
  a paper that fails download or analysis is replaced by the next relevant one,
  and the rest are not attempted once `--max-papers` analyses succeed. Every
  paper gets an outcome in `paper_selection` (`analyzed`, `failed`,
  `not_attempted_max_papers`, `not_attempted_below_relevance`,
  `not_attempted_quota`); only `failed` papers are `excluded_papers`. Scores live
  in `paper_relevance`, outside `PaperRecord`; the report lists every paper
  excluded by the threshold with its score. If fewer than `--minimum-papers`
  relevant papers can be analyzed, the run stops with an insufficient-evidence
  reason instead of analyzing unrelated papers.
* **Stops** are explicit and recorded in `workflow_status` / `stop_reason`:
  too few papers retrieved or analyzed, no candidate gaps, incomplete evidence
  validation, an incomplete Critic review, revision limit reached, all gaps
  rejected, no safe revision applicable, or incomplete question generation.
* **No silent rewriting.** The default Critic returns only `ACCEPT` or
  `REJECT`, and revision rounds default to 0. The revision path below remains
  only for legacy Critic modes and saved runs: for a `REVISE` decision the graph
  applies only safe text/origin/evidence suggestions whose citations exist in the
  comparison, never reduces a gap to fewer than two supporting papers, then
  revalidates only the revised gaps and asks the Critic again.
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
* Evidence-matrix quotes are exact substrings of the stored chunk text. Before
  any validation request, every quote is re-checked verbatim against the stored
  chunk; if any deterministic check fails for any gap, no validation request is
  sent and the run stops (resumable).
* A contribution counts as author-stated only when the model's quote occurs
  verbatim in that chunk; a gap declared `explicitly_stated` without such a
  quote is flagged as inferred in the joint assessment.
* **Strict acceptance rule** (`src/agents/gap_acceptance.py`). A gap can be
  accepted only when its gap-level validation and joint assessment are both
  `FULLY_SUPPORTED`, **every** cited chunk is `SUPPORTED` with a verified
  verbatim quote, that evidence comes from at least two distinct papers, the
  validated evidence exactly matches the gap's citations, no unresolved
  contradiction remains, and the deterministic checks passed. A
  `PARTIALLY_SUPPORTED` or `UNSUPPORTED` gap therefore can never be accepted.
  The Critic rejects any other gap deterministically (no request); a model,
  legacy-mode or reused cached `ACCEPT` that fails the rule is overridden
  (`REJECT` in verdict mode, `REVISE` in legacy modes). The workflow gate, the
  Research Question Agent and targeted recovery re-check the same rule, so no
  question or hypothesis is produced for such a gap. An `ACCEPT` must also cite
  validated evidence the Critic was shown, otherwise the review is incomplete.
* **Evidence-matrix precision** (`src/agents/evidence_precision.py`, matrix
  `evidence_matrix_v3`). Compound analysis statements are split into clauses, and
  a clause is grounded only in a sentence of its own paper that contains its
  claim (key terms), never in an author/title header, bibliography entry or
  keyword line. Statements that only say something is not mentioned ("does not
  specify a dataset") are never clustered unless the quote itself states the
  absence. Quotes that cite or summarize another work (`[14]`, "et al.",
  author-year, literature-table rows, judged with the quote's own sentence or
  row) are marked secondary, never count as the paper's own finding, and are not
  clustered; validation never marks them author-stated.
* **Contradictions are adjudicated** in every validation mode (at most three per
  gap; more, or an adjudication failure, leaves the gap `VALIDATION_INCOMPLETE`).
  A contradiction is confirmed, and blocks `FULLY_SUPPORTED` and acceptance, only
  when each proposition is stated by its own quote and they directly conflict;
  otherwise it is recorded in `unverified_contradictions` and shown in the report
  as unverified, never as a blocker or as support. Saved artifacts keep their
  recorded contradictions, which still block.
* **One evidence set, one gap.** A candidate gap whose cited evidence set equals,
  is contained in, or contains that of an earlier gap is excluded, and the
  number of gaps is capped by the evidence-matrix clusters shown.
* **Structured-output contract and bounded repair.** Every evidence-validation
  request lists every required JSON key with its type, rendered from the same
  Pydantic model that produces the provider schema (`src/agents/structured_output.py`).
  A schema-invalid response (Groq `json_validate_failed`, unparseable JSON, a
  missing key such as `overgeneralization`, or a blank required string) gets at
  most one corrective re-request naming the exact problem. Nothing is filled in:
  if the output is still invalid, the gap stays `VALIDATION_INCOMPLETE` with the
  precise per-attempt errors in `semantic_failures`, and the run stops.
* Quotes returned by the validation model are located in the stored chunk with
  tolerance only for typographic variants (Unicode dashes, curly quotes,
  whitespace and line-break hyphenation); the stored quote is always the chunk's
  own exact substring.
* Per-chunk statuses (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`,
  `NOT_FOUND`, `MISMATCHED`) are preserved and never upgraded.
* A gap is `FULLY_SUPPORTED` only when the joint assessment says so, no unresolved
  contradiction remains, at least one cited chunk is `SUPPORTED`, and no cited
  chunk is `UNSUPPORTED`. Partially supported gaps are kept with caveats.
* Research questions are generated only for gaps with a complete Critic `ACCEPT`,
  complete evidence validation, and the strict acceptance rule satisfied. Each question cites exact validated chunks; author-stated contributions
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

**Supported Python version: 3.14** (the version the project is developed and
tested on). The dependency lower bounds in `requirements.txt` are the first
releases tested with Python 3.14 wheels; older Python versions are not tested.

Create a virtual environment and install the dependencies from the repository
root.

Windows (PowerShell):

```powershell
py -3.14 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt       # run the agents and the Streamlit demo
python -m pip install -r requirements-dev.txt   # also run the offline tests
```

macOS / Linux:

```bash
python3.14 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt       # run the agents and the Streamlit demo
python -m pip install -r requirements-dev.txt   # also run the offline tests
```

Then copy `.env.example` to `.env` and add your Groq key (see *LLM provider
configuration* below). `.env` is git-ignored and must never be committed.

Runtime dependencies (`requirements.txt`, compatible ranges within one major
version): `groq` (official Groq SDK), `google-genai` (optional Gemini provider,
and the request-config type the agents build), `pydantic` (agent schemas),
`python-dotenv`, `requests`, `pymupdf`, `sentence-transformers`, `chromadb`,
`langgraph` and `streamlit`. `requirements-dev.txt` includes these and adds
`pytest`. Transitive dependencies are resolved by pip.

### Models and resource requirements

* **LLMs are remote.** Every LLM call goes to Groq (default,
  `openai/gpt-oss-20b` and `openai/gpt-oss-120b`) or, optionally, Gemini. No
  LLM runs locally, and a GPU is not needed. A Groq free-tier account is enough
  for the default pipeline; see the token budget notes below.
* **Embeddings are local.** `BAAI/bge-small-en-v1.5` (384 dimensions) runs on
  the CPU through `sentence-transformers`. It is downloaded from Hugging Face
  on first use (about 130 MB, cached under `~/.cache/huggingface/`), so the
  first run needs internet access. Setting `HF_TOKEN` is optional and only
  raises Hugging Face's download rate limits.
* **Disk.** `sentence-transformers` installs PyTorch, so the virtual environment
  needs about 2 GB (PyTorch alone is about 550 MB on Windows, CPU build). Each
  topic run also stores its downloaded PDFs in `data/papers/`, chunk vectors in
  `chroma_db/` and checkpoints in `data/runs/`, which grow by tens of MB per run.
  All of these are git-ignored.
* **Network.** Retrieval calls OpenAlex, Crossref, Semantic Scholar and arXiv.
  `SEMANTIC_SCHOLAR_API_KEY` is optional and raises its rate limit.

Vectors are stored in the per-model collection
`research_papers__bge_small_en_v1_5` (cosine distance); it refuses to open with
a different model, and re-ingesting an unchanged paper is skipped.

### LLM provider configuration

Copy `.env.example` to `.env`. All provider and model settings are read in one
place, `src/agents/llm_provider.py`:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `groq` (default) or `gemini` |
| `GROQ_API_KEY` | Groq key for every agent; optional `GROQ_API_KEY_<GROUP>` overrides it for one key group |
| `GROQ_MODEL_<ROLE>`, then `GROQ_MODEL` | Model override for one role, then for all roles. Roles: `PLANNER`, `ANALYSIS`, `COMPARISON`, `GAP`, `VALIDATION`, `CRITIC`, `QUESTION`, `BASELINE` |
| `GROQ_REASONING_EFFORT` | Optional `low` / `medium` / `high`, sent only to `openai/gpt-oss-*` models |
| `GROQ_TPM_LIMIT[_<MODEL>]` | Tokens-per-minute budget (default `8000`, Groq's base limit for `openai/gpt-oss-*`; `0` disables). `<MODEL>` is the model ID upper-cased with `_`, e.g. `GROQ_TPM_LIMIT_OPENAI_GPT_OSS_120B` |
| `GROQ_MAX_COMPLETION_TOKENS[_<MODEL>]` | Ceiling for the completion cap (default `4096`) |
| `GROQ_CHARS_PER_TOKEN` | Estimator ratio (default `4.0`; Groq counted about 4.4 on this project's prompts) |
| `GEMINI_MODEL_<ROLE>`, then `GEMINI_MODEL` | Model overrides when `LLM_PROVIDER=gemini` |

Groq keys are read from the environment or `.env` only and are never printed,
logged, or included in error messages. Groq rate limits apply per organization,
so several keys from the same organization do not add capacity.

**Groq model selection.** As of 2026-11-17, the Groq production models with
strict structured outputs (constrained decoding, guaranteed schema adherence)
are `openai/gpt-oss-120b` and `openai/gpt-oss-20b`. Other strict-capable models
(`qwen/qwen3.8-27b`, `meta-models/muse-spark-open`) are preview or enterprise
models, and the Llama production models support only JSON object mode.
The per-paper extraction (many calls), query planning and per-gap evidence
validation use `openai/gpt-oss-20b`. Gap detection, the Critic,
question/hypothesis generation and the baselines use `openai/gpt-oss-120b` for
its stronger reasoning. The cross-paper comparison makes no model call. Every agent keeps its
Pydantic schema. The Groq adapter (`src/agents/groq_client.py`) sends it as a
strict `json_schema` response format. Optional scalar fields become nullable,
and those nulls are removed again before validation. The adapter then
validates the reply with the same Pydantic models as before, and falls back to
best-effort mode for models without strict support.

**Groq token budget.** Groq rejects a request larger than the model's
tokens-per-minute (TPM) limit (HTTP 413) and throttles once a minute's tokens
are used (HTTP 429). The Groq adapter handles both before sending:

* **Preflight.** Every request's input, including the JSON schema, is estimated
  conservatively. `max_completion_tokens` is then set so that input plus
  completion fits 90% of the limit. A request that cannot leave at least 1,024
  completion tokens is refused with a non-retryable error and never sent.
* **Pacing.** A shared 60-second token window per model makes consecutive
  requests wait for capacity instead of hitting HTTP 429. Actual usage reported
  by Groq replaces each reservation.
* **Retry-After.** On a per-minute 429, every agent waits exactly Groq's
  `Retry-After`, and fails closed if that exceeds its maximum backoff. A
  completion cut off at the cap raises an explicit error instead of returning
  incomplete JSON.

**Free-tier sizing.** Each task has a fixed output budget
(`src/agents/output_budget.py`, override with `LLM_OUTPUT_BUDGET_<TASK>`), and
the Groq adapter refuses any request whose estimated input plus that budget
exceeds 90% of the TPM limit. The default pipeline keeps every request inside
that budget by construction:

* **Evidence matrix (no LLM).** Each analysis statement is grounded in the most
  similar sentence of the paper's own retrieved chunks (grounding threshold
  `0.55`), and same-dimension statements are clustered across papers
  (average linkage, similarity threshold `0.70`). Thresholds were calibrated on
  `bge-small-en-v1.5` and can be overridden with
  `EVIDENCE_MATRIX_GROUNDING_THRESHOLD` / `EVIDENCE_MATRIX_SIMILARITY_THRESHOLD`.
* **Gap detection** shows the strongest clusters (paper count, then
  limitations/future work first) that fit one request and records how many were
  left out. Only shown evidence has labels, so unseen chunks cannot be cited. A
  gap keeps at most 2 chunks per paper and 6 in total, so its validation request
  fits.
* **Evidence validation** sends each gap's chunks once, with the gap, in one
  `openai/gpt-oss-20b` request (`evidence_gap` budget 3,072). Every label must
  be answered exactly once; unknown, missing or duplicate labels leave the gap
  incomplete. Proposed contradictions are kept only with verbatim quotes and are
  recorded as unresolved, which blocks `FULLY_SUPPORTED`.
* **The Critic** receives the gap, its compact validation records, the joint
  assessment and only the clusters the gap draws on (`critic_verdict` budget
  2,048).

**Legacy modes.** The earlier implementations remain available for
compatibility and for saved runs: `COMPARISON_MODE=llm` (LLM cross-paper
comparison, staged when it does not fit), `EvidenceValidationAgent(mode="per_chunk")`
(one request per cited chunk plus joint and adjudication requests), and
`CriticAgent(review_mode="auto" | "single" | "split")` (REVISE decisions and
split sub-reviews). They are not used on the default path.

#### Optional: Gemini (six keys)

Set `LLM_PROVIDER=gemini` and fill in one key per agent group:

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

## Streamlit demo

With the virtual environment activated (see *Setup*), on any platform:

```bash
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Then open `http://127.0.0.1:8501`. A run makes live Groq and retrieval calls,
so `.env` must contain a Groq key; **Estimate cost** needs no key and sends
nothing.

Enter a topic, optionally click **Estimate cost (no API calls)** (a dry run that
sends nothing), then **Run**. The page (`streamlit_app.py`) is a thin front end
over `src/workflow/demo.py`, which calls the same `run_topic` runner, agents,
Groq configuration, evidence validation, Critic and acceptance rules as the CLI;
there is no second pipeline. Every UI run gets a new directory
`data/runs/ui-<topic>-<UTC time>-<random>/` (checkpoints and `report.md`) and a
new `data/processed/langgraph_run-ui-....json`; it never resumes or overwrites
an earlier run. The page shows step-by-step progress; retrieved, downloaded,
analyzed, failed (download vs. analysis) and skipped (below relevance, target
reached, quota) paper counts; each gap's validation status, Critic decision and
question/hypothesis status; and a Markdown report download. The target defaults
to 10 analyzed papers (at most 15; results per source at most 10): inaccessible
or failed papers are replaced by the next relevant candidate, and the page and
report say so when fewer than the target could be analyzed. A failed run is shown
as such, never replaced by placeholder results.

## Sample successful run and comparison dataset

The repository includes the report of one complete Streamlit run, so the output
can be inspected without API keys:

```
data/runs/ui-multi-agent-llm-systems-for-scientific-l-20261118T212229Z-a19b94/
  report.md       the full Markdown research gap report
  manifest.json   the run settings
```

* **Topic:** *Multi-agent LLM systems for scientific literature review and
  automated research gap discovery*, with up to 10 results per source and a
  target of 10 analyzed papers.
* **Retrieval:** 110 candidates were retrieved and ranked by relevance (42 at or
  above the threshold). 10 papers were downloaded and analyzed. 10 other
  relevant candidates failed to download and were replaced by the next relevant
  candidate.
* **Gaps:** gap detection proposed 3 candidate gaps. Evidence validation rated
  GAP-001 and GAP-003 `FULLY_SUPPORTED` and GAP-002 `PARTIALLY_SUPPORTED`.
* **Critic:** accepted GAP-001 (standardized multi-agent coordination protocols
  and sandbox environments) and rejected GAP-002 and GAP-003 under the
  acceptance rules.
* **Output:** five research questions and one testable hypothesis (with its null
  hypothesis and variables) for GAP-001. No questions or hypotheses were
  generated for the rejected gaps.

Only `report.md` and `manifest.json` are committed. The run's checkpoint
(`state.json`), its `archive/` snapshots and the workflow record in
`data/processed/` stay local and are git-ignored, so a fresh clone can read the
report but cannot resume or re-adjudicate this run.

**Comparison dataset.** The 10 papers analyzed in this run are the project's
comparison dataset (expected locally under `data/comparison_dataset/`). **The
PDFs are not bundled**, because they are third-party copyrighted papers; every
`*.pdf`, `data/papers/` and `data/comparison_dataset/` is git-ignored. To rebuild
the dataset, download the analyzed papers from the OpenAlex IDs listed in the
report's *Retrieved papers* table (rows whose outcome is `analyzed`), for
example `https://openalex.org/W7222723116`. A new run downloads its PDFs into
`data/papers/` automatically. A rerun on the same topic may retrieve different
papers, because the source indexes change over time.

## Running the workflow

All commands below make **live** retrieval and/or LLM API calls (Groq by default).

```powershell
# Estimate every stage first: calls, input tokens, fixed output budgets, TPM fit
# and share of the daily token budget. Sends nothing.
python experiments/run_langgraph.py --topic "your research topic" --dry-run

# Full LangGraph workflow for a topic (checkpointed; re-run to resume)
python experiments/run_langgraph.py --topic "your research topic"

# Frozen six-paper evaluation corpus with a finalized prompt
python experiments/run_langgraph.py --evaluation-prompt-id PROMPT-001
```

A topic run checkpoints every completed step under `data/runs/<topic>-<hash>/`;
running the same topic with the same settings resumes after the last completed
step (`--fresh` starts over, `--run-dir` chooses the directory). Every finished
run, complete or stopped, writes `report.md` there: topic, retrieved papers, the
evidence matrix and its clusters, candidate gaps, validation results, Critic
decisions, research questions and hypotheses (or an explicit
insufficient-evidence statement), numbered citations with exact paper and chunk
IDs, and limitations. Novelty is stated as limited to the retrieved corpus.

The dry run assumes Groq free-tier daily limits of 200,000 tokens and 1,000
requests per model unless `GROQ_TPD_LIMIT_<MODEL>` / `GROQ_RPD_LIMIT_<MODEL>`
are set; check your own limits in the Groq console. For a new six-paper topic
it projects (worst case: 3 gaps of 6 chunks, all accepted) 7 `gpt-oss-120b`
requests, about 37,500 tokens, and 10 `gpt-oss-20b` requests, about 47,000
tokens, with every request within the 7,200-token per-request budget.

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

* **API quota.** Groq (and Gemini) rate limits return HTTP 429. Every agent uses
  one shared classifier (`src/agents/paper_analysis_agent.py`): per-minute rate
  limits (429) and HTTP 500/502/503/504 are retried with bounded exponential
  backoff; daily-quota exhaustion (a 429 whose limit is per day: Groq
  `requests per day (RPD)` / `tokens per day (TPD)`, Gemini
  `GenerateRequestsPerDayPerProjectPerModel`) and other errors are never retried.
  The Groq adapter disables the SDK's own retries, turns network errors into
  retryable `UNAVAILABLE` errors, and removes organization IDs from error text.
* **Evidence-matrix recall.** Clustering favours precision: loosely related
  statements (for example "synthetic graphs only" vs. "simulated records only",
  similarity about 0.62) are not merged, because unrelated limitations score in
  the same range. Such patterns are missed rather than invented. Clusters are
  similarity groupings, not verified agreement; validation and the Critic judge
  their meaning, and conflicts between papers are not detected by the matrix.
* **Request volume on Groq.** Requests are paced to 8,000 TPM per model, so a
  six-paper run takes several minutes of pacing (see `--dry-run`). Legacy
  modes (LLM comparison, per-chunk validation, split Critic) need several
  times the daily quota of the default path.
* **Legacy diagnostic wording.** A few diagnostic strings keep the word
  "Gemini" whichever provider is active (for example `joint Gemini assessment
  failed`, `Gemini chunk assessment failed`). Saved artifacts, the committed
  fixtures and the resume checks match these strings exactly, so they were
  deliberately left unchanged.
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
* Saved artifacts written before the strict acceptance rule may contain an
  `ACCEPT` for a gap with `PARTIALLY_SUPPORTED` chunks or without verified
  quotes. Their records are not rewritten, but resuming or targeted question
  recovery now refuses such gaps (for example GAP-003/GAP-004 of the committed
  fixtures).
* **Relevance threshold.** The default `0.75` was calibrated on one topic (the
  phase-3 live run: on-topic papers 0.78–0.89, LLM-agent papers from other
  domains 0.69–0.76, unrelated papers 0.52–0.69). Other topics may need a
  different `PAPER_RELEVANCE_THRESHOLD`; ranking uses title and abstract only.
* **Topical clusters.** The evidence matrix still groups statements by
  embedding similarity; a cluster can pair a paper's own method with what a
  review paper surveys. Wording now says so, and the strict acceptance rule (not
  the matrix) decides acceptance; a quote-level agreement check is future work.

## Offline tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest            # pytest.ini limits collection to tests/
```

The suite uses mocked agents and LLM clients (Groq tests drive the real `groq`
SDK through an in-memory HTTP transport); `tests/conftest.py` blocks all
outbound network connections, so a test that tries to reach an API fails instead
of consuming quota, and clears provider, model and Groq key variables so a local
`.env` cannot change results. `experiments/test_*.py` are live scripts and are
never collected. The suite does not depend on the git-ignored local corpus
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
IDs, and source chunk ownership checks. These commands make a live LLM call
(configured provider; the run record stores the provider and model);
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
