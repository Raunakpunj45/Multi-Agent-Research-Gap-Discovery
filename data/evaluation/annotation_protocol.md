# Evaluation annotation protocol

This file defines the initial human review procedure. The current annotation JSONL contains only a `TEMPLATE_NOT_ANNOTATION` row; it is not gold data and must not be included in metric aggregation.

## Before annotation begins

1. The dataset manifest defines a frozen evaluation/testing corpus of six scientific-discovery papers and three shared prompts. Use the same papers and prompts for Single LLM, Conventional RAG, and Multi-Agent comparisons. The two medical-imaging papers are excluded from the primary evaluation corpus. This evaluation configuration does not change the main project scope or objectives.
2. No train/dev/test splits are assigned. Keep the evaluation paper set and prompt text fixed and identical across the three systems. Bibliographic identity checks may still be completed where the manifest marks a mapping for verification.
3. Two reviewers independently annotate each candidate gap from randomized, system-blinded outputs. Reviewers see the same frozen paper set and relevant source passages for every system.
4. A third reviewer adjudicates disagreements. Preserve both original judgments and the adjudicated result; do not replace them.

Create reference-set JSONL records with `reference_gaps` and no `system_run_id` or `candidate_gap_id`. Store blinded per-output judgments in separate records linked to the matching system run and candidate. Each reference gap needs a stable human-assigned ID, prompt ID, description, source claims/citations where applicable, and explicit scope limited to the frozen corpus.

## Per-candidate rubric

- **Gap relevance:** integer 1–5. `1` = unrelated or not a researchable gap; `3` = partly relevant but vague or weakly motivated; `5` = clearly relevant, specific, and researchable given the prompt and supplied corpus. Scores are judgments, not facts.
- **Evidence support:** `SUPPORTED`, `PARTIAL`, or `UNSUPPORTED`, applied to the candidate's substantive claims. Reviewers must inspect each cited passage and note which claim it supports. Missing evidence is not support.
- **Citation support:** `CORRECT`, `PARTIAL`, or `INCORRECT`, based on whether the cited passage belongs to the cited paper and supports the attached claim. Code separately checks exact chunk existence and paper ownership.
- **Novelty indication:** `INDICATED_WITHIN_SAMPLE`, `NOT_INDICATED_WITHIN_SAMPLE`, or `GLOBAL_NOVELTY_OVERCLAIM`. The first two labels concern only the frozen evaluation corpus. Any claim that novelty is globally established receives the final label. Reviewers must not infer absence from papers that were not in the corpus.
- **Comments:** record concise reasons, counterevidence, and the scope limitation relevant to the judgment.

## Blinding and adjudication

Strip system names and run identifiers from the review view; use randomized candidate IDs that can be joined after annotation. Reviewers should not discuss a candidate before submitting independent labels. Adjudication resolves label disagreements with a written rationale and retains the original labels. Record pseudonymous annotator IDs, annotation date, rubric version, and adjudicator status in the annotation record or run metadata.

## Scoring rules

- Gap relevance: map 1–5 to `(rating - 1) / 4` and report the mean, count, and rating distribution.
- Evidence support: `SUPPORTED=1`, `PARTIAL=0.5`, `UNSUPPORTED=0`; report per-claim mean and unsupported rate.
- Automatic citation correctness: exact chunk presence and paper ownership only. Human citation support is reported separately.
- Novelty indication: score whether the system's indication is consistent with the sample-bounded reviewer label and scope. Never call this global novelty accuracy or proof of novelty.
- Sample-local novelty agreement: exact match between the system's `INDICATED_WITHIN_SAMPLE` / `NOT_INDICATED_WITHIN_SAMPLE` value and the reviewer label. An explicit system abstention (`NOT_ASSESSED`) agrees with `NOT_INDICATED_WITHIN_SAMPLE`; any `GLOBAL_NOVELTY_OVERCLAIM` receives zero.
- Comparison aggregation includes only `COMPLETE` runs. Partial and dry-run records remain visible in the report but are excluded from score means.
- Missing human labels remain missing (`null` scores and denominators); never impute them.
