# Test fixtures

The offline suite never reads the git-ignored local corpus (`data/papers/`,
`data/processed/`, `chroma_db/`), so it runs unchanged from a fresh clone.

## `langgraph_runs/`

Sanitized copies of four saved PROMPT-001 LangGraph run artifacts. They
contain excerpts from five CC BY 4.0 papers (see the license table below) and
synthetic stand-in text for a sixth paper whose license does not permit
redistribution. Resume,
Critic-revision and targeted question-recovery regression tests load them.
Tests copy a fixture into `tmp_path` before using it (the
`langgraph_run_fixture` fixture in `tests/conftest.py`), so the committed files
are never modified.

| Fixture | Scenario |
|---|---|
| `critic_two_revise_decisions.json` | Complete Critic review with two `REVISE` decisions |
| `critic_mixed_incomplete_reviews.json` | Critic review with complete and incomplete (HTTP 429) gap reviews |
| `critic_completed_revisions.json` | Complete Critic review whose suggestions change origin, text and evidence refs |
| `questions_gap004_skipped.json` | Completed run where question generation for `GAP-004` was skipped (HTTP 429) |

The saved runs were changed in two steps.

1. **Path sanitization.** These fields are not covered by any fingerprint:
   * `pdf_paths` values were rewritten from absolute local paths to
     `data/papers/<file name>`.
   * `evaluation_chroma_dir` was rewritten to
     `data/evaluation/chroma_multi_agent`.
2. **Synthetic replacement for OpenAlex W4414948975.** All content from this
   paper was replaced with original synthetic text, and the affected
   provenance fingerprints were recomputed. See
   [Synthetic replacement](#synthetic-replacement-for-openalex-w4414948975) below.

Everything else is unchanged saved model output: the analyses of the other
five papers, gap definitions, decisions, statuses and evidence references.
All fingerprint and lineage checks pass against the recomputed chain. The files
contain no API keys or other credentials.

### Quoted source papers and licenses

The fixtures quote chunk excerpts from five papers of the frozen evaluation
corpus (`data/evaluation/dataset_manifest.jsonl`). The paper analyses,
comparison, gaps, validation and Critic records summarize or quote these
excerpts. Each fixture holds 25 such excerpts, about 32,000 characters in
total, with none longer than about 1,550 characters.

Licenses were verified on 2026-11-17 against the official article pages
listed below. For the three ACL Anthology papers, the license is the
Anthology's stated license for materials published in or after 2016. For the
two Springer Nature articles, it is the "Rights and permissions" statement on
the article page.

| # | Paper | Authors | DOI / stable identifier | Verified license | Official source |
|---|---|---|---|---|---|
| 1 | Large Language Models for Automated Open-domain Scientific Hypotheses Discovery (Findings of ACL 2024) | Zonglin Yang, Xinya Du, Junxian Li, Jie Zheng, Soujanya Poria, Erik Cambria | DOI 10.18653/v1/2024.findings-acl.804; ACL Anthology 2024.findings-acl.804; OpenAlex W4402683817 | Creative Commons Attribution 4.0 International (CC BY 4.0) | https://aclanthology.org/2024.findings-acl.804/ |
| 2 | A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery (EMNLP 2024) | Yu Zhang, Xiusi Chen, Bowen Jin, Sheng Wang, Shuiwang Ji, Wei Wang, Jiawei Han | DOI 10.18653/v1/2024.emnlp-main.498; ACL Anthology 2024.emnlp-main.498; OpenAlex W4404782812 | Creative Commons Attribution 4.0 International (CC BY 4.0) | https://aclanthology.org/2024.emnlp-main.498/ |
| 3 | Constitutive scientific generative agent (CSGA): Leveraging large language models for automated constitutive model discovery (*Machine Learning for Computational Science and Engineering*, 2025) | Marius Tacke, Matthias Busch, Kartik Bali, Kian Abdolazizi, Kevin Linka, Christian Cyron, Roland Aydin | DOI 10.1007/s44379-025-00022-2; OpenAlex W4410862096 | Creative Commons Attribution 4.0 International (CC BY 4.0) | https://link.springer.com/article/10.1007/s44379-025-00022-2 |
| 4 | Exploring the role of large language models in the scientific method: from hypothesis to discovery (*npj Artificial Intelligence*, 2025) | Yanbo Zhang, Sumeer A. Khan, Adnan Mahmud, Huck Yang, Alexander Lavin, Michael Levin, Jeremy Frey, Jared Dunnmon, James Evans, Alan Bundy, Saso Dzeroski, Jesper Tegner, Hector Zenil | DOI 10.1038/s44387-025-00019-5; OpenAlex W4412954363 | Creative Commons Attribution 4.0 International (CC BY 4.0) | https://www.nature.com/articles/s44387-025-00019-5 |
| 5 | From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery (EMNLP 2025) | Tianshi Zheng, Zheye Deng, Hong Ting Tsang, Weiqi Wang, Jiaxin Bai, Zihao Wang, Yangqiu Song | DOI 10.18653/v1/2025.emnlp-main.895; ACL Anthology 2025.emnlp-main.895; OpenAlex W4416036182 | Creative Commons Attribution 4.0 International (CC BY 4.0) | https://aclanthology.org/2025.emnlp-main.895/ |

**Attribution and changes.** Excerpts are reproduced from papers 1–5 above
under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The extraction
pipeline normalized whitespace and split the text into fixed-size chunks, and
the surrounding records contain model-generated summaries of the excerpts. The
original authors did not produce or endorse these summaries.

### Synthetic replacement for OpenAlex W4414948975

The evaluation corpus also includes Yu et al., "Empowering scientific
discovery with explainable small domain-specific and large language models",
*Artificial Intelligence Review* (2025), DOI 10.1007/s10462-025-11365-w,
https://link.springer.com/article/10.1007/s10462-025-11365-w. Its article page
gives the license as Creative Commons Attribution-NonCommercial-NoDerivatives
4.0 International (CC BY-NC-ND 4.0). That license states: "You do not have
permission under this licence to share adapted material derived from this
article or parts of it." **The fixtures therefore contain no text from this
article and no summary of it.** The citation above identifies the record that
was replaced; it does not mean any of the article's content is included.

All content attributed to this paper (labelled `P5` in model output) is
original synthetic text written for these fixtures:

* **Excerpts.** The five chunk excerpts (`_chunk_65`, `_chunk_76`, `_chunk_77`,
  `_chunk_51`, `_chunk_69`) are invented passages about grounding
  language-model assistants in curated reference material. Each begins with
  `[Synthetic fixture text; not from the cited article.]`.
* **Paper analysis.** The analysis of these excerpts is synthetic, and its
  `problem` field starts with a synthetic-content notice.
* **Validation records.** The evidence-validation records for paper-5 chunks
  (`GAP-003` and `GAP-004`) were rewritten in every field that described or
  quoted the chunk. Their quotes are verbatim substrings of the synthetic
  excerpts. Copies of that text in the research-question facts and rationales
  were updated to match.
* **Cross-paper sentences.** Five sentences that used this paper's distinctive
  wording or closely paraphrased it were rewritten:
  * one comparison methodological-difference title and description;
  * one future-work dimension summary;
  * one common-limitation description;
  * one `GAP-003` joint-assessment sentence.
* **Bibliographic fields.** In `papers` and `evaluation_papers`, this paper's
  title, authors and DOI were replaced with a placeholder. Only the OpenAlex
  ID `https://openalex.org/W4414948975` and the chunk IDs remain. The frozen
  evaluation corpus requires these exact IDs, so the identity, ownership and
  evidence-reference checks still exercise the original structure.

Other model-written cross-paper statements that mention `P5` refer to it only
by generic topics: incorrect or hallucinated outputs, the faithfulness of model
explanations, and domain-knowledge injection through prompting, retrieval or
fine-tuning. The synthetic excerpts cover the same topics, so these statements
were kept. They contain no wording from the article.

Every paper ID, chunk ID, status, decision, flag, gap membership and evidence
reference is unchanged. After the text replacement, the provenance chain was
recomputed in pipeline order with the project's own functions:

1. The comparison's `paper_analyses` input fingerprint, rebuilt from the
   updated analyses.
2. The gap-detection `comparison` input fingerprint.
3. The evidence-validation `gap_detection` input fingerprint (unchanged,
   because no gap text changed).
4. The Critic `input_fingerprints` and `critic_input_fingerprints`.

No validation was bypassed. The resume, revision and targeted-recovery tests
run their full fingerprint and integrity checks against these fixtures.

## Evaluation corpus

The evaluation tests use the session fixture `evaluation_corpus` in
`tests/conftest.py`. It copies the tracked
`data/evaluation/dataset_manifest.jsonl` into a temporary root and writes a
small, real, text-extractable PDF (generated with PyMuPDF) at each record's
`relative_path`. It then replaces each record's `sha256` with the hash of that
PDF. All other manifest metadata (paper IDs, prompts, inclusion decisions) is
the real data. Chroma access in these tests is replaced by an in-memory fake.
