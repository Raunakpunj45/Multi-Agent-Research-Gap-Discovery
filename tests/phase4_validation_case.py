"""Saved Phase 4 live-run case (data/runs/multi-agent-llm-systems-for-scientific-literatur-
79deeebd9373, git-ignored): two analyzed papers, GAP-001 validated PARTIALLY_SUPPORTED,
GAP-002 validation failed with Groq HTTP 400 json_validate_failed (missing
'overgeneralization'). Copied verbatim from the saved state.json.
"""

TOPIC = 'Multi-agent LLM systems for scientific literature review and automated research gap discovery'

GAPS = [{'gap_id': 'GAP-001',
  'title': 'Systematic evaluation of deterministic vs. LLM‑based agent components',
  'description': 'Both systems combine deterministic retrieval/ranking modules with LLM‑based '
                 'semantic agents, but no study directly compares the contribution of each '
                 'component to overall performance.',
  'gap_category': 'Methodological Gap',
  'reasoning': 'The methodology cluster shows that P1 uses deterministic modules for '
               'retrieval/ranking and LLM agents for synthesis, while P2 implements a '
               'multi‑agent RAG platform that also mixes deterministic and LLM components. '
               'Neither paper reports an ablation or side‑by‑side comparison of these parts, '
               'indicating a methodological gap.',
  'supporting_paper_ids': ['https://openalex.org/W7233992214',
                           '72cbe3c76d80508388c752e9ae087448714b7efa'],
  'supporting_chunk_ids': ['https://openalex.org/W7233992214_chunk_3',
                           '72cbe3c76d80508388c752e9ae087448714b7efa_chunk_1'],
  'supporting_evidence': [{'paper_id': 'https://openalex.org/W7233992214',
                           'chunk_ids': ['https://openalex.org/W7233992214_chunk_3']},
                          {'paper_id': '72cbe3c76d80508388c752e9ae087448714b7efa',
                           'chunk_ids': ['72cbe3c76d80508388c752e9ae087448714b7efa_chunk_1']}],
  'evidence_summary': 'P1 describes deterministic retrieval/ranking plus LLM‑based semantic '
                      'agents; P2 presents a multi‑agent RAG platform that similarly mixes '
                      'deterministic and LLM modules.',
  'gap_origin': 'inferred_cross_paper',
  'confidence': 'medium',
  'confidence_reason': 'Two papers describe similar architectures without providing '
                       'comparative analysis, suggesting a clear missing evaluation.',
  'research_opportunity': 'Future work could conduct controlled ablations to quantify how '
                          'deterministic and LLM components each affect retrieval quality, '
                          'synthesis accuracy, and overall system efficiency.'},
 {'gap_id': 'GAP-002',
  'title': 'Lack of a standardized, annotated full‑text dataset for multi‑agent literature '
           'review',
  'description': 'Both systems operate on titles/abstracts or uploaded PDFs, but no publicly '
                 'released, fully annotated dataset covering full‑text scientific articles is '
                 'provided for benchmarking.',
  'gap_category': 'Dataset / Evaluation Gap',
  'reasoning': 'The dataset cluster shows P1 works with titles and abstracts from real '
               'corpora, while P2 processes PDF uploads. Neither paper reports a curated '
               'dataset with ground‑truth annotations (e.g., citation links, claim extraction) '
               'for systematic evaluation, highlighting a dataset gap.',
  'supporting_paper_ids': ['https://openalex.org/W7233992214',
                           '72cbe3c76d80508388c752e9ae087448714b7efa'],
  'supporting_chunk_ids': ['https://openalex.org/W7233992214_chunk_29',
                           '72cbe3c76d80508388c752e9ae087448714b7efa_chunk_2'],
  'supporting_evidence': [{'paper_id': 'https://openalex.org/W7233992214',
                           'chunk_ids': ['https://openalex.org/W7233992214_chunk_29']},
                          {'paper_id': '72cbe3c76d80508388c752e9ae087448714b7efa',
                           'chunk_ids': ['72cbe3c76d80508388c752e9ae087448714b7efa_chunk_2']}],
  'evidence_summary': 'P1 uses titles and abstracts from scientific corpora; P2 processes PDF '
                      'uploads from users, with no mention of a shared annotated benchmark.',
  'gap_origin': 'inferred_cross_paper',
  'confidence': 'medium',
  'confidence_reason': 'The two papers mention using real corpora or user‑uploaded PDFs, yet '
                       'no shared benchmark or annotation schema is described, indicating an '
                       'empirical resource gap.',
  'research_opportunity': 'Creating and releasing a benchmark dataset of full‑text scientific '
                          'papers with detailed annotations (e.g., sections, claims, '
                          'citations) would enable consistent evaluation of multi‑agent LLM '
                          'review systems.'}]

PAPER_IDS = ['https://openalex.org/W7233992214', '72cbe3c76d80508388c752e9ae087448714b7efa']

CHUNKS = {'https://openalex.org/W7233992214_chunk_29': ('https://openalex.org/W7233992214',
                                               'and other LLM providers remains untested. 6. '
                                               'Conclusion and Future Work We presented a '
                                               'multi-agent system that suggests and '
                                               'prioritizes candidate research gaps to support '
                                               'human researchers. The contribution is '
                                               'architectural, and it is aimed at the agents '
                                               'community as much as at literature tooling: it '
                                               'is a concrete, measured placement of the '
                                               'boundary between deterministic and LLM-based '
                                               'work inside a multi-agent document- analysis '
                                               'system. Role-specialized Chain-of-Thought '
                                               'agents communicate through a typed shared '
                                               'state under a fixed coordination protocol; '
                                               'every suggestion is grounded in corpus '
                                               'evidence, feasibility is assessed, and an '
                                               'auditable, ranked report abstains when '
                                               'evidence is weak and can be gated on a '
                                               'recorded human decision before its expensive '
                                               'stage. On three real corpora, the system’s '
                                               'gaps are far more often grounded and judged '
                                               'useful, and never trivial, than a '
                                               'single-prompt baseline’s; its main limitation '
                                               'is the mirror image of that '
                                               'selectivity-recall. Future work therefore '
                                               'centers on less conservative candidate '
                                               'generation, temporal verification of whether '
                                               'later work has already closed a candidate, '
                                               'mining full text rather than abstracts alone, '
                                               'a human-expert study to complement the LLM '
                                               'judge, matched'),
 'https://openalex.org/W7233992214_chunk_3': ('https://openalex.org/W7233992214',
                                              'ing prior work, and determining which '
                                              'unresolved questions remain relevant and fea- '
                                              'sible. Gaps may involve underexplored concept '
                                              'combinations, missing experiments, unsupported '
                                              'assertions, or methodological limitations. As '
                                              'scientific literature grows across increasingly '
                                              'diverse venues, performing this analysis '
                                              'manually becomes impracti- cal [Abd-alrazaq et '
                                              'al. 2024, Salem et al. 2025]. Large language '
                                              'models (LLMs) support tasks such as literature '
                                              'review, scien- tific question answering, '
                                              'manuscript assessment, and idea generation '
                                              '[Wang et al. 2024, L´ala et al. 2023, Liang et '
                                              'al. 2024, Si et al. 2024]. However, '
                                              'research-gap identification remains '
                                              'challenging: single-prompt methods often '
                                              'produce generic suggestions, blur dis- tinct '
                                              'evidence types, and lack traceability to source '
                                              'documents, reducing their reliability for '
                                              'research workflows. To address these '
                                              'challenges, we introduce a multi-agent system '
                                              'that detects and prioritizes candidate research '
                                              'gaps. Deterministic modules handle retrieval '
                                              'and ranking, while LLM-based agents perform '
                                              'semantic tasks such as synthesis, '
                                              'classification, and critique. Gaps are detected '
                                              'from rare but meaningful entity associations in '
                                              'a co-occurrence graph and from explicit '
                                              'limitations or future-work statements. Each '
                                              'candidate is linked to supporting evidence, '
                                              'assigned a confidence level, and included in'),
 '72cbe3c76d80508388c752e9ae087448714b7efa_chunk_1': ('72cbe3c76d80508388c752e9ae087448714b7efa',
                                                      'International Journal of Creative and '
                                                      'Open Research in Engineering and '
                                                      'Management ISSN: 3108-1754 (Online) '
                                                      'Volume 02 Issue 10 October-2026 | '
                                                      'Impact Factor: 3.5 © 2026 The '
                                                      'Author(s). Published by IJCOPE Journal. '
                                                      'Website: https://ijcope.org/ 1 '
                                                      'ScholarForge AI: An Open, Multi-Agent '
                                                      'RAG Platform for Automated Research '
                                                      'Paper Analysis Ms. Shravasti '
                                                      'Chaudhari¹,Ms. Akansha Mahore², Mr. '
                                                      'Tanmay Ranotkar³, Mr. Shravan Ningot⁴ '
                                                      '5Dr. Maithili S. Deshmukh 1234 Student, '
                                                      'Information Technology, Prof. Ram Meghe '
                                                      'Institute of Technology and Research, '
                                                      'Bandera, Maharashtra, India 5Assistant '
                                                      'Professor, Information Technology, '
                                                      'Prof. Ram Meghe Institute of Technology '
                                                      'and Research, Badnera, Maharashtra, '
                                                      'India Student Email: '
                                                      'shravastichaudhari57@gmail.com '
                                                      'Abstract— The growing volume of '
                                                      'scientific literature has made '
                                                      'research-paper analysis increasingly '
                                                      'time-consuming for students, '
                                                      'researchers, and academicians. Tasks '
                                                      'such as literature review generation, '
                                                      'methodology understanding, research gap '
                                                      'identification, citation formatting, '
                                                      'and trend discovery often require '
                                                      'substantial manual effort. To address '
                                                      'these challenges, this paper presents '
                                                      'ScholarForge AI, an open, multi-agent '
                                                      'Retrieval-Augmented Generation (RAG) '
                                                      'platform designed to automate '
                                                      'research-paper analysis and transform '
                                                      'scholarly documents into actionable '
                                                      'research intelligence. The proposed '
                                                      'system integrates Large Language Models '
                                                      '(LLMs), LangGraph-based multi-agent '
                                                      'orchestration, ChromaDB vector storage, '
                                                      'a Neo4j knowledge graph,'),
 '72cbe3c76d80508388c752e9ae087448714b7efa_chunk_2': ('72cbe3c76d80508388c752e9ae087448714b7efa',
                                                      'to automate research-paper analysis and '
                                                      'transform scholarly documents into '
                                                      'actionable research intelligence. The '
                                                      'proposed system integrates Large '
                                                      'Language Models (LLMs), LangGraph-based '
                                                      'multi-agent orchestration, ChromaDB '
                                                      'vector storage, a Neo4j knowledge '
                                                      'graph, and Natural Language Processing '
                                                      '(NLP) techniques to support intelligent '
                                                      'document processing. Research papers in '
                                                      'PDF format are processed using PyMuPDF '
                                                      'and EasyOCR, then chunked and embedded '
                                                      'with the BAAI/bge-small-en-v1.5 '
                                                      'sentence transformer model for semantic '
                                                      'retrieval. The platform includes '
                                                      'specialized agents for literature '
                                                      'review generation, methodology '
                                                      'analysis, research gap detection, trend '
                                                      'analysis, research idea generation, '
                                                      'citation formatting, interactive '
                                                      'question answering and difficulty '
                                                      'prediction. In addition, a research '
                                                      'difficulty predictor classifies papers '
                                                      'as Easy, Medium, or Hard, while the '
                                                      'knowledge graph visualizes '
                                                      'relationships among papers, authors, '
                                                      'keywords, methods, and topics. '
                                                      'Experimental results show that '
                                                      'ScholarForge AI improves the efficiency '
                                                      'of literature survey preparation, '
                                                      'supports accurate gap identification, '
                                                      'and provides context-aware research '
                                                      'recommendations. Built entirely on '
                                                      'open-source technologies, the platform '
                                                      'offers a scalable and accessible '
                                                      'solution for academic research support. '
                                                      'Keywords— Artificial intelligence, '
                                                      'research paper analysis '
                                                      ',retrieval-augmented generation (RAG) , '
                                                      'multi-agent systems, literature review, '
                                                      'knowledge graph, ChromaDB, Neo4j. I. '
                                                      'INTRODUCTION The modern academic')}

SEMANTIC_FAILURE = "GAP-002: joint evidence assessment failed: Groq API error (HTTP 400) [json_validate_failed]: Generated JSON does not match the expected schema. Please adjust your prompt. See 'failed_generation' for more details. Error: jsonschema: '' does not validate with /required: missing properties: 'overgeneralization'"

SAVED_VALIDATIONS = {'GAP-001': {'gap_id': 'GAP-001',
             'status': 'PARTIALLY_SUPPORTED',
             'explanation': 'The joint assessment or paper-specific findings support only part '
                            'of the complete gap synthesis.',
             'evidence_validations': [{'paper_id': 'https://openalex.org/W7233992214',
                                       'chunk_id': 'https://openalex.org/W7233992214_chunk_3',
                                       'status': 'SUPPORTED',
                                       'findings': '{"status": "SUPPORTED", '
                                                   '"attributed_contribution": "Both systems '
                                                   'combine deterministic retrieval/ranking '
                                                   'modules with LLM‑based semantic agents", '
                                                   '"claim_evidence_alignment": "Both systems '
                                                   'combine deterministic retrieval/ranking '
                                                   'modules with LLM‑based semantic agents", '
                                                   '"evidence_summary_accuracy": "Accurate: '
                                                   'the evidence summaries correctly reflect '
                                                   'the content of the excerpts.", '
                                                   '"limitations": "", "explanation": "Both '
                                                   'systems combine deterministic '
                                                   'retrieval/ranking modules with LLM‑based '
                                                   'semantic agents", "author_stated": false, '
                                                   '"deterministic_checks": "passed", '
                                                   '"supported_contribution": "deterministic '
                                                   'retrieval/ranking modules and LLM‑based '
                                                   'semantic agents", '
                                                   '"supported_contribution_quote": '
                                                   '"Deterministic modules handle retrieval '
                                                   'and ranking, while LLM-based agents '
                                                   'perform semantic tasks such as synthesis, '
                                                   'classification, and critique."}',
                                       'supported_contribution': 'deterministic '
                                                                 'retrieval/ranking modules '
                                                                 'and LLM‑based semantic '
                                                                 'agents',
                                       'supported_contribution_quote': 'Deterministic modules '
                                                                       'handle retrieval and '
                                                                       'ranking, while '
                                                                       'LLM-based agents '
                                                                       'perform semantic tasks '
                                                                       'such as synthesis, '
                                                                       'classification, and '
                                                                       'critique.',
                                       'mentions_llm': None,
                                       'mentions_faithfulness': None,
                                       'supports_physics_informed_ml': None,
                                       'supports_causal_reasoning': None,
                                       'supports_domain_knowledge_injection': None,
                                       'supports_social_science_survey_scope': None,
                                       'supports_cross_disciplinary_transferability': None,
                                       'supports_peer_review_quality_assessment': None,
                                       'supports_novelty_assessment': None,
                                       'supports_critique_homogenization': None,
                                       'supports_deficiency_identification': None,
                                       'supports_scientific_claim_critique': None,
                                       'claim_evidence_alignment': 'Both systems combine '
                                                                   'deterministic '
                                                                   'retrieval/ranking modules '
                                                                   'with LLM‑based semantic '
                                                                   'agents',
                                       'evidence_summary_accuracy': 'Accurate: the evidence '
                                                                    'summaries correctly '
                                                                    'reflect the content of '
                                                                    'the excerpts.',
                                       'cross_paper_inference': None,
                                       'overgeneralization': None,
                                       'research_opportunity': None,
                                       'gap_origin_assessment': 'INFERRED',
                                       'retrieval_error': None,
                                       'assessment_error': None,
                                       'contradiction_warning': None},
                                      {'paper_id': '72cbe3c76d80508388c752e9ae087448714b7efa',
                                       'chunk_id': '72cbe3c76d80508388c752e9ae087448714b7efa_chunk_1',
                                       'status': 'PARTIALLY_SUPPORTED',
                                       'findings': '{"status": "PARTIALLY_SUPPORTED", '
                                                   '"attributed_contribution": "Both systems '
                                                   'combine deterministic retrieval/ranking '
                                                   'modules with LLM‑based semantic agents", '
                                                   '"claim_evidence_alignment": "Both systems '
                                                   'combine deterministic retrieval/ranking '
                                                   'modules with LLM‑based semantic agents", '
                                                   '"evidence_summary_accuracy": "Accurate: '
                                                   'the evidence summaries correctly reflect '
                                                   'the content of the excerpts.", '
                                                   '"limitations": "", "explanation": "Both '
                                                   'systems combine deterministic '
                                                   'retrieval/ranking modules with LLM‑based '
                                                   'semantic agents", "author_stated": false, '
                                                   '"deterministic_checks": "passed", '
                                                   '"supported_contribution": "LLM-based '
                                                   'semantic agents within a multi‑agent RAG '
                                                   'platform", "supported_contribution_quote": '
                                                   '"ScholarForge AI, an open, multi-agent '
                                                   'Retrieval-Augmented Generation (RAG) '
                                                   'platform designed to automate '
                                                   'research-paper analysis and transform '
                                                   'scholarly documents into actionable '
                                                   'research intelligence."}',
                                       'supported_contribution': 'LLM-based semantic agents '
                                                                 'within a multi‑agent RAG '
                                                                 'platform',
                                       'supported_contribution_quote': 'ScholarForge AI, an '
                                                                       'open, multi-agent '
                                                                       'Retrieval-Augmented '
                                                                       'Generation (RAG) '
                                                                       'platform designed to '
                                                                       'automate '
                                                                       'research-paper '
                                                                       'analysis and transform '
                                                                       'scholarly documents '
                                                                       'into actionable '
                                                                       'research intelligence.',
                                       'mentions_llm': None,
                                       'mentions_faithfulness': None,
                                       'supports_physics_informed_ml': None,
                                       'supports_causal_reasoning': None,
                                       'supports_domain_knowledge_injection': None,
                                       'supports_social_science_survey_scope': None,
                                       'supports_cross_disciplinary_transferability': None,
                                       'supports_peer_review_quality_assessment': None,
                                       'supports_novelty_assessment': None,
                                       'supports_critique_homogenization': None,
                                       'supports_deficiency_identification': None,
                                       'supports_scientific_claim_critique': None,
                                       'claim_evidence_alignment': 'Both systems combine '
                                                                   'deterministic '
                                                                   'retrieval/ranking modules '
                                                                   'with LLM‑based semantic '
                                                                   'agents',
                                       'evidence_summary_accuracy': 'Accurate: the evidence '
                                                                    'summaries correctly '
                                                                    'reflect the content of '
                                                                    'the excerpts.',
                                       'cross_paper_inference': None,
                                       'overgeneralization': None,
                                       'research_opportunity': None,
                                       'gap_origin_assessment': 'INFERRED',
                                       'retrieval_error': None,
                                       'assessment_error': None,
                                       'contradiction_warning': None}],
             'cross_paper_inference': 'The inference that both papers employ deterministic '
                                      'retrieval/ranking modules is only fully supported by '
                                      'P1; P2’s excerpt lacks any mention of deterministic '
                                      'components, so cross‑paper inference is limited to the '
                                      'LLM‑based aspect.',
             'contradiction_warnings': [],
             'joint_assessment': {'status': 'PARTIALLY_SUPPORTED',
                                  'complete_gap_assessment': 'The evidence indicates that P1 '
                                                             'explicitly describes a system '
                                                             'combining deterministic '
                                                             'retrieval/ranking modules with '
                                                             'LLM‑based semantic agents, '
                                                             'directly supporting the claim '
                                                             'that such a combination exists. '
                                                             'P2 describes a multi‑agent RAG '
                                                             'platform that incorporates LLMs '
                                                             'and orchestration mechanisms, '
                                                             'but it does not mention '
                                                             'deterministic retrieval or '
                                                             'ranking components. Thus, the '
                                                             'evidence partially supports the '
                                                             'gap: it confirms the presence of '
                                                             'LLM‑based agents in both systems '
                                                             'but does not confirm '
                                                             'deterministic modules in P2. The '
                                                             'gap remains partially supported, '
                                                             'as the claim about deterministic '
                                                             'components in P2 is not '
                                                             'substantiated by the provided '
                                                             'excerpt.',
                                  'cross_paper_inference': 'The inference that both papers '
                                                           'employ deterministic '
                                                           'retrieval/ranking modules is only '
                                                           'fully supported by P1; P2’s '
                                                           'excerpt lacks any mention of '
                                                           'deterministic components, so '
                                                           'cross‑paper inference is limited '
                                                           'to the LLM‑based aspect.',
                                  'evidence_summary_accuracy': 'Accurate: the evidence '
                                                               'summaries correctly reflect '
                                                               'the content of the excerpts.',
                                  'research_opportunity': 'Controlled ablation studies to '
                                                          'isolate the contributions of '
                                                          'deterministic retrieval/ranking '
                                                          'versus LLM‑based semantic synthesis '
                                                          'remain a viable research direction.',
                                  'gap_origin': 'Inferred across papers. Model assessment: '
                                                'inferred_cross_paper',
                                  'overgeneralization': 'None detected.',
                                  'limitations': 'None identified within the provided '
                                                 'excerpts.',
                                  'counterevidence': 'None detected.',
                                  'source_evidence': [{'paper_id': 'https://openalex.org/W7233992214',
                                                       'chunk_ids': ['https://openalex.org/W7233992214_chunk_3']},
                                                      {'paper_id': '72cbe3c76d80508388c752e9ae087448714b7efa',
                                                       'chunk_ids': ['72cbe3c76d80508388c752e9ae087448714b7efa_chunk_1']}],
                                  'contradictions': []},
             'contradictions': [],
             'joint_assessment_error': None,
             'deterministic_check': {'passed': True,
                                     'checks': ['2 of 2 cited chunk(s) exist and belong to '
                                                'their paper',
                                                'evidence from 2 distinct paper(s)',
                                                'every evidence-matrix quote occurs verbatim '
                                                'in its stored chunk'],
                                     'failures': [],
                                     'origin_check': None}},
 'GAP-002': {'gap_id': 'GAP-002',
             'status': 'VALIDATION_INCOMPLETE',
             'explanation': 'At least one referenced chunk or the joint assessment could not '
                            'be fully validated.',
             'evidence_validations': [{'paper_id': 'https://openalex.org/W7233992214',
                                       'chunk_id': 'https://openalex.org/W7233992214_chunk_29',
                                       'status': None,
                                       'findings': 'Semantic assessment was unavailable.',
                                       'supported_contribution': None,
                                       'supported_contribution_quote': None,
                                       'mentions_llm': None,
                                       'mentions_faithfulness': None,
                                       'supports_physics_informed_ml': None,
                                       'supports_causal_reasoning': None,
                                       'supports_domain_knowledge_injection': None,
                                       'supports_social_science_survey_scope': None,
                                       'supports_cross_disciplinary_transferability': None,
                                       'supports_peer_review_quality_assessment': None,
                                       'supports_novelty_assessment': None,
                                       'supports_critique_homogenization': None,
                                       'supports_deficiency_identification': None,
                                       'supports_scientific_claim_critique': None,
                                       'claim_evidence_alignment': None,
                                       'evidence_summary_accuracy': None,
                                       'cross_paper_inference': None,
                                       'overgeneralization': None,
                                       'research_opportunity': None,
                                       'gap_origin_assessment': None,
                                       'retrieval_error': None,
                                       'assessment_error': 'Groq API error (HTTP 400) '
                                                           '[json_validate_failed]: Generated '
                                                           'JSON does not match the expected '
                                                           'schema. Please adjust your prompt. '
                                                           "See 'failed_generation' for more "
                                                           "details. Error: jsonschema: '' "
                                                           'does not validate with /required: '
                                                           'missing properties: '
                                                           "'overgeneralization'",
                                       'contradiction_warning': None},
                                      {'paper_id': '72cbe3c76d80508388c752e9ae087448714b7efa',
                                       'chunk_id': '72cbe3c76d80508388c752e9ae087448714b7efa_chunk_2',
                                       'status': None,
                                       'findings': 'Semantic assessment was unavailable.',
                                       'supported_contribution': None,
                                       'supported_contribution_quote': None,
                                       'mentions_llm': None,
                                       'mentions_faithfulness': None,
                                       'supports_physics_informed_ml': None,
                                       'supports_causal_reasoning': None,
                                       'supports_domain_knowledge_injection': None,
                                       'supports_social_science_survey_scope': None,
                                       'supports_cross_disciplinary_transferability': None,
                                       'supports_peer_review_quality_assessment': None,
                                       'supports_novelty_assessment': None,
                                       'supports_critique_homogenization': None,
                                       'supports_deficiency_identification': None,
                                       'supports_scientific_claim_critique': None,
                                       'claim_evidence_alignment': None,
                                       'evidence_summary_accuracy': None,
                                       'cross_paper_inference': None,
                                       'overgeneralization': None,
                                       'research_opportunity': None,
                                       'gap_origin_assessment': None,
                                       'retrieval_error': None,
                                       'assessment_error': 'Groq API error (HTTP 400) '
                                                           '[json_validate_failed]: Generated '
                                                           'JSON does not match the expected '
                                                           'schema. Please adjust your prompt. '
                                                           "See 'failed_generation' for more "
                                                           "details. Error: jsonschema: '' "
                                                           'does not validate with /required: '
                                                           'missing properties: '
                                                           "'overgeneralization'",
                                       'contradiction_warning': None}],
             'cross_paper_inference': None,
             'contradiction_warnings': [],
             'joint_assessment': None,
             'contradictions': [],
             'joint_assessment_error': 'Groq API error (HTTP 400) [json_validate_failed]: '
                                       'Generated JSON does not match the expected schema. '
                                       "Please adjust your prompt. See 'failed_generation' for "
                                       "more details. Error: jsonschema: '' does not validate "
                                       'with /required: missing properties: '
                                       "'overgeneralization'",
             'deterministic_check': {'passed': True,
                                     'checks': ['2 of 2 cited chunk(s) exist and belong to '
                                                'their paper',
                                                'evidence from 2 distinct paper(s)',
                                                'every evidence-matrix quote occurs verbatim '
                                                'in its stored chunk'],
                                     'failures': [],
                                     'origin_check': None}}}
