"""Saved 10-paper live run (data/runs/ui-multi-agent-llm-systems-for-scientific-l-
20261118T203959Z-667a10, git-ignored): all three gaps PARTIALLY_SUPPORTED and
rejected. Copied verbatim from its state.json: the gaps, their validation records,
the saved evidence matrix, and the full saved analyses (with every retrieved
chunk) of the five papers the gaps cite. LABELS are the run's paper labels.
"""

TOPIC = 'Multi-agent LLM systems for scientific literature review and automated research gap discovery'

LABELS = {'https://openalex.org/W4407028609': 'P7',
 'https://openalex.org/W4407359160': 'P3',
 'https://openalex.org/W7126221181': 'P10',
 'https://openalex.org/W7160249182': 'P9',
 'https://openalex.org/W7166808050': 'P8',
 'https://openalex.org/W7171304406': 'P5',
 'https://openalex.org/W7171313681': 'P6',
 'https://openalex.org/W7214432694': 'P4',
 'https://openalex.org/W7222723116': 'P2',
 'https://openalex.org/W7233992214': 'P1'}

GAPS = [{'gap_id': 'GAP-001',
  'title': 'Insufficient mechanisms for coordinated multi‑agent goal decomposition',
  'description': 'Current multi‑agent frameworks lack robust coordination mechanisms that can '
                 'decompose complex scientific objectives into manageable sub‑tasks, limiting '
                 'their effectiveness for large‑scale literature review.',
  'gap_category': 'Integration Gap',
  'reasoning': 'Both P5 and P4 report that effective multi‑agent coordination is missing '
               '(P5‑E3) and that agent coordination remains challenging (P4‑E2). The '
               'convergence of these observations indicates a shared limitation that has not '
               'been addressed, suggesting a gap in integration methods for coordinated goal '
               'decomposition.',
  'supporting_paper_ids': ['https://openalex.org/W7171304406',
                           'https://openalex.org/W7214432694'],
  'supporting_chunk_ids': ['https://openalex.org/W7171304406_chunk_8',
                           'https://openalex.org/W7214432694_chunk_14'],
  'supporting_evidence': [{'paper_id': 'https://openalex.org/W7171304406',
                           'chunk_ids': ['https://openalex.org/W7171304406_chunk_8']},
                          {'paper_id': 'https://openalex.org/W7214432694',
                           'chunk_ids': ['https://openalex.org/W7214432694_chunk_14']}],
  'evidence_summary': 'P5 highlights the absence of effective multi‑agent coordination '
                      'mechanisms for complex scientific goals, while P4 points out that agent '
                      'coordination and evaluation remain challenging.',
  'gap_origin': 'inferred_cross_paper',
  'confidence': 'medium',
  'confidence_reason': 'Two papers explicitly note coordination weaknesses, but no concrete '
                       'solution is presented.',
  'research_opportunity': 'Develop and evaluate novel coordination protocols or hierarchical '
                          'planning architectures that enable multi‑agent systems to break '
                          'down and jointly solve complex scientific literature review tasks.'},
 {'gap_id': 'GAP-002',
  'title': 'Lack of objective, high‑quality evaluation frameworks for multi‑agent LLM outputs',
  'description': 'Existing systems do not provide standardized, objective evaluation methods '
                 'to assess the reasoning and output quality of LLM‑driven agents, hindering '
                 'reliable performance measurement.',
  'gap_category': 'Dataset / Evaluation Gap',
  'reasoning': 'P4 notes difficulty in establishing objective evaluation mechanisms (P4‑E2) '
               'and P8 emphasizes the need for high‑quality evaluation to assess agent '
               'reasoning (P8‑E2). Together they reveal a shared gap in evaluation '
               'methodology.',
  'supporting_paper_ids': ['https://openalex.org/W7214432694',
                           'https://openalex.org/W7166808050'],
  'supporting_chunk_ids': ['https://openalex.org/W7214432694_chunk_14',
                           'https://openalex.org/W7166808050_chunk_38'],
  'supporting_evidence': [{'paper_id': 'https://openalex.org/W7214432694',
                           'chunk_ids': ['https://openalex.org/W7214432694_chunk_14']},
                          {'paper_id': 'https://openalex.org/W7166808050',
                           'chunk_ids': ['https://openalex.org/W7166808050_chunk_38']}],
  'evidence_summary': 'P4 reports challenges in agent coordination and evaluation, while P8 '
                      'identifies the need for objective, high‑quality evaluation mechanisms '
                      'for LLM‑based agents.',
  'gap_origin': 'inferred_cross_paper',
  'confidence': 'medium',
  'confidence_reason': 'Two independent papers mention evaluation challenges, but no concrete '
                       'evaluation framework is proposed.',
  'research_opportunity': 'Design benchmark suites and metric sets that objectively measure '
                          'multi‑agent reasoning fidelity, output correctness, and '
                          'knowledge‑graph construction quality.'},
 {'gap_id': 'GAP-003',
  'title': 'Absence of publicly available benchmark datasets for multi‑agent LLM literature '
           'review',
  'description': 'Current research does not release or rely on standardized datasets for '
                 'training and evaluating multi‑agent literature review pipelines, limiting '
                 'reproducibility and comparative analysis.',
  'gap_category': 'Dataset / Evaluation Gap',
  'reasoning': 'P2 (P2‑E3), P4 (P4‑E3), and P10 (P10‑E1) each report the lack of a defined '
               'experimental dataset, indicating a consistent gap across the field.',
  'supporting_paper_ids': ['https://openalex.org/W7222723116',
                           'https://openalex.org/W7214432694',
                           'https://openalex.org/W7126221181'],
  'supporting_chunk_ids': ['https://openalex.org/W7222723116_chunk_1',
                           'https://openalex.org/W7214432694_chunk_19',
                           'https://openalex.org/W7126221181_chunk_1'],
  'supporting_evidence': [{'paper_id': 'https://openalex.org/W7222723116',
                           'chunk_ids': ['https://openalex.org/W7222723116_chunk_1']},
                          {'paper_id': 'https://openalex.org/W7214432694',
                           'chunk_ids': ['https://openalex.org/W7214432694_chunk_19']},
                          {'paper_id': 'https://openalex.org/W7126221181',
                           'chunk_ids': ['https://openalex.org/W7126221181_chunk_1']}],
  'evidence_summary': 'P2, P4, and P10 all note that no specific dataset or benchmark was '
                      'employed in their studies.',
  'gap_origin': 'inferred_cross_paper',
  'confidence': 'high',
  'confidence_reason': 'Three papers explicitly state that no specific dataset was used or '
                       'introduced.',
  'research_opportunity': 'Create and publish a curated, multi‑domain benchmark dataset with '
                          'annotated retrieval, summarization, and gap‑identification tasks to '
                          'enable systematic evaluation of multi‑agent LLM systems.'}]

SAVED_VALIDATIONS = {'GAP-001': {'gap_id': 'GAP-001',
             'status': 'PARTIALLY_SUPPORTED',
             'explanation': 'The joint assessment or paper-specific findings support only part '
                            'of the complete gap synthesis.',
             'evidence_validations': [{'paper_id': 'https://openalex.org/W7171304406',
                                       'chunk_id': 'https://openalex.org/W7171304406_chunk_8',
                                       'status': 'SUPPORTED',
                                       'findings': '{"status": "SUPPORTED", '
                                                   '"attributed_contribution": "absence of an '
                                                   'effective mechanism of multi-agent '
                                                   'coordination that can break down the '
                                                   'complex scientific goals", '
                                                   '"claim_evidence_alignment": "The chunk '
                                                   'directly states the limitation.", '
                                                   '"evidence_summary_accuracy": "Accurate: '
                                                   'the assessment reflects the content of the '
                                                   'two evidence chunks.", "limitations": "The '
                                                   'chunk does not discuss the effectiveness '
                                                   'of existing coordination mechanisms or '
                                                   'provide empirical evidence of their '
                                                   'failure.", "explanation": "The chunk '
                                                   'directly states the limitation.", '
                                                   '"author_stated": true, '
                                                   '"deterministic_checks": "passed", '
                                                   '"supported_contribution": "absence of an '
                                                   'effective mechanism of multi-agent '
                                                   'coordination that can break down the '
                                                   'complex scientific goals", '
                                                   '"supported_contribution_quote": "One of '
                                                   'the other significant weaknesses is the '
                                                   'absence of an effective mechanism of '
                                                   'multi-agent coordination that can break '
                                                   'down the complex scientific goals"}',
                                       'supported_contribution': 'absence of an effective '
                                                                 'mechanism of multi-agent '
                                                                 'coordination that can break '
                                                                 'down the complex scientific '
                                                                 'goals',
                                       'supported_contribution_quote': 'One of the other '
                                                                       'significant weaknesses '
                                                                       'is the absence of an '
                                                                       'effective mechanism of '
                                                                       'multi-agent '
                                                                       'coordination that can '
                                                                       'break down the complex '
                                                                       'scientific goals',
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
                                       'claim_evidence_alignment': 'The chunk directly states '
                                                                   'the limitation.',
                                       'evidence_summary_accuracy': 'Accurate: the assessment '
                                                                    'reflects the content of '
                                                                    'the two evidence chunks.',
                                       'cross_paper_inference': None,
                                       'overgeneralization': None,
                                       'research_opportunity': None,
                                       'gap_origin_assessment': 'AUTHOR_STATED',
                                       'retrieval_error': None,
                                       'assessment_error': None,
                                       'contradiction_warning': None},
                                      {'paper_id': 'https://openalex.org/W7214432694',
                                       'chunk_id': 'https://openalex.org/W7214432694_chunk_14',
                                       'status': 'UNSUPPORTED',
                                       'findings': '{"status": "UNSUPPORTED", '
                                                   '"attributed_contribution": "absence of an '
                                                   'effective mechanism of multi-agent '
                                                   'coordination that can break down the '
                                                   'complex scientific goals", '
                                                   '"claim_evidence_alignment": "The chunk '
                                                   'does not mention the limitation; it only '
                                                   'describes agent collaboration and '
                                                   'potential benefits.", '
                                                   '"evidence_summary_accuracy": "Accurate: '
                                                   'the assessment reflects the content of the '
                                                   'two evidence chunks.", "limitations": "The '
                                                   'chunk focuses on agent specialization and '
                                                   'collaboration benefits, not on the lack of '
                                                   'coordination mechanisms.", "explanation": '
                                                   '"The chunk does not mention the '
                                                   'limitation; it only describes agent '
                                                   'collaboration and potential benefits.", '
                                                   '"author_stated": false, '
                                                   '"deterministic_checks": "passed", '
                                                   '"supported_contribution": null, '
                                                   '"supported_contribution_quote": null}',
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
                                       'claim_evidence_alignment': 'The chunk does not mention '
                                                                   'the limitation; it only '
                                                                   'describes agent '
                                                                   'collaboration and '
                                                                   'potential benefits.',
                                       'evidence_summary_accuracy': 'Accurate: the assessment '
                                                                    'reflects the content of '
                                                                    'the two evidence chunks.',
                                       'cross_paper_inference': None,
                                       'overgeneralization': None,
                                       'research_opportunity': None,
                                       'gap_origin_assessment': 'INFERRED',
                                       'retrieval_error': None,
                                       'assessment_error': None,
                                       'contradiction_warning': None}],
             'cross_paper_inference': 'No inference can be drawn that both papers agree on the '
                                      'same limitation; only P5-E1 provides the claim.',
             'contradiction_warnings': [],
             'joint_assessment': {'status': 'PARTIALLY_SUPPORTED',
                                  'complete_gap_assessment': 'The gap is partially supported: '
                                                             'P5-E1 explicitly states the lack '
                                                             'of an effective multi‑agent '
                                                             'coordination mechanism, while '
                                                             'P4-E1 does not mention this '
                                                             'limitation, merely describing '
                                                             'agent collaboration. Thus the '
                                                             'evidence does not fully '
                                                             'corroborate the gap across both '
                                                             'papers.',
                                  'cross_paper_inference': 'No inference can be drawn that '
                                                           'both papers agree on the same '
                                                           'limitation; only P5-E1 provides '
                                                           'the claim.',
                                  'evidence_summary_accuracy': 'Accurate: the assessment '
                                                               'reflects the content of the '
                                                               'two evidence chunks.',
                                  'research_opportunity': 'Develop and evaluate novel '
                                                          'coordination protocols or '
                                                          'hierarchical planning architectures '
                                                          'that enable multi‑agent systems to '
                                                          'break down and jointly solve '
                                                          'complex scientific literature '
                                                          'review tasks.',
                                  'gap_origin': 'Author-stated in P5-E1 (verbatim quote '
                                                'verified). Model assessment: '
                                                'inferred_cross_paper',
                                  'overgeneralization': 'No overgeneralization detected; the '
                                                        'claim is specific to multi‑agent '
                                                        'coordination mechanisms.',
                                  'limitations': 'The assessment is limited by the small '
                                                 'number of evidence chunks and the absence of '
                                                 'explicit discussion of coordination '
                                                 'mechanisms in P4-E1.',
                                  'counterevidence': 'None: P4-E1 does not contradict the '
                                                     'claim, it merely does not mention it.',
                                  'source_evidence': [{'paper_id': 'https://openalex.org/W7171304406',
                                                       'chunk_ids': ['https://openalex.org/W7171304406_chunk_8']},
                                                      {'paper_id': 'https://openalex.org/W7214432694',
                                                       'chunk_ids': ['https://openalex.org/W7214432694_chunk_14']}],
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
             'status': 'PARTIALLY_SUPPORTED',
             'explanation': 'The joint assessment or paper-specific findings support only part '
                            'of the complete gap synthesis.',
             'evidence_validations': [{'paper_id': 'https://openalex.org/W7214432694',
                                       'chunk_id': 'https://openalex.org/W7214432694_chunk_14',
                                       'status': 'PARTIALLY_SUPPORTED',
                                       'findings': '{"status": "PARTIALLY_SUPPORTED", '
                                                   '"attributed_contribution": "Intermediate '
                                                   'validation at various stages", '
                                                   '"claim_evidence_alignment": "Supports the '
                                                   'notion of validation but not objective '
                                                   'evaluation frameworks", '
                                                   '"evidence_summary_accuracy": "The evidence '
                                                   'summary accurately reflects the content of '
                                                   'the cited chunks: P4-E1 discusses '
                                                   'intermediate validation, and P8-E1 '
                                                   'highlights the need for objective '
                                                   'evaluation mechanisms.", "limitations": '
                                                   '"Does not address objective evaluation '
                                                   'frameworks", "explanation": "Supports the '
                                                   'notion of validation but not objective '
                                                   'evaluation frameworks", "author_stated": '
                                                   'false, "deterministic_checks": "passed", '
                                                   '"supported_contribution": "Intermediate '
                                                   'validation at various stages", '
                                                   '"supported_contribution_quote": '
                                                   '"intermediate validation at various '
                                                   'stages"}',
                                       'supported_contribution': 'Intermediate validation at '
                                                                 'various stages',
                                       'supported_contribution_quote': 'intermediate '
                                                                       'validation at various '
                                                                       'stages',
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
                                       'claim_evidence_alignment': 'Supports the notion of '
                                                                   'validation but not '
                                                                   'objective evaluation '
                                                                   'frameworks',
                                       'evidence_summary_accuracy': 'The evidence summary '
                                                                    'accurately reflects the '
                                                                    'content of the cited '
                                                                    'chunks: P4-E1 discusses '
                                                                    'intermediate validation, '
                                                                    'and P8-E1 highlights the '
                                                                    'need for objective '
                                                                    'evaluation mechanisms.',
                                       'cross_paper_inference': None,
                                       'overgeneralization': None,
                                       'research_opportunity': None,
                                       'gap_origin_assessment': 'INFERRED',
                                       'retrieval_error': None,
                                       'assessment_error': None,
                                       'contradiction_warning': None},
                                      {'paper_id': 'https://openalex.org/W7166808050',
                                       'chunk_id': 'https://openalex.org/W7166808050_chunk_38',
                                       'status': 'SUPPORTED',
                                       'findings': '{"status": "SUPPORTED", '
                                                   '"attributed_contribution": "Need for '
                                                   'objective, high‑quality evaluation '
                                                   'mechanisms", "claim_evidence_alignment": '
                                                   '"Directly states the gap", '
                                                   '"evidence_summary_accuracy": "The evidence '
                                                   'summary accurately reflects the content of '
                                                   'the cited chunks: P4-E1 discusses '
                                                   'intermediate validation, and P8-E1 '
                                                   'highlights the need for objective '
                                                   'evaluation mechanisms.", "limitations": '
                                                   '"No specific frameworks proposed", '
                                                   '"explanation": "Directly states the gap", '
                                                   '"author_stated": false, '
                                                   '"deterministic_checks": "passed", '
                                                   '"supported_contribution": null, '
                                                   '"supported_contribution_quote": null}',
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
                                       'claim_evidence_alignment': 'Directly states the gap',
                                       'evidence_summary_accuracy': 'The evidence summary '
                                                                    'accurately reflects the '
                                                                    'content of the cited '
                                                                    'chunks: P4-E1 discusses '
                                                                    'intermediate validation, '
                                                                    'and P8-E1 highlights the '
                                                                    'need for objective '
                                                                    'evaluation mechanisms.',
                                       'cross_paper_inference': None,
                                       'overgeneralization': None,
                                       'research_opportunity': None,
                                       'gap_origin_assessment': 'INFERRED',
                                       'retrieval_error': None,
                                       'assessment_error': None,
                                       'contradiction_warning': None}],
             'cross_paper_inference': 'Both papers acknowledge challenges in evaluating '
                                      'multi‑agent LLM outputs, but only P8-E1 directly '
                                      'identifies the lack of objective evaluation as a gap.',
             'contradiction_warnings': [],
             'joint_assessment': {'status': 'PARTIALLY_SUPPORTED',
                                  'complete_gap_assessment': 'The gap is partially supported: '
                                                             'P8-E1 explicitly states the need '
                                                             'for objective, high‑quality '
                                                             'evaluation mechanisms, while '
                                                             'P4-E1 only hints at intermediate '
                                                             'validation without addressing '
                                                             'objective evaluation frameworks.',
                                  'cross_paper_inference': 'Both papers acknowledge challenges '
                                                           'in evaluating multi‑agent LLM '
                                                           'outputs, but only P8-E1 directly '
                                                           'identifies the lack of objective '
                                                           'evaluation as a gap.',
                                  'evidence_summary_accuracy': 'The evidence summary '
                                                               'accurately reflects the '
                                                               'content of the cited chunks: '
                                                               'P4-E1 discusses intermediate '
                                                               'validation, and P8-E1 '
                                                               'highlights the need for '
                                                               'objective evaluation '
                                                               'mechanisms.',
                                  'research_opportunity': 'Develop standardized benchmark '
                                                          'suites and metric sets that '
                                                          'objectively measure multi‑agent '
                                                          'reasoning fidelity, output '
                                                          'correctness, and knowledge‑graph '
                                                          'construction quality.',
                                  'gap_origin': 'Inferred across papers. Model assessment: '
                                                'inferred_cross_paper',
                                  'overgeneralization': 'None identified.',
                                  'limitations': 'P4-E1 does not provide concrete evaluation '
                                                 'methods; P8-E1 does not propose specific '
                                                 'frameworks, leaving the gap unaddressed in '
                                                 'practice.',
                                  'counterevidence': 'None.',
                                  'source_evidence': [{'paper_id': 'https://openalex.org/W7214432694',
                                                       'chunk_ids': ['https://openalex.org/W7214432694_chunk_14']},
                                                      {'paper_id': 'https://openalex.org/W7166808050',
                                                       'chunk_ids': ['https://openalex.org/W7166808050_chunk_38']}],
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
 'GAP-003': {'gap_id': 'GAP-003',
             'status': 'PARTIALLY_SUPPORTED',
             'explanation': 'The joint assessment or paper-specific findings support only part '
                            'of the complete gap synthesis.',
             'evidence_validations': [{'paper_id': 'https://openalex.org/W7222723116',
                                       'chunk_id': 'https://openalex.org/W7222723116_chunk_1',
                                       'status': 'SUPPORTED',
                                       'findings': '{"status": "SUPPORTED", '
                                                   '"attributed_contribution": "The study does '
                                                   'not employ a defined experimental dataset '
                                                   'or benchmark.", '
                                                   '"claim_evidence_alignment": "The chunk '
                                                   'states that the project uses LLM-driven '
                                                   'multi‑agents and semantic searching but '
                                                   'does not mention a specific dataset or '
                                                   'benchmark.", "evidence_summary_accuracy": '
                                                   '"The evidence summary accurately reflects '
                                                   'the content of the cited chunks: each '
                                                   'notes the absence of a defined dataset or '
                                                   'benchmark. No misrepresentation is '
                                                   'present.", "limitations": "The chunk does '
                                                   'not explicitly state that no dataset '
                                                   'exists in the field, only that this study '
                                                   'does not use one.", "explanation": "The '
                                                   'chunk states that the project uses '
                                                   'LLM-driven multi‑agents and semantic '
                                                   'searching but does not mention a specific '
                                                   'dataset or benchmark.", "author_stated": '
                                                   'false, "deterministic_checks": "passed", '
                                                   '"supported_contribution": "The study does '
                                                   'not employ a defined experimental dataset '
                                                   'or benchmark.", '
                                                   '"supported_contribution_quote": "In this '
                                                   'project, an AI-driven Research Assistant '
                                                   'is introduced to support researchers and '
                                                   'students in finding relevant information '
                                                   'related to their field of interest, '
                                                   'identifying research trends, and creating '
                                                   'research reports systematically."}',
                                       'supported_contribution': 'The study does not employ a '
                                                                 'defined experimental dataset '
                                                                 'or benchmark.',
                                       'supported_contribution_quote': 'In this project, an '
                                                                       'AI-driven Research '
                                                                       'Assistant is '
                                                                       'introduced to support '
                                                                       'researchers and '
                                                                       'students in finding '
                                                                       'relevant information '
                                                                       'related to their field '
                                                                       'of interest, '
                                                                       'identifying research '
                                                                       'trends, and creating '
                                                                       'research reports '
                                                                       'systematically.',
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
                                       'claim_evidence_alignment': 'The chunk states that the '
                                                                   'project uses LLM-driven '
                                                                   'multi‑agents and semantic '
                                                                   'searching but does not '
                                                                   'mention a specific dataset '
                                                                   'or benchmark.',
                                       'evidence_summary_accuracy': 'The evidence summary '
                                                                    'accurately reflects the '
                                                                    'content of the cited '
                                                                    'chunks: each notes the '
                                                                    'absence of a defined '
                                                                    'dataset or benchmark. No '
                                                                    'misrepresentation is '
                                                                    'present.',
                                       'cross_paper_inference': None,
                                       'overgeneralization': None,
                                       'research_opportunity': None,
                                       'gap_origin_assessment': 'INFERRED',
                                       'retrieval_error': None,
                                       'assessment_error': None,
                                       'contradiction_warning': None},
                                      {'paper_id': 'https://openalex.org/W7214432694',
                                       'chunk_id': 'https://openalex.org/W7214432694_chunk_19',
                                       'status': 'SUPPORTED',
                                       'findings': '{"status": "SUPPORTED", '
                                                   '"attributed_contribution": "The study does '
                                                   'not employ a defined experimental dataset '
                                                   'or benchmark.", '
                                                   '"claim_evidence_alignment": "The chunk '
                                                   'lists various related works and guidelines '
                                                   'but does not describe a specific dataset '
                                                   'used in the study.", '
                                                   '"evidence_summary_accuracy": "The evidence '
                                                   'summary accurately reflects the content of '
                                                   'the cited chunks: each notes the absence '
                                                   'of a defined dataset or benchmark. No '
                                                   'misrepresentation is present.", '
                                                   '"limitations": "The chunk does not '
                                                   'explicitly mention the absence of a '
                                                   'dataset in the study; it merely lists '
                                                   'related works.", "explanation": "The chunk '
                                                   'lists various related works and guidelines '
                                                   'but does not describe a specific dataset '
                                                   'used in the study.", "author_stated": '
                                                   'false, "deterministic_checks": "passed", '
                                                   '"supported_contribution": "The study does '
                                                   'not employ a defined experimental dataset '
                                                   'or benchmark.", '
                                                   '"supported_contribution_quote": "It is a '
                                                   'reporting guideline rather than an AI '
                                                   'implementation LLM scientific assistance '
                                                   'survey [17] LLMs in research Reviews how '
                                                   'LLMs can support different scientific '
                                                   'activities Helps identify possible '
                                                   'benefits"}',
                                       'supported_contribution': 'The study does not employ a '
                                                                 'defined experimental dataset '
                                                                 'or benchmark.',
                                       'supported_contribution_quote': 'It is a reporting '
                                                                       'guideline rather than '
                                                                       'an AI implementation '
                                                                       'LLM scientific '
                                                                       'assistance survey [17] '
                                                                       'LLMs in research '
                                                                       'Reviews how LLMs can '
                                                                       'support different '
                                                                       'scientific activities '
                                                                       'Helps identify '
                                                                       'possible benefits',
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
                                       'claim_evidence_alignment': 'The chunk lists various '
                                                                   'related works and '
                                                                   'guidelines but does not '
                                                                   'describe a specific '
                                                                   'dataset used in the study.',
                                       'evidence_summary_accuracy': 'The evidence summary '
                                                                    'accurately reflects the '
                                                                    'content of the cited '
                                                                    'chunks: each notes the '
                                                                    'absence of a defined '
                                                                    'dataset or benchmark. No '
                                                                    'misrepresentation is '
                                                                    'present.',
                                       'cross_paper_inference': None,
                                       'overgeneralization': None,
                                       'research_opportunity': None,
                                       'gap_origin_assessment': 'INFERRED',
                                       'retrieval_error': None,
                                       'assessment_error': None,
                                       'contradiction_warning': None},
                                      {'paper_id': 'https://openalex.org/W7126221181',
                                       'chunk_id': 'https://openalex.org/W7126221181_chunk_1',
                                       'status': 'SUPPORTED',
                                       'findings': '{"status": "SUPPORTED", '
                                                   '"attributed_contribution": "The study does '
                                                   'not employ a defined experimental dataset '
                                                   'or benchmark.", '
                                                   '"claim_evidence_alignment": "The chunk '
                                                   'describes the use of LLMs for data '
                                                   'preprocessing and literature review but '
                                                   'does not reference a specific dataset or '
                                                   'benchmark.", "evidence_summary_accuracy": '
                                                   '"The evidence summary accurately reflects '
                                                   'the content of the cited chunks: each '
                                                   'notes the absence of a defined dataset or '
                                                   'benchmark. No misrepresentation is '
                                                   'present.", "limitations": "The chunk does '
                                                   'not explicitly state that no dataset '
                                                   'exists in the field, only that this study '
                                                   'does not use one.", "explanation": "The '
                                                   'chunk describes the use of LLMs for data '
                                                   'preprocessing and literature review but '
                                                   'does not reference a specific dataset or '
                                                   'benchmark.", "author_stated": false, '
                                                   '"deterministic_checks": "passed", '
                                                   '"supported_contribution": "The study does '
                                                   'not employ a defined experimental dataset '
                                                   'or benchmark.", '
                                                   '"supported_contribution_quote": "These '
                                                   'capabilities enhance efficiency, foster '
                                                   'faster discoveries, and lower the barrier '
                                                   'to"}',
                                       'supported_contribution': 'The study does not employ a '
                                                                 'defined experimental dataset '
                                                                 'or benchmark.',
                                       'supported_contribution_quote': 'These capabilities '
                                                                       'enhance efficiency, '
                                                                       'foster faster '
                                                                       'discoveries, and lower '
                                                                       'the barrier to',
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
                                       'claim_evidence_alignment': 'The chunk describes the '
                                                                   'use of LLMs for data '
                                                                   'preprocessing and '
                                                                   'literature review but does '
                                                                   'not reference a specific '
                                                                   'dataset or benchmark.',
                                       'evidence_summary_accuracy': 'The evidence summary '
                                                                    'accurately reflects the '
                                                                    'content of the cited '
                                                                    'chunks: each notes the '
                                                                    'absence of a defined '
                                                                    'dataset or benchmark. No '
                                                                    'misrepresentation is '
                                                                    'present.',
                                       'cross_paper_inference': None,
                                       'overgeneralization': None,
                                       'research_opportunity': None,
                                       'gap_origin_assessment': 'INFERRED',
                                       'retrieval_error': None,
                                       'assessment_error': None,
                                       'contradiction_warning': None}],
             'cross_paper_inference': 'All three papers independently report the lack of a '
                                      'specific dataset or benchmark in their studies, '
                                      'suggesting a consistent observation across the field. '
                                      'However, the inference that this indicates a universal '
                                      'absence of benchmarks is not directly proven.',
             'contradiction_warnings': ["'The field has publicly available benchmark datasets "
                                        "for multi‑agent LLM literature review.' "
                                        '(https://openalex.org/W7222723116/https://openalex.org/W7222723116_chunk_1) '
                                        "conflicts with 'The field lacks publicly available "
                                        'benchmark datasets for multi‑agent LLM literature '
                                        "review.' "
                                        '(https://openalex.org/W7214432694/https://openalex.org/W7214432694_chunk_19): '
                                        'Both propositions are not directly supported by the '
                                        'evidence; the chunks only state that the studies do '
                                        'not use a specific dataset, not that the field '
                                        'universally lacks benchmarks.'],
             'joint_assessment': {'status': 'PARTIALLY_SUPPORTED',
                                  'complete_gap_assessment': 'The gap that multi‑agent LLM '
                                                             'literature review lacks publicly '
                                                             'available benchmark datasets is '
                                                             'partially supported by the '
                                                             'evidence. Each cited chunk '
                                                             'acknowledges the absence of a '
                                                             'defined experimental dataset or '
                                                             'benchmark, but none explicitly '
                                                             'states that no such datasets '
                                                             'exist in the field or that all '
                                                             'studies lack them. Thus the '
                                                             'claim is partially supported.',
                                  'cross_paper_inference': 'All three papers independently '
                                                           'report the lack of a specific '
                                                           'dataset or benchmark in their '
                                                           'studies, suggesting a consistent '
                                                           'observation across the field. '
                                                           'However, the inference that this '
                                                           'indicates a universal absence of '
                                                           'benchmarks is not directly proven.',
                                  'evidence_summary_accuracy': 'The evidence summary '
                                                               'accurately reflects the '
                                                               'content of the cited chunks: '
                                                               'each notes the absence of a '
                                                               'defined dataset or benchmark. '
                                                               'No misrepresentation is '
                                                               'present.',
                                  'research_opportunity': 'Developing and publishing a '
                                                          'curated, multi‑domain benchmark '
                                                          'dataset with annotated retrieval, '
                                                          'summarization, and '
                                                          'gap‑identification tasks would '
                                                          'address the identified gap and '
                                                          'enable systematic evaluation of '
                                                          'multi‑agent LLM systems.',
                                  'gap_origin': 'Inferred across papers. Model assessment: '
                                                'inferred_cross_paper',
                                  'overgeneralization': 'The claim that the field universally '
                                                        'lacks publicly available benchmark '
                                                        'datasets may be an '
                                                        'overgeneralization, as the evidence '
                                                        'only covers three studies.',
                                  'limitations': 'The assessment is limited to the three '
                                                 'provided evidence chunks; other studies may '
                                                 'have benchmarks that are not cited here.',
                                  'counterevidence': 'None of the provided chunks contradicts '
                                                     'the claim; they only state the absence '
                                                     'of a dataset in their own work.',
                                  'source_evidence': [{'paper_id': 'https://openalex.org/W7222723116',
                                                       'chunk_ids': ['https://openalex.org/W7222723116_chunk_1']},
                                                      {'paper_id': 'https://openalex.org/W7214432694',
                                                       'chunk_ids': ['https://openalex.org/W7214432694_chunk_19']},
                                                      {'paper_id': 'https://openalex.org/W7126221181',
                                                       'chunk_ids': ['https://openalex.org/W7126221181_chunk_1']}],
                                  'contradictions': [{'proposition_a': 'The field has publicly '
                                                                       'available benchmark '
                                                                       'datasets for '
                                                                       'multi‑agent LLM '
                                                                       'literature review.',
                                                      'source_a': {'paper_id': 'https://openalex.org/W7222723116',
                                                                   'chunk_ids': ['https://openalex.org/W7222723116_chunk_1']},
                                                      'supporting_quote_a': 'In this project, '
                                                                            'an AI-driven '
                                                                            'Research '
                                                                            'Assistant is '
                                                                            'introduced to '
                                                                            'support '
                                                                            'researchers and '
                                                                            'students in '
                                                                            'finding relevant '
                                                                            'information '
                                                                            'related to their '
                                                                            'field of '
                                                                            'interest, '
                                                                            'identifying '
                                                                            'research trends, '
                                                                            'and creating '
                                                                            'research reports '
                                                                            'systematically.',
                                                      'proposition_b': 'The field lacks '
                                                                       'publicly available '
                                                                       'benchmark datasets for '
                                                                       'multi‑agent LLM '
                                                                       'literature review.',
                                                      'source_b': {'paper_id': 'https://openalex.org/W7214432694',
                                                                   'chunk_ids': ['https://openalex.org/W7214432694_chunk_19']},
                                                      'supporting_quote_b': 'It is a reporting '
                                                                            'guideline rather '
                                                                            'than an AI '
                                                                            'implementation '
                                                                            'LLM scientific '
                                                                            'assistance survey '
                                                                            '[17] LLMs in '
                                                                            'research Reviews '
                                                                            'how LLMs can '
                                                                            'support different '
                                                                            'scientific '
                                                                            'activities Helps '
                                                                            'identify possible '
                                                                            'benefits',
                                                      'explanation': 'Both propositions are '
                                                                     'not directly supported '
                                                                     'by the evidence; the '
                                                                     'chunks only state that '
                                                                     'the studies do not use a '
                                                                     'specific dataset, not '
                                                                     'that the field '
                                                                     'universally lacks '
                                                                     'benchmarks.',
                                                      'unresolved': True}]},
             'contradictions': [{'proposition_a': 'The field has publicly available benchmark '
                                                  'datasets for multi‑agent LLM literature '
                                                  'review.',
                                 'source_a': {'paper_id': 'https://openalex.org/W7222723116',
                                              'chunk_ids': ['https://openalex.org/W7222723116_chunk_1']},
                                 'supporting_quote_a': 'In this project, an AI-driven Research '
                                                       'Assistant is introduced to support '
                                                       'researchers and students in finding '
                                                       'relevant information related to their '
                                                       'field of interest, identifying '
                                                       'research trends, and creating research '
                                                       'reports systematically.',
                                 'proposition_b': 'The field lacks publicly available '
                                                  'benchmark datasets for multi‑agent LLM '
                                                  'literature review.',
                                 'source_b': {'paper_id': 'https://openalex.org/W7214432694',
                                              'chunk_ids': ['https://openalex.org/W7214432694_chunk_19']},
                                 'supporting_quote_b': 'It is a reporting guideline rather '
                                                       'than an AI implementation LLM '
                                                       'scientific assistance survey [17] LLMs '
                                                       'in research Reviews how LLMs can '
                                                       'support different scientific '
                                                       'activities Helps identify possible '
                                                       'benefits',
                                 'explanation': 'Both propositions are not directly supported '
                                                'by the evidence; the chunks only state that '
                                                'the studies do not use a specific dataset, '
                                                'not that the field universally lacks '
                                                'benchmarks.',
                                 'unresolved': True}],
             'joint_assessment_error': None,
             'deterministic_check': {'passed': True,
                                     'checks': ['3 of 3 cited chunk(s) exist and belong to '
                                                'their paper',
                                                'evidence from 3 distinct paper(s)',
                                                'every evidence-matrix quote occurs verbatim '
                                                'in its stored chunk'],
                                     'failures': [],
                                     'origin_check': None}}}

SAVED_MATRIX = {'paper_ids': ['https://openalex.org/W7233992214',
               'https://openalex.org/W7222723116',
               'https://openalex.org/W4407359160',
               'https://openalex.org/W7214432694',
               'https://openalex.org/W7171304406',
               'https://openalex.org/W7171313681',
               'https://openalex.org/W4407028609',
               'https://openalex.org/W7166808050',
               'https://openalex.org/W7160249182',
               'https://openalex.org/W7126221181'],
 'dimensions': ['problem', 'methodology', 'dataset', 'results', 'limitations', 'future_work'],
 'cells': [{'paper_id': 'https://openalex.org/W7233992214',
            'dimension': 'problem',
            'statement': 'The paper addresses the challenge of identifying and prioritizing '
                         'promising research gaps in scientific literature, aiming to support '
                         'researchers during literature reviews by providing grounded, '
                         'feasible, and ranked gap suggestions.',
            'chunk_id': 'https://openalex.org/W7233992214_chunk_1',
            'quote': 'Identifying promising research gaps remains a significant challenge in '
                     'scientific research.',
            'grounding_similarity': 0.8783,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7233992214',
            'dimension': 'methodology',
            'statement': 'A constrained multi‑agent system is employed, comprising five '
                         'role‑specialized agents (Query Planner, Researcher, Writer, '
                         'Reviewer, Prioritizer) coordinated via a deterministic state machine '
                         'and a persisted typed blackboard.',
            'chunk_id': 'https://openalex.org/W7233992214_chunk_7',
            'quote': 'Positioning within agent and MAS research From an agent-theoretic '
                     'perspective, the system is a constrained multi-agent organization '
                     'defined by four elements.',
            'grounding_similarity': 0.8055,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7233992214',
            'dimension': 'methodology',
            'statement': 'Deterministic modules handle resource retrieval, ranking, and '
                         'feasibility checks, while LLM‑based agents perform semantic tasks '
                         'such as evidence synthesis, classification, critique, and ranking.',
            'chunk_id': 'https://openalex.org/W7233992214_chunk_3',
            'quote': 'Deterministic modules handle retrieval and ranking, while LLM-based '
                     'agents perform semantic tasks such as synthesis, classification, and '
                     'critique.',
            'grounding_similarity': 0.9675,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7233992214',
            'dimension': 'methodology',
            'statement': 'The system uses a co‑occurrence graph to detect rare entity '
                         'associations and an LLM classifier to identify explicit limitations '
                         'or future‑work statements, linking each candidate gap to supporting '
                         'evidence and assigning a confidence level.',
            'chunk_id': 'https://openalex.org/W7233992214_chunk_3',
            'quote': 'Gaps are detected from rare but meaningful entity associations in a '
                     'co-occurrence graph and from explicit limitations or future-work '
                     'statements.',
            'grounding_similarity': 0.8635,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7233992214',
            'dimension': 'dataset',
            'statement': 'The system operates on real corpora of scientific papers, using '
                         'titles and abstracts for signal extraction.',
            'chunk_id': 'https://openalex.org/W7233992214_chunk_29',
            'quote': 'The contribution is architectural, and it is aimed at the agents '
                     'community as much as at literature tooling: it is a concrete, measured '
                     'placement of the boundary between deterministic and LLM-based work '
                     'inside a multi-agent document- analysis system.',
            'grounding_similarity': 0.697,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7233992214',
            'dimension': 'dataset',
            'statement': 'Specific datasets are not named, but the evaluation involved three '
                         'real corpora of scientific literature.',
            'chunk_id': 'https://openalex.org/W7233992214_chunk_1',
            'quote': 'Identifying promising research gaps remains a significant challenge in '
                     'scientific research.',
            'grounding_similarity': 0.6958,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7233992214',
            'dimension': 'results',
            'statement': 'On the three real corpora, the multi‑agent system produced gaps that '
                         'were far more often grounded in evidence and judged useful, and '
                         'never trivial, compared to a single‑prompt baseline.',
            'chunk_id': 'https://openalex.org/W7233992214_chunk_29',
            'quote': 'On three real corpora, the system’s gaps are far more often grounded and '
                     'judged useful, and never trivial, than a single-prompt baseline’s; its '
                     'main limitation is the mirror image of that selectivity-recall.',
            'grounding_similarity': 0.8573,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7233992214',
            'dimension': 'results',
            'statement': 'The system’s gaps were also more frequently considered useful by '
                         'evaluators.',
            'chunk_id': 'https://openalex.org/W7233992214_chunk_7',
            'quote': 'Our contribution is the integration of these elements around the '
                     'specific problem of suggesting and ranking candidate gaps.',
            'grounding_similarity': 0.7403,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7233992214',
            'dimension': 'limitations',
            'statement': 'The system’s main limitation is a conservative candidate generation '
                         'that trades recall for selectivity, leading to fewer but '
                         'higher‑quality gaps.',
            'chunk_id': 'https://openalex.org/W7233992214_chunk_29',
            'quote': 'On three real corpora, the system’s gaps are far more often grounded and '
                     'judged useful, and never trivial, than a single-prompt baseline’s; its '
                     'main limitation is the mirror image of that selectivity-recall.',
            'grounding_similarity': 0.7265,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7233992214',
            'dimension': 'limitations',
            'statement': 'It currently operates only on titles and abstracts, not full text, '
                         'and relies on LLMs whose performance across providers remains '
                         'untested.',
            'chunk_id': 'https://openalex.org/W7233992214_chunk_29',
            'quote': 'and other LLM providers remains untested.',
            'grounding_similarity': 0.7751,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7233992214',
            'dimension': 'future_work',
            'statement': 'Future work includes developing less conservative candidate '
                         'generation, temporally verifying whether later work has closed a '
                         'candidate, mining full text rather than abstracts alone, conducting '
                         'a human‑expert study to complement the LLM judge, and matching gaps '
                         'to suitable datasets, benchmarks, or frameworks.',
            'chunk_id': 'https://openalex.org/W7233992214_chunk_29',
            'quote': 'Future work therefore centers on less conservative candidate generation, '
                     'temporal verification of whether later work has already closed a '
                     'candidate, mining full text rather than abstracts alone, a human-expert '
                     'study to complement the LLM judge, matched',
            'grounding_similarity': 0.9532,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7222723116',
            'dimension': 'problem',
            'statement': 'The paper addresses the challenge of efficiently reviewing large '
                         'volumes of scientific literature by proposing an AI‑driven research '
                         'assistant that automates literature analysis, identifies research '
                         'trends, detects research gaps, and generates structured research '
                         'reports.',
            'chunk_id': 'https://openalex.org/W7222723116_chunk_7',
            'quote': 'CONCLUSION AND FUTURE SCOPE This work presents an AI-powered academic '
                     'research assistant capable of automated literature analysis, research '
                     'trend identification, and research gap detection using LLM- based '
                     'workflows.',
            'grounding_similarity': 0.9088,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7222723116',
            'dimension': 'methodology',
            'statement': 'The system is built on a modular architecture using LangChain with '
                         'multiple agents.',
            'chunk_id': 'https://openalex.org/W7222723116_chunk_4',
            'quote': 'AI Agent Pipeline and Task Coordination The main intelligence of the '
                     'system is implemented using LangChain.',
            'grounding_similarity': 0.8256,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7222723116',
            'dimension': 'methodology',
            'statement': 'It performs semantic literature retrieval, LLM‑driven summarization, '
                         'embedding‑based trend analysis, and graph‑based research gap '
                         'detection.',
            'chunk_id': 'https://openalex.org/W7222723116_chunk_2',
            'quote': 'Large Language Models (LLMs), Semantic Retrieval, Research Gap '
                     'Detection, Multi-Agent Systems, Academic Literature Analysis.',
            'grounding_similarity': 0.8409,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7222723116',
            'dimension': 'methodology',
            'statement': 'The user interface is implemented with Streamlit, and outputs are '
                         'saved as PDFs and optionally stored on Amazon S3.',
            'chunk_id': 'https://openalex.org/W7222723116_chunk_1',
            'quote': 'Moreover, the reports are created in PDF form and can be saved with '
                     'cloud storage solutions such as Amazon S3.',
            'grounding_similarity': 0.7936,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7222723116',
            'dimension': 'dataset',
            'statement': 'The paper does not specify a particular dataset or benchmark',
            'chunk_id': 'https://openalex.org/W7222723116_chunk_1',
            'quote': '0 International LLM-Driven Research Assistant for Automated Literature '
                     'Analysis and Research Gap Detection Bijoyini Bagchi1, Kaushik Banerjee1, '
                     'Tunir De1, Trina Chakraborty1, Sayudh Chakraborty1 Department of '
                     'Computer Science and Engineering Institute of Engineering and '
                     'Management, Kolkata, India',
            'grounding_similarity': 0.5896,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7222723116',
            'dimension': 'dataset',
            'statement': 'it relies on semantic retrieval from scholarly databases accessed '
                         'via the LangChain framework.',
            'chunk_id': 'https://openalex.org/W7222723116_chunk_4',
            'quote': 'Multiple LangChain based agents coordinate tasks such as semantic '
                     'literature retrieval, LLM-driven summarization, embedding-based trend '
                     'analysis, and automated research gap detection.',
            'grounding_similarity': 0.7871,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7222723116',
            'dimension': 'results',
            'statement': 'Experimental results show a dramatic reduction in manual effort: '
                         'literature review time drops from 4.5 h to 6.2 min, report '
                         'generation time from 45 min to 18 s, and overall workflow time from '
                         '5 h 15 min to 9.8 min.',
            'chunk_id': 'https://openalex.org/W7222723116_chunk_7',
            'quote': '3 s Report Generation Time 45 min 18 s End-to-End Workflow Time 5 hours '
                     '15 min 9.',
            'grounding_similarity': 0.8079,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7222723116',
            'dimension': 'results',
            'statement': 'The system produces automated research reports, trend summaries, and '
                         'gap detection insights.',
            'chunk_id': 'https://openalex.org/W7222723116_chunk_3',
            'quote': 'The suggested system offers automatic analysis of literature, helps find '
                     'research trends, identifies gaps in research, and creates reports in PDF '
                     'format using a flexible AI-powered process.',
            'grounding_similarity': 0.8398,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7222723116',
            'dimension': 'limitations',
            'statement': 'however, the authors note that existing systems lack integrated '
                         'support for trend analysis, gap detection, and cloud storage, '
                         'implying that their system may still face challenges in retrieval '
                         'quality and coverage.',
            'chunk_id': 'https://openalex.org/W7222723116_chunk_3',
            'quote': 'However, most lack integrated support for research trend analysis, '
                     'research gap detection, automated report generation, and cloud- based '
                     'storage.',
            'grounding_similarity': 0.8179,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7222723116',
            'dimension': 'future_work',
            'statement': 'Future work will focus on integrating domain‑specific LLMs, '
                         'GraphRAG, and additional scholarly databases to further improve '
                         'retrieval quality and research insight generation.',
            'chunk_id': 'https://openalex.org/W7222723116_chunk_7',
            'quote': 'Future work will focus on integrating domain-specific LLMs, GraphRAG, '
                     'and additional scholarly databases to further improve retrieval quality '
                     'and research',
            'grounding_similarity': 0.9623,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407359160',
            'dimension': 'problem',
            'statement': 'The paper addresses the challenge of maintaining comprehensive and '
                         'up-to-date knowledge graphs (KGs) in scientific domains, where '
                         'manual curation cannot scale with the rapid growth of literature.',
            'chunk_id': 'https://openalex.org/W4407359160_chunk_1',
            'quote': 'cn Abstract Maintaining comprehensive and up-to-date knowledge graphs '
                     '(KGs) is critical for modern AI systems, but manual curation struggles '
                     'to scale with the rapid growth of scientific literature.',
            'grounding_similarity': 0.9214,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407359160',
            'dimension': 'problem',
            'statement': 'It proposes an automated KG enrichment framework that can parse '
                         'unstructured scientific text, discover new entities and relations, '
                         'align them with existing schema, and resolve conflicts.',
            'chunk_id': 'https://openalex.org/W4407359160_chunk_3',
            'quote': 'Moreover, extracting and integrating knowledge into existing KGs '
                     'requires robust mechanisms for schema alignment, consistency, and '
                     'conflict resolution [5].',
            'grounding_similarity': 0.7804,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407359160',
            'dimension': 'methodology',
            'statement': 'KARMA is a hierarchical multi‑agent system that decomposes the KG '
                         'enrichment pipeline into nine specialized LLM‑driven agents: '
                         'ingestion, summarization, entity recognition, relationship '
                         'extraction, schema alignment, conflict resolution, evaluation, and '
                         'storage.',
            'chunk_id': 'https://openalex.org/W4407359160_chunk_8',
            'quote': '3 Methodology In this section, we introduce KARMA, a hierarchical '
                     'multi-agent system (see Figure 2) that leverages specialized LLMs to '
                     'perform end-to-end KG enrichment.',
            'grounding_similarity': 0.9204,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407359160',
            'dimension': 'methodology',
            'statement': 'Agents communicate via a blackboard architecture, enabling '
                         'cross‑agent verification and iterative refinement.',
            'chunk_id': 'https://openalex.org/W4407359160_chunk_4',
            'quote': 'First, the multi-agent architecture enables cross-agent verification, '
                     'enhancing the reliability of extracted knowledge.',
            'grounding_similarity': 0.8393,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407359160',
            'dimension': 'methodology',
            'statement': 'Domain‑adaptive prompting and LLM‑based debate mechanisms are used '
                         'to mitigate hallucination and ensure schema consistency.',
            'chunk_id': 'https://openalex.org/W4407359160_chunk_8',
            'quote': 'prompting strategies that enable more accurate extraction across diverse '
                     'scientific fields, (3) LLM-based verification mechanisms that mitigate '
                     'issues such as hallucination and schema inconsistency.',
            'grounding_similarity': 0.8383,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407359160',
            'dimension': 'dataset',
            'statement': 'Experiments were conducted on 1,200 PubMed articles drawn from three '
                         'distinct scientific domains.',
            'chunk_id': 'https://openalex.org/W4407359160_chunk_1',
            'quote': 'Experiments on 1,200 PubMed articles from three different domains '
                     'demonstrate the effectiveness of KARMA in knowledge graph enrichment, '
                     'with the identification of up to 38,230 new entities while achieving 83.',
            'grounding_similarity': 0.7831,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407359160',
            'dimension': 'dataset',
            'statement': 'The dataset was used to evaluate the framework’s ability to identify '
                         'new entities and relations and to integrate them into existing KGs.',
            'chunk_id': 'https://openalex.org/W4407359160_chunk_4',
            'quote': 'Through proof-of-concept experiments on datasets from three distinct '
                     'domains,',
            'grounding_similarity': 0.7005,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407359160',
            'dimension': 'results',
            'statement': 'KARMA identified up to 38,230 new entities across the domains, '
                         'achieved 83.1 % LLM‑verified correctness, and reduced conflict edges '
                         'by 18.6 % compared to baseline methods.',
            'chunk_id': 'https://openalex.org/W4407359160_chunk_1',
            'quote': 'Experiments on 1,200 PubMed articles from three different domains '
                     'demonstrate the effectiveness of KARMA in knowledge graph enrichment, '
                     'with the identification of up to 38,230 new entities while achieving 83.',
            'grounding_similarity': 0.7419,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407359160',
            'dimension': 'limitations',
            'statement': 'These are acknowledged as areas that the proposed framework seeks to '
                         'address.',
            'chunk_id': 'https://openalex.org/W4407359160_chunk_4',
            'quote': 'Our framework offers three key innovations.',
            'grounding_similarity': 0.7249,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7214432694',
            'dimension': 'problem',
            'statement': 'The paper addresses the challenge of automating and improving the '
                         'literature review process by proposing an AI‑driven system, IRIS, '
                         'that integrates paper retrieval, scientific information extraction, '
                         'summarization, citation‑aware synthesis, knowledge‑graph '
                         'construction, research‑gap detection, and structured report '
                         'generation.',
            'chunk_id': 'https://openalex.org/W7214432694_chunk_1',
            'quote': 'This paper reviews research related to IRIS (Intelligent Research & '
                     'Information System), a proposed AI-based research assistant which aims '
                     'to combine paper retrieval, scientific information extraction, '
                     'summarization, citation-aware synthesis, knowledge-graph building, '
                     'research-gap detection and',
            'grounding_similarity': 0.898,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7214432694',
            'dimension': 'methodology',
            'statement': 'IRIS employs a multi‑agent workflow where domain‑specialized '
                         'autonomous agents collaborate on retrieval, analysis, summarization, '
                         'gap detection, and report generation.',
            'chunk_id': 'https://openalex.org/W7214432694_chunk_14',
            'quote': 'Multiple agents collaborate on retrieval, analysis, summarization, gap '
                     'detection, and report generation [1].',
            'grounding_similarity': 0.8669,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7214432694',
            'dimension': 'methodology',
            'statement': 'The system leverages retrieval‑augmented generation (RAG), '
                         'long‑context processing, step‑by‑step reasoning, and knowledge‑graph '
                         'construction techniques to support these tasks.',
            'chunk_id': 'https://openalex.org/W7214432694_chunk_1',
            'quote': 'Recent progress in artificial intelligence, natural language processing, '
                     'large language models, retrieval-augmented generation, knowledge graphs '
                     'and autonomous agent workflows creates a chance to support this process.',
            'grounding_similarity': 0.7538,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7214432694',
            'dimension': 'dataset',
            'statement': 'The paper reviews eighteen representative studies across five '
                         'technical areas (AI‑assisted systematic literature review, '
                         'scientific NLP, semantic search, RAG, agent reasoning, and knowledge '
                         'graphs).',
            'chunk_id': 'https://openalex.org/W7214432694_chunk_1',
            'quote': 'We reviewed eighteen representative papers for five technical areas: AI- '
                     'assisted systematic literature review',
            'grounding_similarity': 0.9274,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7214432694',
            'dimension': 'dataset',
            'statement': 'No specific experimental dataset is introduced',
            'chunk_id': 'https://openalex.org/W7214432694_chunk_19',
            'quote': 'AI implementation LLM scientific assistance survey [17] LLMs in research '
                     'Reviews how LLMs can support different scientific activities Helps '
                     'identify possible benefits',
            'grounding_similarity': 0.5945,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7214432694',
            'dimension': 'dataset',
            'statement': 'the work is a comprehensive review rather than an empirical study.',
            'chunk_id': 'https://openalex.org/W7214432694_chunk_2',
            'quote': 'INTRODUCTION A literature review combines the work that has been done on '
                     'a topic to define the state of the knowledge in a research area.',
            'grounding_similarity': 0.6697,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7214432694',
            'dimension': 'results',
            'statement': 'The review identifies that existing approaches excel at individual '
                         'sub‑problems but struggle with integration, provenance detection, '
                         'long‑context handling, explainability, and systematic gap detection.',
            'chunk_id': 'https://openalex.org/W7214432694_chunk_2',
            'quote': 'We observed that existing approaches typically handle individual '
                     'subproblems well but struggle with integration, provenance detection, '
                     'long-context processing, explainability, systematic gap detection, etc.',
            'grounding_similarity': 0.9029,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7214432694',
            'dimension': 'results',
            'statement': 'IRIS is proposed as a human‑in‑the‑loop system that aims to overcome '
                         'these limitations by combining the strengths of the surveyed '
                         'methods.',
            'chunk_id': 'https://openalex.org/W7214432694_chunk_2',
            'quote': 'With these observations, we propose a conceptual IRIS framework.',
            'grounding_similarity': 0.8162,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7214432694',
            'dimension': 'limitations',
            'statement': 'The evidence indicates that agent coordination and evaluation remain '
                         'challenging, automatic knowledge‑graph construction can introduce '
                         'incorrect or duplicate relationships, and long‑context processing '
                         'still limits information utilization.',
            'chunk_id': 'https://openalex.org/W7214432694_chunk_14',
            'quote': 'Looking at this, whether agent coordination increases evidence coverage '
                     'and decreases unsubstantiated claims.',
            'grounding_similarity': 0.7512,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7214432694',
            'dimension': 'limitations',
            'statement': 'The paper does not provide empirical validation of IRIS, so its '
                         'effectiveness is theoretical.',
            'chunk_id': 'https://openalex.org/W7214432694_chunk_2',
            'quote': 'we summarize some evidence as to why IRIS is better considered as a '
                     'human-in-the- loop system rather than a fully automated system of '
                     'retrieving literature reviews, consistent with common understandings in '
                     'research.',
            'grounding_similarity': 0.6983,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7214432694',
            'dimension': 'future_work',
            'statement': 'Future directions include developing and evaluating the multi‑agent '
                         'IRIS architecture, improving agent coordination and evaluation, '
                         'refining automatic knowledge‑graph extraction to reduce errors, and '
                         'integrating systematic review guidelines (e.g., PRISMA 2020) to '
                         'enhance transparency and traceability.',
            'chunk_id': 'https://openalex.org/W7214432694_chunk_19',
            'quote': 'Automatic KG Reviews approaches for acquiring and refining knowledge- '
                     'graph information Provides guidance for building the IRIS research graph '
                     'Automatic extraction may introduce incorrect or duplicate relationships '
                     'PRISMA 2020 [16] Systematic review reporting Provides guidelines for '
                     'making the',
            'grounding_similarity': 0.7769,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171304406',
            'dimension': 'problem',
            'statement': 'The paper addresses the challenge of automating the entire '
                         'scientific research cycle by integrating multiple AI technologies to '
                         'reduce manual effort and improve efficiency in data‑intensive, '
                         'computationally complex research environments.',
            'chunk_id': 'https://openalex.org/W7171304406_chunk_1',
            'quote': 'This paper proposes an Agentic AI Framework for Autonomous Scientific '
                     'Research Assistance that integrates Large Language Models (LLMs), '
                     'Retrieval-Augmented Generation (RAG), and collaborative multi- agent '
                     'systems to automate key research activities.',
            'grounding_similarity': 0.8458,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171304406',
            'dimension': 'methodology',
            'statement': 'An Agentic AI Framework that combines Large Language Models (LLMs), '
                         'Retrieval‑Augmented Generation (RAG), collaborative multi‑agent '
                         'systems, semantic knowledge graphs, reinforcement learning, '
                         'explainable AI, and evidence‑based decision‑making to autonomously '
                         'manage literature review, knowledge extraction, hypothesis '
                         'generation, experiment planning, statistical analysis, citation '
                         'verification, manuscript preparation, and research validation.',
            'chunk_id': 'https://openalex.org/W7171304406_chunk_1',
            'quote': 'This paper proposes an Agentic AI Framework for Autonomous Scientific '
                     'Research Assistance that integrates Large Language Models (LLMs), '
                     'Retrieval-Augmented Generation (RAG), and collaborative multi- agent '
                     'systems to automate key research activities.',
            'grounding_similarity': 0.9217,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171304406',
            'dimension': 'dataset',
            'statement': 'The evaluation used a collection of 4 publications, conference '
                         'proceedings, patents, technical reports, experimental repositories, '
                         'simulation datasets, institutional databases, and open‑access '
                         'research platforms to test literature retrieval, hypothesis '
                         'generation, citation verification, and other metrics.',
            'chunk_id': 'https://openalex.org/W7171304406_chunk_36',
            'quote': 'Experimental evaluation shows that the proposed framework outperforms '
                     'conventional AI-assisted research systems in literature retrieval '
                     'accuracy, workflow automation, quality of hypothesis generation, '
                     'citation verification, explainability, manuscript preparation, and '
                     'research productivity.',
            'grounding_similarity': 0.6792,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171304406',
            'dimension': 'results',
            'statement': 'Experimental results show the framework consistently achieves over '
                         '90% accuracy on most evaluation metrics, including literature '
                         'retrieval accuracy, hypothesis quality, citation verification, '
                         'workflow automation, knowledge discovery, manuscript preparation, '
                         'explainability, and overall research productivity.',
            'chunk_id': 'https://openalex.org/W7171304406_chunk_33',
            'quote': 'The results show that the framework constantly produces values above 90% '
                     'for most of the evaluation metrics, including literature retrieval '
                     'accuracy, quality of hypothesis generation, citation verification, '
                     'workflow automation, knowledge discovery, manuscript preparation, '
                     'explainability, and',
            'grounding_similarity': 0.9051,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171304406',
            'dimension': 'results',
            'statement': 'The centralized knowledge graph and RAG reduce hallucinations and '
                         'improve citation reliability, while multi‑agent collaboration '
                         'enhances computational efficiency through parallel execution and '
                         'adaptive workload balancing.',
            'chunk_id': 'https://openalex.org/W7171304406_chunk_36',
            'quote': 'Besides, the use of collaborative multi-agent technology has increased '
                     'computational efficiency via intelligent task scheduling, parallel '
                     'execution, adaptive learning, and constant improvement of research '
                     'results.',
            'grounding_similarity': 0.8503,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171304406',
            'dimension': 'limitations',
            'statement': 'The evidence notes several limitations: difficulty integrating '
                         'heterogeneous scientific knowledge due to varying metadata '
                         'standards, terminologies, document structures, and citation formats',
            'chunk_id': 'https://openalex.org/W7171304406_chunk_8',
            'quote': 'The challenge of integrating with heterogeneous scientific knowledge '
                     'from scholarly ISSN: 3071-5636 PP: 01-15.',
            'grounding_similarity': 0.8636,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171304406',
            'dimension': 'limitations',
            'statement': 'challenges in maintaining long‑term context across extended research '
                         'initiatives leading to disjointed reasoning',
            'chunk_id': 'https://openalex.org/W7171304406_chunk_8',
            'quote': 'Even the current AI- powered research assistance tools are challenged in '
                     'maintaining long-term context over longer research initiatives, '
                     'resulting in disjointed reasoning, varied recommendations, and not '
                     'sustaining continuity across the entire research process.',
            'grounding_similarity': 0.7555,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171304406',
            'dimension': 'limitations',
            'statement': 'and the current lack of effective multi‑agent coordination '
                         'mechanisms for complex scientific goals.',
            'chunk_id': 'https://openalex.org/W7171304406_chunk_8',
            'quote': 'One of the other significant weaknesses is the absence of an effective '
                     'mechanism of multi-agent coordination that can break down the complex '
                     'scientific goals',
            'grounding_similarity': 0.9115,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171304406',
            'dimension': 'future_work',
            'statement': 'The paper identifies the need to develop scalable, reliable, and '
                         'transparent Agentic AI frameworks that include common coordination '
                         'strategies, adaptive knowledge management, and improved multi‑agent '
                         'orchestration to overcome the highlighted integration and '
                         'context‑maintenance challenges.',
            'chunk_id': 'https://openalex.org/W7171304406_chunk_36',
            'quote': 'This paper presents an Agentic AI Framework for Autonomous Scientific '
                     'Research Assistance that integrates autonomous planning, collaborative '
                     'multi-agent intelligence, Retrieval-Augmented Generation (RAG), semantic '
                     'knowledge graphs, reinforcement learning, explainable artificial '
                     'intelligence, and',
            'grounding_similarity': 0.8335,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171313681',
            'dimension': 'problem',
            'statement': 'The paper addresses the challenge of efficiently discovering, '
                         'summarizing, and analyzing scientific literature in the context of '
                         'rapidly expanding digital repositories, aiming to enable intelligent '
                         'knowledge discovery, semantic search, and automated research '
                         'workflows using Large Language Models (LLMs).',
            'chunk_id': 'https://openalex.org/W7171313681_chunk_2',
            'quote': 'This study presents a conceptual framework that combines semantic '
                     'retrieval, intelligent reasoning, automated literature analysis, and '
                     'workflow orchestration, demonstrating how LLM-powered systems can '
                     'transform scientific research into scalable, accurate, ethical, and '
                     'collaborative knowledge',
            'grounding_similarity': 0.8818,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171313681',
            'dimension': 'methodology',
            'statement': 'The authors propose a conceptual framework that combines '
                         'Retrieval-Augmented Generation (RAG), vector databases, semantic '
                         'retrieval, knowledge graphs, citation networks, and workflow '
                         'orchestration.',
            'chunk_id': 'https://openalex.org/W7171313681_chunk_1',
            'quote': 'By integrating Retrieval-Augmented Generation (RAG), vector databases, '
                     'knowledge graphs, citation networks, and domain-specific ontologies, '
                     'LLMs improve contextual relevance, reduce hallucinations, and enhance '
                     'research accuracy.',
            'grounding_similarity': 0.8723,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171313681',
            'dimension': 'methodology',
            'statement': 'The framework embeds transformer-based LLMs with vector databases '
                         'for semantic similarity search, employs an LLM-based knowledge '
                         'extraction engine for summarization and concept extraction, and '
                         'constructs dynamic knowledge graphs to support interdisciplinary '
                         'reasoning.',
            'chunk_id': 'https://openalex.org/W7171313681_chunk_21',
            'quote': 'The proposed LLM-Based Knowledge Extraction Engine offers a more '
                     'accurate, reliable, explainable, and scalable approach to AI-powered '
                     'scientific research, ensuring trustworthy knowledge representation for '
                     'intelligent research automation, evidence-based decision-making, and '
                     'interdisciplinary',
            'grounding_similarity': 0.8759,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171313681',
            'dimension': 'methodology',
            'statement': 'Human-in-the-loop validation and explainable AI (XAI) methods are '
                         'integrated for verification and transparency.',
            'chunk_id': 'https://openalex.org/W7171313681_chunk_2',
            'quote': 'Modern AI-assisted research systems increasingly incorporate '
                     'human-in-the-loop validation, explainable AI, and responsible governance '
                     'to ensure trustworthy outcomes.',
            'grounding_similarity': 0.8306,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171313681',
            'dimension': 'dataset',
            'statement': 'The study evaluates the framework on scientific literature '
                         'repositories, employing vector databases to index publications and '
                         'using curated scholarly databases, digital libraries, and citation '
                         'repositories for cross-referencing.',
            'chunk_id': 'https://openalex.org/W7171313681_chunk_28',
            'quote': 'The experimental results showed that the approach of embedding '
                     'transformer models with vector databases significantly outperformed the '
                     'traditional keyword-based search methods in accurately retrieving '
                     'publications that are conceptually related to the query, regardless of '
                     'the use of different',
            'grounding_similarity': 0.7808,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171313681',
            'dimension': 'dataset',
            'statement': 'Specific datasets are not named, but the experimental results '
                         'involve retrieval and summarization tasks on a collection of '
                         'research papers across multiple domains.',
            'chunk_id': 'https://openalex.org/W7171313681_chunk_1',
            'quote': 'By integrating Retrieval-Augmented Generation (RAG), vector databases, '
                     'knowledge graphs, citation networks, and domain-specific ontologies, '
                     'LLMs improve contextual relevance, reduce hallucinations, and enhance '
                     'research accuracy.',
            'grounding_similarity': 0.7004,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171313681',
            'dimension': 'results',
            'statement': 'Embedding transformer models with vector databases significantly '
                         'outperformed traditional keyword-based search in retrieving '
                         'conceptually related publications, regardless of terminology '
                         'differences.',
            'chunk_id': 'https://openalex.org/W7171313681_chunk_28',
            'quote': 'The experimental results showed that the approach of embedding '
                     'transformer models with vector databases significantly outperformed the '
                     'traditional keyword-based search methods in accurately retrieving '
                     'publications that are conceptually related to the query, regardless of '
                     'the use of different',
            'grounding_similarity': 0.9622,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171313681',
            'dimension': 'results',
            'statement': 'The LLM-based knowledge extraction engine achieved high semantic '
                         'accuracy in generating literature summaries, extracting key '
                         'concepts, recognizing methodologies and experimental findings, and '
                         'recommending relevant citations.',
            'chunk_id': 'https://openalex.org/W7171313681_chunk_28',
            'quote': 'The LLM-based knowledge extraction engine demonstrated high semantic '
                     'accuracy for generating literature summaries, extracting key research '
                     'concepts, identifying methods, recognizing experimental results, and '
                     'suggesting relevant citations from scientific literature, in both '
                     'specific and broader',
            'grounding_similarity': 0.9813,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171313681',
            'dimension': 'results',
            'statement': 'Dynamic knowledge graph construction enhanced reasoning, revealing '
                         'hidden semantic relationships among authors, institutions, datasets, '
                         'algorithms, and application domains.',
            'chunk_id': 'https://openalex.org/W7171313681_chunk_28',
            'quote': 'The capability of creating dynamic knowledge graphs further enhanced the '
                     'reasoning capability of the framework, enabling inter-disciplinary '
                     'knowledge discovery and analysis of collaboration by identifying any '
                     'hidden semantic relationships between the authors, institutions, '
                     'datasets, algorithms,',
            'grounding_similarity': 0.8518,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171313681',
            'dimension': 'results',
            'statement': 'The automated pipeline demonstrated substantial time savings in '
                         'document organization and literature review tasks.',
            'chunk_id': 'https://openalex.org/W7171313681_chunk_28',
            'quote': 'The automated research pipeline also showed tremendous savings in '
                     'document',
            'grounding_similarity': 0.9035,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171313681',
            'dimension': 'limitations',
            'statement': 'The paper acknowledges challenges such as hallucination, bias, '
                         'outdated knowledge, explainability, privacy, intellectual property, '
                         'reproducibility, and computational requirements.',
            'chunk_id': 'https://openalex.org/W7171313681_chunk_1',
            'quote': 'However, challenges such as hallucination, bias, outdated knowledge, '
                     'explainability, privacy, intellectual property, reproducibility, and '
                     'computational requirements',
            'grounding_similarity': 0.8957,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171313681',
            'dimension': 'limitations',
            'statement': 'It also notes the need for human-in-the-loop verification and '
                         'responsible governance to ensure trustworthy outcomes.',
            'chunk_id': 'https://openalex.org/W7171313681_chunk_2',
            'quote': 'Modern AI-assisted research systems increasingly incorporate '
                     'human-in-the-loop validation, explainable AI, and responsible governance '
                     'to ensure trustworthy outcomes.',
            'grounding_similarity': 0.7598,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7171313681',
            'dimension': 'future_work',
            'statement': 'Future research directions include further integration of '
                         'human-in-the-loop validation, explainable AI, and responsible '
                         'governance mechanisms to enhance trustworthiness, as well as '
                         'expanding the framework to support more extensive interdisciplinary '
                         'collaboration and evidence-based decision-making.',
            'chunk_id': 'https://openalex.org/W7171313681_chunk_2',
            'quote': 'Modern AI-assisted research systems increasingly incorporate '
                     'human-in-the-loop validation, explainable AI, and responsible governance '
                     'to ensure trustworthy outcomes.',
            'grounding_similarity': 0.8253,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407028609',
            'dimension': 'problem',
            'statement': 'The paper addresses the challenge of automating the final stages of '
                         'systematic literature reviews (SLRs) by leveraging fine‑tuned large '
                         'language models (LLMs) to perform knowledge synthesis while '
                         'mitigating hallucination and ensuring traceability of responses.',
            'chunk_id': 'https://openalex.org/W4407028609_chunk_1',
            'quote': 'of finetuned Large Language Models (LLMs) to automate Systematic '
                     'Literature Reviews (SLRs), presenting a significant and novel '
                     'contribution in integrating AI to enhance academic research '
                     'methodologies.',
            'grounding_similarity': 0.8648,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407028609',
            'dimension': 'methodology',
            'statement': 'It employs parameter‑efficient fine‑tuning (PEFT) techniques such as '
                         'NEFTune and LoRA on open‑source LLMs, trained on datasets '
                         'automatically extracted from a corpus of selected academic papers '
                         'specific to an SLR topic.',
            'chunk_id': 'https://openalex.org/W4407028609_chunk_84',
            'quote': 'This study has demonstrated that indeed modern PEFT finetuning '
                     'approaches like NEFTune and LoRA can be used for finetuning effective '
                     'LLMs (RQ1), and the proposed automated approach for extracting '
                     'finetuning datasets from a corpus of selected academic papers can '
                     'successfully',
            'grounding_similarity': 0.8511,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407028609',
            'dimension': 'methodology',
            'statement': 'The framework includes mechanisms to track LLM outputs back to their '
                         'source documents and to reduce hallucination.',
            'chunk_id': 'https://openalex.org/W4407028609_chunk_2',
            'quote': 'Our research proposed solutions for mitigating LLM hallucination and '
                     'proposed mechanisms for tracking LLM responses to their sources of '
                     'information, thus demonstrating how this approach can meet the rigorous '
                     'demands of scholarly research.',
            'grounding_similarity': 0.7695,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407028609',
            'dimension': 'dataset',
            'statement': 'The datasets consist of automatically extracted, SLR‑specific '
                         'training data derived from the selected academic papers for each '
                         'target review.',
            'chunk_id': 'https://openalex.org/W4407028609_chunk_7',
            'quote': 'This work devises a novel way to automatically extract information from '
                     'a set of academic papers in order to create SLR-specific datasets which '
                     'can be leveraged for finetuning LLMs so that they can support question '
                     'and answering downstream tasks.',
            'grounding_similarity': 0.821,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407028609',
            'dimension': 'dataset',
            'statement': 'The study also uses a benchmark PRISMA‑conforming SLR as ground '
                         'truth for validation.',
            'chunk_id': 'https://openalex.org/W4407028609_chunk_86',
            'quote': 'Finally, the entire proposed framework for automating the final stages '
                     'of an SLR was validated for viability by replicating an actual published '
                     'SLR study, using it as a benchmark and source of ground truth.',
            'grounding_similarity': 0.8228,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407028609',
            'dimension': 'results',
            'statement': 'The fine‑tuned LLMs achieved high factual fidelity in their '
                         'responses, successfully replicating the findings of an existing '
                         'PRISMA‑conforming SLR.',
            'chunk_id': 'https://openalex.org/W4407028609_chunk_2',
            'quote': 'The results maintained high fidelity in factual accuracy in LLM '
                     'responses, and were validated through the replication of an existing '
                     'PRISMA-conforming SLR.',
            'grounding_similarity': 0.9343,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407028609',
            'dimension': 'results',
            'statement': 'The approach demonstrated reliable question‑answering capabilities '
                         'and effective knowledge synthesis across multiple documents.',
            'chunk_id': 'https://openalex.org/W4407028609_chunk_84',
            'quote': 'datasets from a corpus of selected academic papers can successfully '
                     'construct finetuning datasets (RQ2) that support question and answer '
                     'downstream tasks necessary for executing',
            'grounding_similarity': 0.8024,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407028609',
            'dimension': 'limitations',
            'statement': 'The evidence notes that LLMs still exhibit hallucination and limited '
                         'efficacy in narrow domain areas, and that their decision processes '
                         'remain opaque, requiring further development to fully audit and '
                         'control outputs.',
            'chunk_id': 'https://openalex.org/W4407028609_chunk_7',
            'quote': '[28] found that LLM usage for SLRs is hampered by limitations such as '
                     'reduced efficacy in domain-specific and narrower subject areas, their '
                     'propensity for hallucinating and generating misleading information, '
                     'alongside their opaque decision processes which cannot be audited.',
            'grounding_similarity': 0.7508,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W4407028609',
            'dimension': 'future_work',
            'statement': 'The authors advocate updating PRISMA reporting guidelines to '
                         'incorporate AI‑driven processes, suggesting future work on '
                         'methodological transparency and reliability for AI‑enhanced SLRs.',
            'chunk_id': 'https://openalex.org/W4407028609_chunk_2',
            'quote': 'Given the potential of this approach and its applicability across all '
                     'research domains, this foundational study also advocated for updating '
                     'PRISMA reporting guidelines to incorporate AI-driven processes, ensuring '
                     'methodological transparency and reliability in future SLRs.',
            'grounding_similarity': 0.878,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7166808050',
            'dimension': 'problem',
            'statement': 'The paper addresses the need for a framework that supports iterative '
                         'refinement and problem reformulation across successive scientific '
                         'research cycles, enabling continuous and open-ended scientific '
                         'exploration.',
            'chunk_id': 'https://openalex.org/W7166808050_chunk_8',
            'quote': 'This gap highlights the need for frameworks that support iterative '
                     'refinement and problem reformulation across successive research cycles.',
            'grounding_similarity': 0.8353,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7166808050',
            'dimension': 'methodology',
            'statement': 'It proposes a heterogeneous multi-agent framework mirroring '
                         'real-world research laboratories, where diverse LLM-based agents '
                         'operate under role-specific objectives and collaborate through a '
                         'multi-round feedback loop with a bio-inspired evolution mechanism '
                         '(selection, crossover, mutation).',
            'chunk_id': 'https://openalex.org/W7166808050_chunk_6',
            'quote': '• We construct a heterogeneous multi-agent framework that mirrors '
                     'real-world research laboratories, where diverse agents operate un- der '
                     'role-specific objectives.',
            'grounding_similarity': 0.8858,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7166808050',
            'dimension': 'dataset',
            'statement': 'The framework is grounded on datasets derived from real scientists, '
                         'enabling authentic simulation of collaborative scientific workflows.',
            'chunk_id': 'https://openalex.org/W7166808050_chunk_6',
            'quote': 'The framework is grounded on datasets derived from real sci- entists, '
                     'enabling more authentic simulation of collaborative scientific '
                     'workflows.',
            'grounding_similarity': 0.9566,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7166808050',
            'dimension': 'results',
            'statement': 'In real-world research topic experiments, the system (EvoSci) '
                         'significantly outperforms strong baselines in LLM-based structured '
                         'peer-review and comparative ranking evaluations, achieving the '
                         'highest overall peer-review score (ICLR 4.90) and top ranking '
                         '(Top‑10 = 54).',
            'chunk_id': 'https://openalex.org/W7166808050_chunk_2',
            'quote': 'real-world research topics demonstrate that EvoSci significantly '
                     'outperforms strong baselines in LLM-based structured peer-review and '
                     'comparative rank- ing evaluations, achieving the highest overall '
                     'peer-review score (ICLR 4.',
            'grounding_similarity': 0.9539,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7166808050',
            'dimension': 'limitations',
            'statement': 'The paper notes that establishing objective and high-quality '
                         'evaluation mechanisms is challenging, as it is essential for '
                         'LLM-based agents to better assess their own reasoning and outputs '
                         'for effective self-improvement and sustained innovation.',
            'chunk_id': 'https://openalex.org/W7166808050_chunk_38',
            'quote': 'A key chal- lenge in achieving such continuous evolution lies in '
                     'establishing more objective and high-quality evalu- ation mechanisms '
                     'that allow LLM-based agents to better assess their own reasoning and '
                     'outputs, which is essential for truly effective self-improvement and '
                     'sustained',
            'grounding_similarity': 0.8668,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7160249182',
            'dimension': 'problem',
            'statement': 'The paper addresses the challenge of automatically identifying and '
                         'prioritizing research gaps and generating high‑quality research '
                         'ideas from large collections of scholarly literature, without '
                         'relying on domain‑specific training data.',
            'chunk_id': 'https://openalex.org/W7160249182_chunk_6',
            'quote': 'across thirteen research domains without requiring domain-specific '
                     'tuning.',
            'grounding_similarity': 0.833,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7160249182',
            'dimension': 'methodology',
            'statement': 'It employs a five‑stage, multi‑agent pipeline orchestrated with '
                         'LangGraph: (1) query parsing, (2) literature retrieval, (3) '
                         'clustering of retrieved papers, (4) gap analysis using an '
                         'unsupervised composite score that combines statistical '
                         'under‑representation, temporal publication trends, and '
                         'citation‑network centrality, and (5) idea synthesis via a large '
                         'language model that receives structured cluster‑gap context and '
                         'produces ranked, ready‑for‑follow‑up research ideas.',
            'chunk_id': 'https://openalex.org/W7160249182_chunk_6',
            'quote': 'The four principal technical contributions reported here are: (i) a '
                     'five-stage agent architecture that separates query parsing, retrieval, '
                     'clustering, gap analysis, and Idea synthesis into independently testable '
                     'components; (ii) a composite gap-significance formula that weights '
                     'statistical',
            'grounding_similarity': 0.826,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7160249182',
            'dimension': 'results',
            'statement': 'Experiments demonstrate that the pipeline runs end‑to‑end in under '
                         'two minutes, produces insights that would normally require days of '
                         'manual review, and yields a three‑dimensional idea quality metric '
                         'that has been empirically validated against structured expert '
                         'evaluation.',
            'chunk_id': 'https://openalex.org/W7160249182_chunk_5',
            'quote': 'The system runs end-to-end in under two minutes, surfaces insights that '
                     'human reviewers would typically take days to compile, and does so '
                     'reproducibly across thirteen research domains without requiring '
                     'domain-specific tuning.',
            'grounding_similarity': 0.787,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7160249182',
            'dimension': 'results',
            'statement': 'The system’s gap‑significance scoring and idea ranking outperform '
                         'baseline approaches that rely on flat paper lists or manual gap '
                         'statements.',
            'chunk_id': 'https://openalex.org/W7160249182_chunk_6',
            'quote': 'composite gap-significance formula that weights statistical '
                     'under-representation, temporal publication trends, and citation-network '
                     'centrality in a single normalised score; (iii) a three- dimensional idea '
                     'quality metric that has been empirically validated against structured '
                     'expert evaluation; and',
            'grounding_similarity': 0.7599,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7160249182',
            'dimension': 'limitations',
            'statement': 'The approach is unsupervised and therefore may miss nuanced, '
                         'domain‑specific gaps that require expert labeling',
            'chunk_id': 'https://openalex.org/W7160249182_chunk_9',
            'quote': 'Abd- alrazaq and collaborators [10] trained supervised classifiers on '
                     'labelled review protocols to predict gap labels automatically, achieving '
                     'good precision on clinical sub-domains but requiring expensive human- '
                     'annotated training sets and performing poorly outside the labelling '
                     'distribution.',
            'grounding_similarity': 0.7615,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7160249182',
            'dimension': 'limitations',
            'statement': 'it relies on the quality of the underlying citation network and '
                         'publication metadata, which can be incomplete or biased.',
            'chunk_id': 'https://openalex.org/W7160249182_chunk_9',
            'quote': 'Tools such as Rayyan and Elicit accelerate article screening but still '
                     'depend on researchers to articulate gaps themselves [10].',
            'grounding_similarity': 0.6937,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7160249182',
            'dimension': 'limitations',
            'statement': 'The evaluation is limited to thirteen domains and does not report '
                         'performance on highly specialized or emerging fields.',
            'chunk_id': 'https://openalex.org/W7160249182_chunk_6',
            'quote': 'across thirteen research domains without requiring domain-specific '
                     'tuning.',
            'grounding_similarity': 0.6321,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7126221181',
            'dimension': 'problem',
            'statement': 'The paper investigates how Large Language Models (LLMs) can '
                         'transform scientific computing and research by automating data '
                         'preprocessing, literature review, knowledge integration, problem '
                         'solving, and scientific programming.',
            'chunk_id': 'https://openalex.org/W7126221181_chunk_1',
            'quote': 'This paper explores the transformative role of Large Language Models '
                     '(LLMs) in scientific computing and research.',
            'grounding_similarity': 0.9568,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7126221181',
            'dimension': 'methodology',
            'statement': 'It surveys and discusses the application of LLMs—particularly '
                         'through techniques such as Named Entity Recognition (NER), text '
                         'summarization, knowledge‑graph construction, and code generation—to '
                         'convert unstructured scientific text into structured data, '
                         'accelerate literature reviews, and support simulation verification '
                         'and code generation.',
            'chunk_id': 'https://openalex.org/W7126221181_chunk_1',
            'quote': 'By utilizing techniques such as Named Entity Recognition (NER), LLMs can '
                     'extract critical information like chemical names, experimental '
                     'conditions, and research findings, significantly reducing manual '
                     'efforts.',
            'grounding_similarity': 0.8315,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7126221181',
            'dimension': 'dataset',
            'statement': 'The paper does not report the use of a specific experimental dataset',
            'chunk_id': 'https://openalex.org/W7126221181_chunk_1',
            'quote': 'As scientific datasets increasingly contain unstructured or '
                     'semi-structured text, preprocessing and structuring this data are '
                     'crucial challenges.',
            'grounding_similarity': 0.6274,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7126221181',
            'dimension': 'dataset',
            'statement': 'it focuses on the general use of LLMs across scientific domains such '
                         'as computational chemistry, bioinformatics, and climate science.',
            'chunk_id': 'https://openalex.org/W7126221181_chunk_10',
            'quote': 'LLMs can also be applied to climate science.',
            'grounding_similarity': 0.8429,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7126221181',
            'dimension': 'results',
            'statement': 'The paper reports that LLMs can significantly reduce manual effort '
                         'in data extraction, literature summarization, and code generation, '
                         'thereby enhancing research efficiency and enabling faster scientific '
                         'discoveries across multiple fields.',
            'chunk_id': 'https://openalex.org/W7126221181_chunk_6',
            'quote': 'LLMs can automate literature surveys, information retrieval, and '
                     'research summarization.',
            'grounding_similarity': 0.8379,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7126221181',
            'dimension': 'limitations',
            'statement': 'The paper identifies key limitations: (1) hallucinations produced by '
                         'LLMs, which can lead to inaccurate or fabricated outputs',
            'chunk_id': 'https://openalex.org/W7126221181_chunk_26',
            'quote': 'However, despite their immense potential, LLMs still face numerous '
                     'challenges in scientific applications, particularly the phenomenon of '
                     '“hallucinations” and data privacy concerns.',
            'grounding_similarity': 0.7944,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7126221181',
            'dimension': 'limitations',
            'statement': '(2) data privacy concerns when handling sensitive scientific '
                         'information',
            'chunk_id': 'https://openalex.org/W7126221181_chunk_1',
            'quote': 'As scientific datasets increasingly contain unstructured or '
                     'semi-structured text, preprocessing and structuring this data are '
                     'crucial challenges.',
            'grounding_similarity': 0.6744,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7126221181',
            'dimension': 'limitations',
            'statement': 'and (3) the need for rigorous validation of LLM outputs to ensure '
                         'reliability.',
            'chunk_id': 'https://openalex.org/W7126221181_chunk_26',
            'quote': 'To maximize the advantages of LLMs, researchers must rigorously validate '
                     'their outputs and ensure robust data security and privacy protections.',
            'grounding_similarity': 0.7925,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7126221181',
            'dimension': 'future_work',
            'statement': 'Future directions mentioned include: (1) developing robust data '
                         'security and privacy solutions for LLM usage',
            'chunk_id': 'https://openalex.org/W7126221181_chunk_26',
            'quote': 'In conclusion, while there are significant challenges to the adoption of '
                     'LLMs in scientific computing, the future looks promising.',
            'grounding_similarity': 0.8079,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7126221181',
            'dimension': 'future_work',
            'statement': '(2) improving validation protocols to mitigate hallucinations',
            'chunk_id': 'https://openalex.org/W7126221181_chunk_10',
            'quote': '1 Hallucination in Large Language Models When using large language '
                     "models (LLMs), such as OpenAI's GPT series, for scientific computing, "
                     'the phenomenon of “hallucination”',
            'grounding_similarity': 0.7807,
            'grounded': True},
           {'paper_id': 'https://openalex.org/W7126221181',
            'dimension': 'future_work',
            'statement': 'and (3) advancing LLM capabilities to support interdisciplinary '
                         'research and cross‑domain knowledge synthesis.',
            'chunk_id': 'https://openalex.org/W7126221181_chunk_26',
            'quote': 'Through increased efficiency, interdisciplinary collaboration, and '
                     'enhanced accuracy, LLMs are poised to revolutionize scientific research '
                     'and computing, making it more accessible and impactful than ever before.',
            'grounding_similarity': 0.783,
            'grounded': True}],
 'clusters': [{'cluster_id': 'LIM-1',
               'dimension': 'limitations',
               'paper_ids': ['https://openalex.org/W7214432694',
                             'https://openalex.org/W7171304406',
                             'https://openalex.org/W7166808050'],
               'members': [{'paper_id': 'https://openalex.org/W7214432694',
                            'chunk_id': 'https://openalex.org/W7214432694_chunk_14',
                            'statement': 'The evidence indicates that agent coordination and '
                                         'evaluation remain challenging, automatic '
                                         'knowledge‑graph construction can introduce incorrect '
                                         'or duplicate relationships, and long‑context '
                                         'processing still limits information utilization.',
                            'quote': 'Looking at this, whether agent coordination increases '
                                     'evidence coverage and decreases unsubstantiated claims.'},
                           {'paper_id': 'https://openalex.org/W7171304406',
                            'chunk_id': 'https://openalex.org/W7171304406_chunk_8',
                            'statement': 'and the current lack of effective multi‑agent '
                                         'coordination mechanisms for complex scientific '
                                         'goals.',
                            'quote': 'One of the other significant weaknesses is the absence '
                                     'of an effective mechanism of multi-agent coordination '
                                     'that can break down the complex scientific goals'},
                           {'paper_id': 'https://openalex.org/W7166808050',
                            'chunk_id': 'https://openalex.org/W7166808050_chunk_38',
                            'statement': 'The paper notes that establishing objective and '
                                         'high-quality evaluation mechanisms is challenging, '
                                         'as it is essential for LLM-based agents to better '
                                         'assess their own reasoning and outputs for effective '
                                         'self-improvement and sustained innovation.',
                            'quote': 'A key chal- lenge in achieving such continuous evolution '
                                     'lies in establishing more objective and high-quality '
                                     'evalu- ation mechanisms that allow LLM-based agents to '
                                     'better assess their own reasoning and outputs, which is '
                                     'essential for truly effective self-improvement and '
                                     'sustained'}],
               'representative_statement': 'The evidence indicates that agent coordination and '
                                           'evaluation remain challenging, automatic '
                                           'knowledge‑graph construction can introduce '
                                           'incorrect or duplicate relationships, and '
                                           'long‑context processing still limits information '
                                           'utilization.',
               'mean_similarity': 0.7484},
              {'cluster_id': 'LIM-2',
               'dimension': 'limitations',
               'paper_ids': ['https://openalex.org/W7171313681',
                             'https://openalex.org/W7126221181'],
               'members': [{'paper_id': 'https://openalex.org/W7171313681',
                            'chunk_id': 'https://openalex.org/W7171313681_chunk_1',
                            'statement': 'The paper acknowledges challenges such as '
                                         'hallucination, bias, outdated knowledge, '
                                         'explainability, privacy, intellectual property, '
                                         'reproducibility, and computational requirements.',
                            'quote': 'However, challenges such as hallucination, bias, '
                                     'outdated knowledge, explainability, privacy, '
                                     'intellectual property, reproducibility, and '
                                     'computational requirements'},
                           {'paper_id': 'https://openalex.org/W7126221181',
                            'chunk_id': 'https://openalex.org/W7126221181_chunk_26',
                            'statement': 'The paper identifies key limitations: (1) '
                                         'hallucinations produced by LLMs, which can lead to '
                                         'inaccurate or fabricated outputs',
                            'quote': 'However, despite their immense potential, LLMs still '
                                     'face numerous challenges in scientific applications, '
                                     'particularly the phenomenon of “hallucinations” and data '
                                     'privacy concerns.'}],
               'representative_statement': 'The paper acknowledges challenges such as '
                                           'hallucination, bias, outdated knowledge, '
                                           'explainability, privacy, intellectual property, '
                                           'reproducibility, and computational requirements.',
               'mean_similarity': 0.7591},
              {'cluster_id': 'FUT-1',
               'dimension': 'future_work',
               'paper_ids': ['https://openalex.org/W7214432694',
                             'https://openalex.org/W7171313681',
                             'https://openalex.org/W4407028609'],
               'members': [{'paper_id': 'https://openalex.org/W7214432694',
                            'chunk_id': 'https://openalex.org/W7214432694_chunk_19',
                            'statement': 'Future directions include developing and evaluating '
                                         'the multi‑agent IRIS architecture, improving agent '
                                         'coordination and evaluation, refining automatic '
                                         'knowledge‑graph extraction to reduce errors, and '
                                         'integrating systematic review guidelines (e.g., '
                                         'PRISMA 2020) to enhance transparency and '
                                         'traceability.',
                            'quote': 'Automatic KG Reviews approaches for acquiring and '
                                     'refining knowledge- graph information Provides guidance '
                                     'for building the IRIS research graph Automatic '
                                     'extraction may introduce incorrect or duplicate '
                                     'relationships PRISMA 2020 [16] Systematic review '
                                     'reporting Provides guidelines for making the'},
                           {'paper_id': 'https://openalex.org/W7171313681',
                            'chunk_id': 'https://openalex.org/W7171313681_chunk_2',
                            'statement': 'Future research directions include further '
                                         'integration of human-in-the-loop validation, '
                                         'explainable AI, and responsible governance '
                                         'mechanisms to enhance trustworthiness, as well as '
                                         'expanding the framework to support more extensive '
                                         'interdisciplinary collaboration and evidence-based '
                                         'decision-making.',
                            'quote': 'Modern AI-assisted research systems increasingly '
                                     'incorporate human-in-the-loop validation, explainable '
                                     'AI, and responsible governance to ensure trustworthy '
                                     'outcomes.'},
                           {'paper_id': 'https://openalex.org/W4407028609',
                            'chunk_id': 'https://openalex.org/W4407028609_chunk_2',
                            'statement': 'The authors advocate updating PRISMA reporting '
                                         'guidelines to incorporate AI‑driven processes, '
                                         'suggesting future work on methodological '
                                         'transparency and reliability for AI‑enhanced SLRs.',
                            'quote': 'Given the potential of this approach and its '
                                     'applicability across all research domains, this '
                                     'foundational study also advocated for updating PRISMA '
                                     'reporting guidelines to incorporate AI-driven processes, '
                                     'ensuring methodological transparency and reliability in '
                                     'future SLRs.'}],
               'representative_statement': 'Future directions include developing and '
                                           'evaluating the multi‑agent IRIS architecture, '
                                           'improving agent coordination and evaluation, '
                                           'refining automatic knowledge‑graph extraction to '
                                           'reduce errors, and integrating systematic review '
                                           'guidelines (e.g., PRISMA 2020) to enhance '
                                           'transparency and traceability.',
               'mean_similarity': 0.7477},
              {'cluster_id': 'FUT-2',
               'dimension': 'future_work',
               'paper_ids': ['https://openalex.org/W7222723116',
                             'https://openalex.org/W7126221181'],
               'members': [{'paper_id': 'https://openalex.org/W7222723116',
                            'chunk_id': 'https://openalex.org/W7222723116_chunk_7',
                            'statement': 'Future work will focus on integrating '
                                         'domain‑specific LLMs, GraphRAG, and additional '
                                         'scholarly databases to further improve retrieval '
                                         'quality and research insight generation.',
                            'quote': 'Future work will focus on integrating domain-specific '
                                     'LLMs, GraphRAG, and additional scholarly databases to '
                                     'further improve retrieval quality and research'},
                           {'paper_id': 'https://openalex.org/W7126221181',
                            'chunk_id': 'https://openalex.org/W7126221181_chunk_26',
                            'statement': 'and (3) advancing LLM capabilities to support '
                                         'interdisciplinary research and cross‑domain '
                                         'knowledge synthesis.',
                            'quote': 'Through increased efficiency, interdisciplinary '
                                     'collaboration, and enhanced accuracy, LLMs are poised to '
                                     'revolutionize scientific research and computing, making '
                                     'it more accessible and impactful than ever before.'}],
               'representative_statement': 'Future work will focus on integrating '
                                           'domain‑specific LLMs, GraphRAG, and additional '
                                           'scholarly databases to further improve retrieval '
                                           'quality and research insight generation.',
               'mean_similarity': 0.7628},
              {'cluster_id': 'MET-1',
               'dimension': 'methodology',
               'paper_ids': ['https://openalex.org/W7222723116',
                             'https://openalex.org/W7214432694',
                             'https://openalex.org/W7171304406',
                             'https://openalex.org/W7171313681',
                             'https://openalex.org/W7160249182',
                             'https://openalex.org/W7126221181'],
               'members': [{'paper_id': 'https://openalex.org/W7222723116',
                            'chunk_id': 'https://openalex.org/W7222723116_chunk_2',
                            'statement': 'It performs semantic literature retrieval, '
                                         'LLM‑driven summarization, embedding‑based trend '
                                         'analysis, and graph‑based research gap detection.',
                            'quote': 'Large Language Models (LLMs), Semantic Retrieval, '
                                     'Research Gap Detection, Multi-Agent Systems, Academic '
                                     'Literature Analysis.'},
                           {'paper_id': 'https://openalex.org/W7214432694',
                            'chunk_id': 'https://openalex.org/W7214432694_chunk_1',
                            'statement': 'The system leverages retrieval‑augmented generation '
                                         '(RAG), long‑context processing, step‑by‑step '
                                         'reasoning, and knowledge‑graph construction '
                                         'techniques to support these tasks.',
                            'quote': 'Recent progress in artificial intelligence, natural '
                                     'language processing, large language models, '
                                     'retrieval-augmented generation, knowledge graphs and '
                                     'autonomous agent workflows creates a chance to support '
                                     'this process.'},
                           {'paper_id': 'https://openalex.org/W7171304406',
                            'chunk_id': 'https://openalex.org/W7171304406_chunk_1',
                            'statement': 'An Agentic AI Framework that combines Large Language '
                                         'Models (LLMs), Retrieval‑Augmented Generation (RAG), '
                                         'collaborative multi‑agent systems, semantic '
                                         'knowledge graphs, reinforcement learning, '
                                         'explainable AI, and evidence‑based decision‑making '
                                         'to autonomously manage literature review, knowledge '
                                         'extraction, hypothesis generation, experiment '
                                         'planning, statistical analysis, citation '
                                         'verification, manuscript preparation, and research '
                                         'validation.',
                            'quote': 'This paper proposes an Agentic AI Framework for '
                                     'Autonomous Scientific Research Assistance that '
                                     'integrates Large Language Models (LLMs), '
                                     'Retrieval-Augmented Generation (RAG), and collaborative '
                                     'multi- agent systems to automate key research '
                                     'activities.'},
                           {'paper_id': 'https://openalex.org/W7171313681',
                            'chunk_id': 'https://openalex.org/W7171313681_chunk_1',
                            'statement': 'The authors propose a conceptual framework that '
                                         'combines Retrieval-Augmented Generation (RAG), '
                                         'vector databases, semantic retrieval, knowledge '
                                         'graphs, citation networks, and workflow '
                                         'orchestration.',
                            'quote': 'By integrating Retrieval-Augmented Generation (RAG), '
                                     'vector databases, knowledge graphs, citation networks, '
                                     'and domain-specific ontologies, LLMs improve contextual '
                                     'relevance, reduce hallucinations, and enhance research '
                                     'accuracy.'},
                           {'paper_id': 'https://openalex.org/W7160249182',
                            'chunk_id': 'https://openalex.org/W7160249182_chunk_6',
                            'statement': 'It employs a five‑stage, multi‑agent pipeline '
                                         'orchestrated with LangGraph: (1) query parsing, (2) '
                                         'literature retrieval, (3) clustering of retrieved '
                                         'papers, (4) gap analysis using an unsupervised '
                                         'composite score that combines statistical '
                                         'under‑representation, temporal publication trends, '
                                         'and citation‑network centrality, and (5) idea '
                                         'synthesis via a large language model that receives '
                                         'structured cluster‑gap context and produces ranked, '
                                         'ready‑for‑follow‑up research ideas.',
                            'quote': 'The four principal technical contributions reported here '
                                     'are: (i) a five-stage agent architecture that separates '
                                     'query parsing, retrieval, clustering, gap analysis, and '
                                     'Idea synthesis into independently testable components; '
                                     '(ii) a composite gap-significance formula that weights '
                                     'statistical'},
                           {'paper_id': 'https://openalex.org/W7126221181',
                            'chunk_id': 'https://openalex.org/W7126221181_chunk_1',
                            'statement': 'It surveys and discusses the application of '
                                         'LLMs—particularly through techniques such as Named '
                                         'Entity Recognition (NER), text summarization, '
                                         'knowledge‑graph construction, and code generation—to '
                                         'convert unstructured scientific text into structured '
                                         'data, accelerate literature reviews, and support '
                                         'simulation verification and code generation.',
                            'quote': 'By utilizing techniques such as Named Entity Recognition '
                                     '(NER), LLMs can extract critical information like '
                                     'chemical names, experimental conditions, and research '
                                     'findings, significantly reducing manual efforts.'}],
               'representative_statement': 'It performs semantic literature retrieval, '
                                           'LLM‑driven summarization, embedding‑based trend '
                                           'analysis, and graph‑based research gap detection.',
               'mean_similarity': 0.7504},
              {'cluster_id': 'MET-2',
               'dimension': 'methodology',
               'paper_ids': ['https://openalex.org/W7233992214',
                             'https://openalex.org/W7222723116',
                             'https://openalex.org/W4407359160',
                             'https://openalex.org/W7214432694',
                             'https://openalex.org/W7166808050'],
               'members': [{'paper_id': 'https://openalex.org/W7233992214',
                            'chunk_id': 'https://openalex.org/W7233992214_chunk_7',
                            'statement': 'A constrained multi‑agent system is employed, '
                                         'comprising five role‑specialized agents (Query '
                                         'Planner, Researcher, Writer, Reviewer, Prioritizer) '
                                         'coordinated via a deterministic state machine and a '
                                         'persisted typed blackboard.',
                            'quote': 'Positioning within agent and MAS research From an '
                                     'agent-theoretic perspective, the system is a constrained '
                                     'multi-agent organization defined by four elements.'},
                           {'paper_id': 'https://openalex.org/W7222723116',
                            'chunk_id': 'https://openalex.org/W7222723116_chunk_4',
                            'statement': 'The system is built on a modular architecture using '
                                         'LangChain with multiple agents.',
                            'quote': 'AI Agent Pipeline and Task Coordination The main '
                                     'intelligence of the system is implemented using '
                                     'LangChain.'},
                           {'paper_id': 'https://openalex.org/W4407359160',
                            'chunk_id': 'https://openalex.org/W4407359160_chunk_4',
                            'statement': 'Agents communicate via a blackboard architecture, '
                                         'enabling cross‑agent verification and iterative '
                                         'refinement.',
                            'quote': 'First, the multi-agent architecture enables cross-agent '
                                     'verification, enhancing the reliability of extracted '
                                     'knowledge.'},
                           {'paper_id': 'https://openalex.org/W7214432694',
                            'chunk_id': 'https://openalex.org/W7214432694_chunk_14',
                            'statement': 'IRIS employs a multi‑agent workflow where '
                                         'domain‑specialized autonomous agents collaborate on '
                                         'retrieval, analysis, summarization, gap detection, '
                                         'and report generation.',
                            'quote': 'Multiple agents collaborate on retrieval, analysis, '
                                     'summarization, gap detection, and report generation '
                                     '[1].'},
                           {'paper_id': 'https://openalex.org/W7166808050',
                            'chunk_id': 'https://openalex.org/W7166808050_chunk_6',
                            'statement': 'It proposes a heterogeneous multi-agent framework '
                                         'mirroring real-world research laboratories, where '
                                         'diverse LLM-based agents operate under role-specific '
                                         'objectives and collaborate through a multi-round '
                                         'feedback loop with a bio-inspired evolution '
                                         'mechanism (selection, crossover, mutation).',
                            'quote': '• We construct a heterogeneous multi-agent framework '
                                     'that mirrors real-world research laboratories, where '
                                     'diverse agents operate un- der role-specific '
                                     'objectives.'}],
               'representative_statement': 'A constrained multi‑agent system is employed, '
                                           'comprising five role‑specialized agents (Query '
                                           'Planner, Researcher, Writer, Reviewer, '
                                           'Prioritizer) coordinated via a deterministic state '
                                           'machine and a persisted typed blackboard.',
               'mean_similarity': 0.7483},
              {'cluster_id': 'DAT-1',
               'dimension': 'dataset',
               'paper_ids': ['https://openalex.org/W7233992214',
                             'https://openalex.org/W4407359160',
                             'https://openalex.org/W7171304406',
                             'https://openalex.org/W7171313681'],
               'members': [{'paper_id': 'https://openalex.org/W7233992214',
                            'chunk_id': 'https://openalex.org/W7233992214_chunk_1',
                            'statement': 'Specific datasets are not named, but the evaluation '
                                         'involved three real corpora of scientific '
                                         'literature.',
                            'quote': 'Identifying promising research gaps remains a '
                                     'significant challenge in scientific research.'},
                           {'paper_id': 'https://openalex.org/W4407359160',
                            'chunk_id': 'https://openalex.org/W4407359160_chunk_1',
                            'statement': 'Experiments were conducted on 1,200 PubMed articles '
                                         'drawn from three distinct scientific domains.',
                            'quote': 'Experiments on 1,200 PubMed articles from three '
                                     'different domains demonstrate the effectiveness of KARMA '
                                     'in knowledge graph enrichment, with the identification '
                                     'of up to 38,230 new entities while achieving 83.'},
                           {'paper_id': 'https://openalex.org/W7171304406',
                            'chunk_id': 'https://openalex.org/W7171304406_chunk_36',
                            'statement': 'The evaluation used a collection of 4 publications, '
                                         'conference proceedings, patents, technical reports, '
                                         'experimental repositories, simulation datasets, '
                                         'institutional databases, and open‑access research '
                                         'platforms to test literature retrieval, hypothesis '
                                         'generation, citation verification, and other '
                                         'metrics.',
                            'quote': 'Experimental evaluation shows that the proposed '
                                     'framework outperforms conventional AI-assisted research '
                                     'systems in literature retrieval accuracy, workflow '
                                     'automation, quality of hypothesis generation, citation '
                                     'verification, explainability, manuscript preparation, '
                                     'and research productivity.'},
                           {'paper_id': 'https://openalex.org/W7171313681',
                            'chunk_id': 'https://openalex.org/W7171313681_chunk_1',
                            'statement': 'Specific datasets are not named, but the '
                                         'experimental results involve retrieval and '
                                         'summarization tasks on a collection of research '
                                         'papers across multiple domains.',
                            'quote': 'By integrating Retrieval-Augmented Generation (RAG), '
                                     'vector databases, knowledge graphs, citation networks, '
                                     'and domain-specific ontologies, LLMs improve contextual '
                                     'relevance, reduce hallucinations, and enhance research '
                                     'accuracy.'}],
               'representative_statement': 'Specific datasets are not named, but the '
                                           'experimental results involve retrieval and '
                                           'summarization tasks on a collection of research '
                                           'papers across multiple domains.',
               'mean_similarity': 0.7228},
              {'cluster_id': 'DAT-2',
               'dimension': 'dataset',
               'paper_ids': ['https://openalex.org/W7222723116',
                             'https://openalex.org/W7214432694',
                             'https://openalex.org/W7126221181'],
               'members': [{'paper_id': 'https://openalex.org/W7222723116',
                            'chunk_id': 'https://openalex.org/W7222723116_chunk_1',
                            'statement': 'The paper does not specify a particular dataset or '
                                         'benchmark',
                            'quote': '0 International LLM-Driven Research Assistant for '
                                     'Automated Literature Analysis and Research Gap Detection '
                                     'Bijoyini Bagchi1, Kaushik Banerjee1, Tunir De1, Trina '
                                     'Chakraborty1, Sayudh Chakraborty1 Department of Computer '
                                     'Science and Engineering Institute of Engineering and '
                                     'Management, Kolkata, India'},
                           {'paper_id': 'https://openalex.org/W7214432694',
                            'chunk_id': 'https://openalex.org/W7214432694_chunk_19',
                            'statement': 'No specific experimental dataset is introduced',
                            'quote': 'AI implementation LLM scientific assistance survey [17] '
                                     'LLMs in research Reviews how LLMs can support different '
                                     'scientific activities Helps identify possible benefits'},
                           {'paper_id': 'https://openalex.org/W7126221181',
                            'chunk_id': 'https://openalex.org/W7126221181_chunk_1',
                            'statement': 'The paper does not report the use of a specific '
                                         'experimental dataset',
                            'quote': 'As scientific datasets increasingly contain unstructured '
                                     'or semi-structured text, preprocessing and structuring '
                                     'this data are crucial challenges.'}],
               'representative_statement': 'The paper does not report the use of a specific '
                                           'experimental dataset',
               'mean_similarity': 0.8318},
              {'cluster_id': 'DAT-3',
               'dimension': 'dataset',
               'paper_ids': ['https://openalex.org/W7222723116',
                             'https://openalex.org/W7214432694',
                             'https://openalex.org/W7171313681'],
               'members': [{'paper_id': 'https://openalex.org/W7222723116',
                            'chunk_id': 'https://openalex.org/W7222723116_chunk_4',
                            'statement': 'it relies on semantic retrieval from scholarly '
                                         'databases accessed via the LangChain framework.',
                            'quote': 'Multiple LangChain based agents coordinate tasks such as '
                                     'semantic literature retrieval, LLM-driven summarization, '
                                     'embedding-based trend analysis, and automated research '
                                     'gap detection.'},
                           {'paper_id': 'https://openalex.org/W7214432694',
                            'chunk_id': 'https://openalex.org/W7214432694_chunk_1',
                            'statement': 'The paper reviews eighteen representative studies '
                                         'across five technical areas (AI‑assisted systematic '
                                         'literature review, scientific NLP, semantic search, '
                                         'RAG, agent reasoning, and knowledge graphs).',
                            'quote': 'We reviewed eighteen representative papers for five '
                                     'technical areas: AI- assisted systematic literature '
                                     'review'},
                           {'paper_id': 'https://openalex.org/W7171313681',
                            'chunk_id': 'https://openalex.org/W7171313681_chunk_28',
                            'statement': 'The study evaluates the framework on scientific '
                                         'literature repositories, employing vector databases '
                                         'to index publications and using curated scholarly '
                                         'databases, digital libraries, and citation '
                                         'repositories for cross-referencing.',
                            'quote': 'The experimental results showed that the approach of '
                                     'embedding transformer models with vector databases '
                                     'significantly outperformed the traditional keyword-based '
                                     'search methods in accurately retrieving publications '
                                     'that are conceptually related to the query, regardless '
                                     'of the use of different'}],
               'representative_statement': 'The study evaluates the framework on scientific '
                                           'literature repositories, employing vector '
                                           'databases to index publications and using curated '
                                           'scholarly databases, digital libraries, and '
                                           'citation repositories for cross-referencing.',
               'mean_similarity': 0.7417},
              {'cluster_id': 'RES-1',
               'dimension': 'results',
               'paper_ids': ['https://openalex.org/W7171304406',
                             'https://openalex.org/W7171313681',
                             'https://openalex.org/W4407028609',
                             'https://openalex.org/W7126221181'],
               'members': [{'paper_id': 'https://openalex.org/W7171304406',
                            'chunk_id': 'https://openalex.org/W7171304406_chunk_36',
                            'statement': 'The centralized knowledge graph and RAG reduce '
                                         'hallucinations and improve citation reliability, '
                                         'while multi‑agent collaboration enhances '
                                         'computational efficiency through parallel execution '
                                         'and adaptive workload balancing.',
                            'quote': 'Besides, the use of collaborative multi-agent technology '
                                     'has increased computational efficiency via intelligent '
                                     'task scheduling, parallel execution, adaptive learning, '
                                     'and constant improvement of research results.'},
                           {'paper_id': 'https://openalex.org/W7171313681',
                            'chunk_id': 'https://openalex.org/W7171313681_chunk_28',
                            'statement': 'The LLM-based knowledge extraction engine achieved '
                                         'high semantic accuracy in generating literature '
                                         'summaries, extracting key concepts, recognizing '
                                         'methodologies and experimental findings, and '
                                         'recommending relevant citations.',
                            'quote': 'The LLM-based knowledge extraction engine demonstrated '
                                     'high semantic accuracy for generating literature '
                                     'summaries, extracting key research concepts, identifying '
                                     'methods, recognizing experimental results, and '
                                     'suggesting relevant citations from scientific '
                                     'literature, in both specific and broader'},
                           {'paper_id': 'https://openalex.org/W4407028609',
                            'chunk_id': 'https://openalex.org/W4407028609_chunk_84',
                            'statement': 'The approach demonstrated reliable '
                                         'question‑answering capabilities and effective '
                                         'knowledge synthesis across multiple documents.',
                            'quote': 'datasets from a corpus of selected academic papers can '
                                     'successfully construct finetuning datasets (RQ2) that '
                                     'support question and answer downstream tasks necessary '
                                     'for executing'},
                           {'paper_id': 'https://openalex.org/W7126221181',
                            'chunk_id': 'https://openalex.org/W7126221181_chunk_6',
                            'statement': 'The paper reports that LLMs can significantly reduce '
                                         'manual effort in data extraction, literature '
                                         'summarization, and code generation, thereby '
                                         'enhancing research efficiency and enabling faster '
                                         'scientific discoveries across multiple fields.',
                            'quote': 'LLMs can automate literature surveys, information '
                                     'retrieval, and research summarization.'}],
               'representative_statement': 'The LLM-based knowledge extraction engine achieved '
                                           'high semantic accuracy in generating literature '
                                           'summaries, extracting key concepts, recognizing '
                                           'methodologies and experimental findings, and '
                                           'recommending relevant citations.',
               'mean_similarity': 0.7477},
              {'cluster_id': 'RES-2',
               'dimension': 'results',
               'paper_ids': ['https://openalex.org/W7222723116',
                             'https://openalex.org/W7160249182'],
               'members': [{'paper_id': 'https://openalex.org/W7222723116',
                            'chunk_id': 'https://openalex.org/W7222723116_chunk_3',
                            'statement': 'The system produces automated research reports, '
                                         'trend summaries, and gap detection insights.',
                            'quote': 'The suggested system offers automatic analysis of '
                                     'literature, helps find research trends, identifies gaps '
                                     'in research, and creates reports in PDF format using a '
                                     'flexible AI-powered process.'},
                           {'paper_id': 'https://openalex.org/W7160249182',
                            'chunk_id': 'https://openalex.org/W7160249182_chunk_6',
                            'statement': 'The system’s gap‑significance scoring and idea '
                                         'ranking outperform baseline approaches that rely on '
                                         'flat paper lists or manual gap statements.',
                            'quote': 'composite gap-significance formula that weights '
                                     'statistical under-representation, temporal publication '
                                     'trends, and citation-network centrality in a single '
                                     'normalised score; (iii) a three- dimensional idea '
                                     'quality metric that has been empirically validated '
                                     'against structured expert evaluation; and'}],
               'representative_statement': 'The system produces automated research reports, '
                                           'trend summaries, and gap detection insights.',
               'mean_similarity': 0.7482},
              {'cluster_id': 'RES-3',
               'dimension': 'results',
               'paper_ids': ['https://openalex.org/W7171313681',
                             'https://openalex.org/W7160249182'],
               'members': [{'paper_id': 'https://openalex.org/W7171313681',
                            'chunk_id': 'https://openalex.org/W7171313681_chunk_28',
                            'statement': 'The automated pipeline demonstrated substantial time '
                                         'savings in document organization and literature '
                                         'review tasks.',
                            'quote': 'The automated research pipeline also showed tremendous '
                                     'savings in document'},
                           {'paper_id': 'https://openalex.org/W7160249182',
                            'chunk_id': 'https://openalex.org/W7160249182_chunk_5',
                            'statement': 'Experiments demonstrate that the pipeline runs '
                                         'end‑to‑end in under two minutes, produces insights '
                                         'that would normally require days of manual review, '
                                         'and yields a three‑dimensional idea quality metric '
                                         'that has been empirically validated against '
                                         'structured expert evaluation.',
                            'quote': 'The system runs end-to-end in under two minutes, '
                                     'surfaces insights that human reviewers would typically '
                                     'take days to compile, and does so reproducibly across '
                                     'thirteen research domains without requiring '
                                     'domain-specific tuning.'}],
               'representative_statement': 'The automated pipeline demonstrated substantial '
                                           'time savings in document organization and '
                                           'literature review tasks.',
               'mean_similarity': 0.7273}],
 'embedding_model': 'BAAI/bge-small-en-v1.5',
 'grounding_threshold': 0.55,
 'similarity_threshold': 0.7,
 'min_supporting_papers': 2,
 'caveats': ["Deterministic matrix: statements come from each paper's structured analysis and "
             "are grounded only in that paper's own retrieved chunks; no LLM was used.",
             'Clusters group statements by embedding similarity (threshold 0.7); similar '
             'statements may still disagree, so clusters are evidence leads, not verified '
             'findings.',
             'Only clusters supported by at least 2 distinct papers are reported; conflicts '
             'between papers are not detected here.']}

ANALYSES = [{'analysis': {'problem': 'The paper addresses the challenge of efficiently reviewing large '
                          'volumes of scientific literature by proposing an AI‑driven research '
                          'assistant that automates literature analysis, identifies research '
                          'trends, detects research gaps, and generates structured research '
                          'reports.',
               'methodology': 'The system is built on a modular architecture using LangChain '
                              'with multiple agents. It performs semantic literature '
                              'retrieval, LLM‑driven summarization, embedding‑based trend '
                              'analysis, and graph‑based research gap detection. The user '
                              'interface is implemented with Streamlit, and outputs are saved '
                              'as PDFs and optionally stored on Amazon S3.',
               'dataset': 'The paper does not specify a particular dataset or benchmark; it '
                          'relies on semantic retrieval from scholarly databases accessed via '
                          'the LangChain framework.',
               'results': 'Experimental results show a dramatic reduction in manual effort: '
                          'literature review time drops from 4.5\u202fh to 6.2\u202fmin, '
                          'report generation time from 45\u202fmin to 18\u202fs, and overall '
                          'workflow time from 5\u202fh\u202f15\u202fmin to 9.8\u202fmin. The '
                          'system produces automated research reports, trend summaries, and '
                          'gap detection insights.',
               'limitations': 'The evidence does not detail specific limitations; however, the '
                              'authors note that existing systems lack integrated support for '
                              'trend analysis, gap detection, and cloud storage, implying that '
                              'their system may still face challenges in retrieval quality and '
                              'coverage.',
               'future_work': 'Future work will focus on integrating domain‑specific LLMs, '
                              'GraphRAG, and additional scholarly databases to further improve '
                              'retrieval quality and research insight generation.'},
  'evidence': [{'chunk_id': 'https://openalex.org/W7222723116_chunk_1',
                'paper_id': 'https://openalex.org/W7222723116',
                'text': '<Conference abbreviation> <Conference Series name> <Volume number and '
                        'Year> <DOI Number> All Open Access articles are peer-reviewed, '
                        'distributed under the Creative Commons Attribution-Non Commercial 4.0 '
                        'International LLM-Driven Research Assistant for Automated Literature '
                        'Analysis and Research Gap Detection Bijoyini Bagchi1, Kaushik '
                        'Banerjee1, Tunir De1, Trina Chakraborty1, Sayudh Chakraborty1 '
                        'Department of Computer Science and Engineering Institute of '
                        'Engineering and Management, Kolkata, India '
                        'bijoyini.bagchi@iem.edu.in, kaushik.banerjee@iem.edu.in, '
                        'tunirde2005@gmail.com, trinachakraborty14@gmail.com, '
                        'sayudh2203@gmail.com Abstract. In this project, an AI-driven Research '
                        'Assistant is introduced to support researchers and students in '
                        'finding relevant information related to their field of interest, '
                        'identifying research trends, and creating research reports '
                        'systematically. To accomplish this task, LLM-driven multi-agents are '
                        'used along with semantic searching and automated research gap '
                        'detection by using the LangChain framework. The research process '
                        'includes literature review, finding research gaps, and preparing '
                        'proposals. For user interaction, the app is designed with Streamlit '
                        'to provide users with the ability to view their automatically created '
                        'research report summaries, trends, and research direction '
                        'suggestions. Moreover, the reports are created in PDF form and can be '
                        'saved with cloud storage solutions such as Amazon S3.',
                'distance': 0.13681036233901978},
               {'chunk_id': 'https://openalex.org/W7222723116_chunk_2',
                'paper_id': 'https://openalex.org/W7222723116',
                'text': 'automatically created research report summaries, trends, and research '
                        'direction suggestions. Moreover, the reports are created in PDF form '
                        'and can be saved with cloud storage solutions such as Amazon S3. '
                        'Keywords. Large Language Models (LLMs), Semantic Retrieval, Research '
                        'Gap Detection, Multi-Agent Systems, Academic Literature Analysis. 1. '
                        'INTRODUCTION In the past few years, the fast increase in scientific '
                        'papers has made it harder for researchers to effectively look through '
                        'current literature and find important research paths. The usual way '
                        'of doing a literature review takes a lot of time and work because '
                        'people have to look through many research papers by hand, find the '
                        'main ideas, and figure out where there might be gaps in the research. '
                        'Research helpers powered by AI can quickly look through a lot of '
                        'academic papers and provide organized insights. This project suggests '
                        'an AI-driven Research Assistant that helps users explore academic '
                        'papers using an engaging interface built with Streamlit. The system '
                        'uses workflows powered by LangChain with multiple agents to '
                        'automatically summarize literature, identify research trends, find '
                        'gaps in research, and generate reports based on',
                'distance': 0.155767560005188},
               {'chunk_id': 'https://openalex.org/W7222723116_chunk_4',
                'paper_id': 'https://openalex.org/W7222723116',
                'text': 'a modular architecture where different components work sequentially '
                        'to analyze the user’s research topic and produce structured outputs. '
                        '3.1. User Input and Query Processing The workflow begins when the '
                        'user enters a research topic through an interactive web interface '
                        'built using Streamlit. The system captures the user’s query and '
                        'performs pre- processing before further analysis. 3.2. AI Agent '
                        'Pipeline and Task Coordination The main intelligence of the system is '
                        'implemented using LangChain. Multiple LangChain based agents '
                        'coordinate tasks such as semantic literature retrieval, LLM-driven '
                        'summarization, embedding-based trend analysis, and automated research '
                        'gap detection. 3.3. Research Trend Identification The system performs '
                        'semantic clustering of retrieved publications using sentence '
                        'embeddings to identify emerging research trends. By analyzing topic '
                        'similarity and keyword distributions, it highlights evolving research '
                        'areas and assists users in understanding recent developments within '
                        'the chosen domain. 3.4. Research Gap Detection The system applies '
                        'embedding similarity and semantic analysis to identify underexplored '
                        'research areas and unresolved problems within the analyzed '
                        'literature. The graph-based similarity analysis helps identify '
                        'clusters of closely related research papers and detect underexplored '
                        'regions with limited interconnections, supporting',
                'distance': 0.18420398235321045},
               {'chunk_id': 'https://openalex.org/W7222723116_chunk_3',
                'paper_id': 'https://openalex.org/W7222723116',
                'text': 'interface built with Streamlit. The system uses workflows powered by '
                        'LangChain with multiple agents to automatically summarize literature, '
                        'identify research trends, find gaps in research, and generate reports '
                        'based on PDFs. The suggested system offers automatic analysis of '
                        'literature, helps find research trends, identifies gaps in research, '
                        'and creates reports in PDF format using a flexible AI-powered '
                        'process. 2 2. LITERATURE REVIEW Recent advancements in LLMs and NLP '
                        'have significantly improved automated literature analysis and '
                        'research summarization systems. Existing studies mainly focus on '
                        'scientific summarization and manuscript generation using AI-driven '
                        'frameworks and LangChain based workflows [1–3]. Existing systems '
                        'primarily focus on isolated tasks such as document summarization or '
                        'recommendation systems. However, most lack integrated support for '
                        'research trend analysis, research gap detection, automated report '
                        'generation, and cloud- based storage. The proposed system addresses '
                        'these limitations through a unified AI-driven research assistance '
                        'framework. 3. METHODOLOGY The proposed AI Research Assistant follows '
                        'a modular architecture where different components work sequentially '
                        'to analyze the user’s research topic and produce structured outputs. '
                        '3.1. User Input and Query Processing The workflow begins when the '
                        'user',
                'distance': 0.19069808721542358},
               {'chunk_id': 'https://openalex.org/W7222723116_chunk_7',
                'paper_id': 'https://openalex.org/W7222723116',
                'text': 'method generation and grant proposal writing. Additionally, the '
                        'system generated comprehensive PDF reports and supported cloud-based '
                        'storage through AWS S3, significantly reducing manual effort while '
                        'improving the speed and quality of academic research planning. Metric '
                        'Manual Process Proposed System Literature Review Time 4.5 hours 6.2 '
                        'min Average Retrieval Time N/A 2.3 s Report Generation Time 45 min 18 '
                        's End-to-End Workflow Time 5 hours 15 min 9.8 min Table 1: '
                        'Experimental Performance Comparison of Manual and Proposed Research '
                        'Work flow 7. CONCLUSION AND FUTURE SCOPE This work presents an '
                        'AI-powered academic research assistant capable of automated '
                        'literature analysis, research trend identification, and research gap '
                        'detection using LLM- based workflows. Based on experimental '
                        'observations, it has been established that the system helps to '
                        'decrease the amount of manual research work performed by researchers '
                        'while at the same time producing efficient research insights and '
                        'reports. This framework highlights the practical viability of '
                        'research automation using LLMs for use in academic assistance '
                        'systems. Future work will focus on integrating domain-specific LLMs, '
                        'GraphRAG, and additional scholarly databases to further improve '
                        'retrieval quality and research',
                'distance': 0.19911330938339233}]},
 {'analysis': {'problem': 'The paper addresses the challenge of automating and improving the '
                          'literature review process by proposing an AI‑driven system, IRIS, '
                          'that integrates paper retrieval, scientific information extraction, '
                          'summarization, citation‑aware synthesis, knowledge‑graph '
                          'construction, research‑gap detection, and structured report '
                          'generation.',
               'methodology': 'IRIS employs a multi‑agent workflow where domain‑specialized '
                              'autonomous agents collaborate on retrieval, analysis, '
                              'summarization, gap detection, and report generation. The system '
                              'leverages retrieval‑augmented generation (RAG), long‑context '
                              'processing, step‑by‑step reasoning, and knowledge‑graph '
                              'construction techniques to support these tasks.',
               'dataset': 'The paper reviews eighteen representative studies across five '
                          'technical areas (AI‑assisted systematic literature review, '
                          'scientific NLP, semantic search, RAG, agent reasoning, and '
                          'knowledge graphs). No specific experimental dataset is introduced; '
                          'the work is a comprehensive review rather than an empirical study.',
               'results': 'The review identifies that existing approaches excel at individual '
                          'sub‑problems but struggle with integration, provenance detection, '
                          'long‑context handling, explainability, and systematic gap '
                          'detection. IRIS is proposed as a human‑in‑the‑loop system that aims '
                          'to overcome these limitations by combining the strengths of the '
                          'surveyed methods.',
               'limitations': 'The evidence indicates that agent coordination and evaluation '
                              'remain challenging, automatic knowledge‑graph construction can '
                              'introduce incorrect or duplicate relationships, and '
                              'long‑context processing still limits information utilization. '
                              'The paper does not provide empirical validation of IRIS, so its '
                              'effectiveness is theoretical.',
               'future_work': 'Future directions include developing and evaluating the '
                              'multi‑agent IRIS architecture, improving agent coordination and '
                              'evaluation, refining automatic knowledge‑graph extraction to '
                              'reduce errors, and integrating systematic review guidelines '
                              '(e.g., PRISMA 2020) to enhance transparency and traceability.'},
  'evidence': [{'chunk_id': 'https://openalex.org/W7214432694_chunk_2',
                'paper_id': 'https://openalex.org/W7214432694',
                'text': 'combine paper retrieval, scientific information extraction, '
                        'summarization, citation-aware synthesis, knowledge-graph building, '
                        'research-gap detection and structured report generation.We reviewed '
                        'eighteen representative papers for five technical areas: AI- assisted '
                        'systematic literature review (SLR), Scientific natural language '
                        'processing (NLP), semantic search, RAG, agent reasoning and KN. We '
                        'observed that existing approaches typically handle individual '
                        'subproblems well but struggle with integration, provenance detection, '
                        'long-context processing, explainability, systematic gap detection, '
                        'etc. With these observations, we propose a conceptual IRIS framework. '
                        'It consists of several domain- specialized autonomous agents agenting '
                        'over a citation aware scholarly corpus, and a researchy knowledge '
                        'graph. we summarize some evidence as to why IRIS is better considered '
                        'as a human-in-the- loop system rather than a fully automated system '
                        'of retrieving literature reviews, consistent with common '
                        'understandings in research. I. INTRODUCTION A literature review '
                        'combines the work that has been done on a topic to define the state '
                        'of the knowledge in a research area. The traditional approach for a '
                        'researcher browses the several scholarships sources looking for the '
                        'relevant papers, reads the abstract and body texts of the papers, '
                        'records the',
                'distance': 0.14235562086105347},
               {'chunk_id': 'https://openalex.org/W7214432694_chunk_14',
                'paper_id': 'https://openalex.org/W7214432694',
                'text': 'interesting, reflection, tool use, and multi-agent collaboration into '
                        'RAG pipelines. This is extremely relevant to the IRIS design because '
                        'diverse agents can be employed to specialize — surprisingly — and '
                        'this is key — — in retrieval, paper analysis, citation checking, '
                        'knowledge graph construction, and gap detection Fairly '
                        'straightforward, right? Agent specialization on top of that opens up '
                        'opportunities for intermediate validation at various stages.The way I '
                        'see it, The project seminar specifically proposes an agentic multi- '
                        'Al workflow in. Multiple agents collaborate on retrieval, analysis, '
                        'summarization, gap detection, and report generation [1]. The key '
                        'analysis question isn’t whether agents can generate text. Looking at '
                        'this, whether agent coordination increases evidence coverage and '
                        'decreases unsubstantiated claims. Worth remembering. E. Knowledge '
                        'Graphs and Graph-Based Retrieval Hogan et al. [9] present an overview '
                        'of knowledge graphs, including graph data models, knowledge '
                        'extraction, quality assessment, and applications Makes you think. Ji '
                        'et al. [10] survey representation learning, knowledge acquisition, '
                        'completion, temporal graphs, and knowledge-aware applications. I find '
                        'that Together, these works establish the basic principles for '
                        'representing scholarly information in the form of',
                'distance': 0.167619526386261},
               {'chunk_id': 'https://openalex.org/W7214432694_chunk_1',
                'paper_id': 'https://openalex.org/W7214432694',
                'text': 'Page 1 A COMPREHENSIVE REVIEW ON AI- DRIVEN LITERATURE REVIEW, '
                        'RETRIEVAL-AUGMENTED GENERATION, KNOWLEDGE GRAPHS, AND RESEARCH GAP '
                        'DETECTION Under the Guidance of Prof Monika S. Shirbhate ,Prof Aparna '
                        'R. Khairkar Riya Tatode (57), Anjali Raut (45), Manaswi Gawande (23), '
                        'Trupti Bhongle (05) Department of Information Technology, Prof. Ram '
                        'Meghe College of Engineering and Management, Badnera, Amravati, '
                        'Maharashtra, India | Academic Session 2026–27 I ABSTRACT Literature '
                        'review is a core task in academic and industry research, but the '
                        'usual process takes major effort to find relevant papers, screen '
                        'them, pull out information, compare results, trace citations, combine '
                        'themes and spot research directions that remain unexplored. Recent '
                        'progress in artificial intelligence, natural language processing, '
                        'large language models, retrieval-augmented generation, knowledge '
                        'graphs and autonomous agent workflows creates a chance to support '
                        'this process. This paper reviews research related to IRIS '
                        '(Intelligent Research & Information System), a proposed AI-based '
                        'research assistant which aims to combine paper retrieval, scientific '
                        'information extraction, summarization, citation-aware synthesis, '
                        'knowledge-graph building, research-gap detection and structured '
                        'report generation.We reviewed eighteen representative papers for five '
                        'technical areas: AI- assisted systematic literature review',
                'distance': 0.17028331756591797},
               {'chunk_id': 'https://openalex.org/W7214432694_chunk_19',
                'paper_id': 'https://openalex.org/W7214432694',
                'text': 'significant effort Liu et al. [12] Long context Examines how models '
                        'handle information at different positions in long inputs Supports the '
                        'need for hierarchical processing in IRIS Long-context processing can '
                        'still reduce information utilization Wei et al. [13] LLM reasoning '
                        'Shows how step-by- step reasoning can support complex tasks Provides '
                        'a basis for breaking IRIS tasks into smaller stages Not specifically '
                        'designed for literature-review applications Singh et al. [14] Agentic '
                        'RAG Discusses planning, reflection, tool use, and agent collaboration '
                        'Supports the use of multiple specialized agents in IRIS Agent '
                        'coordination and evaluation remain challenging KG construction survey '
                        '[15] Automatic KG Reviews approaches for acquiring and refining '
                        'knowledge- graph information Provides guidance for building the IRIS '
                        'research graph Automatic extraction may introduce incorrect or '
                        'duplicate relationships PRISMA 2020 [16] Systematic review reporting '
                        'Provides guidelines for making the review process transparent Can '
                        'help IRIS maintain a more organized and traceable review process It '
                        'is a reporting guideline rather than an AI implementation LLM '
                        'scientific assistance survey [17] LLMs in research Reviews how LLMs '
                        'can support different scientific activities Helps identify possible '
                        'benefits',
                'distance': 0.18298161029815674},
               {'chunk_id': 'https://openalex.org/W7214432694_chunk_4',
                'paper_id': 'https://openalex.org/W7214432694',
                'text': 'the automated retrieval, extraction, summarization, and trend '
                        'identification [2]. The human-centered approaches have also '
                        'demonstrated that the language models can help the researchers in '
                        'developing the literature-review narratives rather than generating '
                        'the summaries [3]. A useful review synthesizes the information while '
                        'preserving the relations between the works, distinguishing the '
                        'evidence and interpretation, highlighting the disagreements, and '
                        'making the trail from the claims backed to the sources clear. The '
                        'scientific NLP models such as SciBERT [4] and document-level '
                        'representations such as SPECTER [5] provide the foundation for '
                        'understanding and retrieving the scholarly documents. The RAG [6], '
                        'dense passage retrieval [7], and self-reflective RAG [8] provide the '
                        'mechanisms for the grounding of the generated outputs in the '
                        'retrieved evidence. The Knowledge graphs [9], [10] and GraphRAG [11] '
                        'provide the structured approaches to representing and answering the '
                        'corpus-level questions. IRIS is proposed to combine all these ideas '
                        'in a workflow. The project seminar describes an architecture with the '
                        'research-paper retrieval through the scholarly APIs, NLP and '
                        'information extraction, AI summarization and paper analysis, '
                        'research-gap and thematic analysis, knowledge-graph generation, and '
                        'citation-aware report generation',
                'distance': 0.19572222232818604}]},
 {'analysis': {'problem': 'The paper addresses the challenge of automating the entire '
                          'scientific research cycle by integrating multiple AI technologies '
                          'to reduce manual effort and improve efficiency in data‑intensive, '
                          'computationally complex research environments.',
               'methodology': 'An Agentic AI Framework that combines Large Language Models '
                              '(LLMs), Retrieval‑Augmented Generation (RAG), collaborative '
                              'multi‑agent systems, semantic knowledge graphs, reinforcement '
                              'learning, explainable AI, and evidence‑based decision‑making to '
                              'autonomously manage literature review, knowledge extraction, '
                              'hypothesis generation, experiment planning, statistical '
                              'analysis, citation verification, manuscript preparation, and '
                              'research validation.',
               'dataset': 'The evaluation used a collection of 4 publications, conference '
                          'proceedings, patents, technical reports, experimental repositories, '
                          'simulation datasets, institutional databases, and open‑access '
                          'research platforms to test literature retrieval, hypothesis '
                          'generation, citation verification, and other metrics.',
               'results': 'Experimental results show the framework consistently achieves over '
                          '90% accuracy on most evaluation metrics, including literature '
                          'retrieval accuracy, hypothesis quality, citation verification, '
                          'workflow automation, knowledge discovery, manuscript preparation, '
                          'explainability, and overall research productivity. The centralized '
                          'knowledge graph and RAG reduce hallucinations and improve citation '
                          'reliability, while multi‑agent collaboration enhances computational '
                          'efficiency through parallel execution and adaptive workload '
                          'balancing.',
               'limitations': 'The evidence notes several limitations: difficulty integrating '
                              'heterogeneous scientific knowledge due to varying metadata '
                              'standards, terminologies, document structures, and citation '
                              'formats; challenges in maintaining long‑term context across '
                              'extended research initiatives leading to disjointed reasoning; '
                              'and the current lack of effective multi‑agent coordination '
                              'mechanisms for complex scientific goals.',
               'future_work': 'The paper identifies the need to develop scalable, reliable, '
                              'and transparent Agentic AI frameworks that include common '
                              'coordination strategies, adaptive knowledge management, and '
                              'improved multi‑agent orchestration to overcome the highlighted '
                              'integration and context‑maintenance challenges.'},
  'evidence': [{'chunk_id': 'https://openalex.org/W7171304406_chunk_1',
                'paper_id': 'https://openalex.org/W7171304406',
                'text': 'Copyright © The Author(s) This article is distributed under a '
                        'Creative Commons Attribution-Share A like 4.0 International License. '
                        '1 International Journal of Emerging Trends in Multidisciplinary '
                        'Research Vol. 7, No. 2, 2024 Doi: '
                        '10.67228/30715636/IJETMR-2024PII6Q2K PP. 01-15 World Comet Research '
                        'Group Original Article Agentic AI Framework for Autonomous Scientific '
                        'Research Assistance Anatoly Kitov1, Mikhail Kartsev2 1Colonel & '
                        'Computer Scientist, Soviet Academy of Sciences, Russia. 2Chief '
                        'Designer, Research Institute of Computer Systems, Russia. Received: '
                        '10-07-2024 Revised: 10-26-2024 Accepted: 11-01-2024 Published: '
                        '11-03-2024 ABSTRACT The rapid expansion of scientific publications, '
                        'experimental data, and digital repositories has made research '
                        'increasingly data-intensive and computationally complex. Traditional '
                        'research tools, such as search engines and digital libraries, require '
                        'substantial manual effort and provide limited support for '
                        'comprehensive research workflows. This paper proposes an Agentic AI '
                        'Framework for Autonomous Scientific Research Assistance that '
                        'integrates Large Language Models (LLMs), Retrieval-Augmented '
                        'Generation (RAG), and collaborative multi- agent systems to automate '
                        'key research activities. The framework coordinates specialized agents '
                        'for literature review, knowledge graph construction, hypothesis '
                        'generation, experiment planning, data analysis, citation management, '
                        'research validation, and manuscript generation. The architecture '
                        'combines domain-specific',
                'distance': 0.13616156578063965},
               {'chunk_id': 'https://openalex.org/W7171304406_chunk_12',
                'paper_id': 'https://openalex.org/W7171304406',
                'text': 'of collaborative agents is designed to enable multiple specialised '
                        'agents to coordinate their actions in the task of literature review, '
                        'experiment design, statistical analysis and result interpretation, '
                        'which greatly enhances the efficiency of research and decreases human '
                        'labour. Leveraging these impressive advances, much of the current AI '
                        'in use by Agentics is still very much domain-specific and lacks '
                        'generalised orchestration techniques that enable management of larger '
                        'scientific workflows composed of multiple steps and performed in '
                        'different scientific environments. They have yet to be widely adopted '
                        'in autonomous scientific discovery due to a lack of common '
                        'coordination strategies, understanding processes and adaptive '
                        'knowledge management, necessitating the need for more scalable, '
                        'reliable and transparent Agentic AI frameworks. 2.2. Large Language '
                        'Models in Research Assistance Certain Large Language Models (LLMs) '
                        'have proven to be one of the most impactful technologies in the world '
                        'of AI-enabled scientific research because of their exceptional '
                        'capabilities in comprehending, generating and reasoning about '
                        'intricate scientific text. The models are based on transformer '
                        'algorithms and they can successfully process large numbers of '
                        'scholarly articles, provide summaries to technical documents,',
                'distance': 0.15159279108047485},
               {'chunk_id': 'https://openalex.org/W7171304406_chunk_33',
                'paper_id': 'https://openalex.org/W7171304406',
                'text': 'the assistance provided by AI to scientific research work compared to '
                        'conventional AI models. Unlike traditional large language model '
                        '(LLM)-based research assistants, which are mostly aimed at generating '
                        'conversational responses or text, the presented Agentic AI framework '
                        'implements autonomous management of the entire research cycle through '
                        'intelligent agent collaborations. The results show that the framework '
                        'constantly produces values above 90% for most of the evaluation '
                        'metrics, including literature retrieval accuracy, quality of '
                        'hypothesis generation, citation verification, workflow automation, '
                        'knowledge discovery, manuscript preparation, explainability, and '
                        'research productivity. The improvements can be attributed to the '
                        'combination of evidence-based reasoning and semantic knowledge '
                        'representation along with dynamic task orchestration. Specifically, '
                        'the introduction of a centralized knowledge graph made it possible to '
                        'discover implicit links between scientific objects and make '
                        'literature analysis more efficient and enable interdisciplinary '
                        'knowledge integration. Additionally, the application of '
                        'Retrieval-Augmented Generation helped to ground recommendations and '
                        'generated content in verified scholarly evidence, thus eliminating '
                        'hallucinations and improving citation reliability. The collaborative '
                        'communication of autonomous agents increased computational efficiency '
                        'by means of parallel task execution, adaptive workload balancing, and '
                        'continuous improvement',
                'distance': 0.1546797752380371},
               {'chunk_id': 'https://openalex.org/W7171304406_chunk_36',
                'paper_id': 'https://openalex.org/W7171304406',
                'text': 'end-to-end scientific research. This paper presents an Agentic AI '
                        'Framework for Autonomous Scientific Research Assistance that '
                        'integrates autonomous planning, collaborative multi-agent '
                        'intelligence, Retrieval-Augmented Generation (RAG), semantic '
                        'knowledge graphs, reinforcement learning, explainable artificial '
                        'intelligence, and evidence-based decision-making into a single '
                        'framework. The framework makes it possible to collaborate intelligent '
                        'agents for execution of all stages of the research cycle, including '
                        'literature retrieval, knowledge extraction, research gap '
                        'identification, hypothesis generation, methodology design, experiment '
                        'planning, statistical analysis, citation verification, manuscript '
                        'preparation, and research validation. Experimental evaluation shows '
                        'that the proposed framework outperforms conventional AI-assisted '
                        'research systems in literature retrieval accuracy, workflow '
                        'automation, quality of hypothesis generation, citation verification, '
                        'explainability, manuscript preparation, and research productivity. '
                        'Integration of semantic knowledge graphs has positively affected '
                        'contextual awareness and knowledge discovery, while the '
                        'implementation of RAG made it possible to eliminate hallucinations by '
                        'grounding AI-generated recommendations and responses in scholarly '
                        'evidence. Besides, the use of collaborative multi-agent technology '
                        'has increased computational efficiency via intelligent task '
                        'scheduling, parallel execution, adaptive learning, and constant '
                        'improvement of research results. Finally, explainable reasoning, '
                        'confidence estimation, and provenance tracking have improved the',
                'distance': 0.16186821460723877},
               {'chunk_id': 'https://openalex.org/W7171304406_chunk_8',
                'paper_id': 'https://openalex.org/W7171304406',
                'text': 'all of the new developments in Artificial Intelligence (AI), Large '
                        'Language Models (LLMs) and Retrieval-Augmented Generation (RAG), '
                        'there are still a number of technological, methodological and '
                        'operational hurdles to overcome in order to enable fully autonomous '
                        'scientific research systems. The challenge of integrating with '
                        'heterogeneous scientific knowledge from scholarly ISSN: 3071-5636 PP: '
                        '01-15. ---------------- Agentic AI Framework for Autonomous '
                        'Scientific Research Assistance Anatoly Kitov & Mikhail Kartsev \uf072 '
                        '4 publications, conference proceedings, patents, technical reports, '
                        'experimental repositories, simulation datasets, institutional '
                        'databases and open-access research platforms are one of the most '
                        'significant challenges. The information sources may be using '
                        'different metadata standards, different domain- specific '
                        'terminologies, different document structure and different citation '
                        'formats, leading to high complexity in semantic integration and '
                        'unified knowledge representation. Even the current AI- powered '
                        'research assistance tools are challenged in maintaining long-term '
                        'context over longer research initiatives, resulting in disjointed '
                        'reasoning, varied recommendations, and not sustaining continuity '
                        'across the entire research process. One of the other significant '
                        'weaknesses is the absence of an effective mechanism of multi-agent '
                        'coordination that can break down the complex scientific goals',
                'distance': 0.16631758213043213}]},
 {'analysis': {'problem': 'The paper addresses the need for a framework that supports '
                          'iterative refinement and problem reformulation across successive '
                          'scientific research cycles, enabling continuous and open-ended '
                          'scientific exploration.',
               'methodology': 'It proposes a heterogeneous multi-agent framework mirroring '
                              'real-world research laboratories, where diverse LLM-based '
                              'agents operate under role-specific objectives and collaborate '
                              'through a multi-round feedback loop with a bio-inspired '
                              'evolution mechanism (selection, crossover, mutation).',
               'dataset': 'The framework is grounded on datasets derived from real scientists, '
                          'enabling authentic simulation of collaborative scientific '
                          'workflows.',
               'results': 'In real-world research topic experiments, the system (EvoSci) '
                          'significantly outperforms strong baselines in LLM-based structured '
                          'peer-review and comparative ranking evaluations, achieving the '
                          'highest overall peer-review score (ICLR 4.90) and top ranking '
                          '(Top‑10 = 54).',
               'limitations': 'The paper notes that establishing objective and high-quality '
                              'evaluation mechanisms is challenging, as it is essential for '
                              'LLM-based agents to better assess their own reasoning and '
                              'outputs for effective self-improvement and sustained '
                              'innovation.',
               'future_work': 'No explicit future research directions are provided in the '
                              'supplied evidence.'},
  'evidence': [{'chunk_id': 'https://openalex.org/W7166808050_chunk_6',
                'paper_id': 'https://openalex.org/W7166808050',
                'text': 'as a problem-oriented process, in which research problems are '
                        'dynamically generated and pro- gressively refined through a '
                        'multi-agent collab- oration loop. • We construct a heterogeneous '
                        'multi-agent framework that mirrors real-world research laboratories, '
                        'where diverse agents operate un- der role-specific objectives. The '
                        'framework is grounded on datasets derived from real sci- entists, '
                        'enabling more authentic simulation of collaborative scientific '
                        'workflows. • We implement a multi-round feedback with bio-inspired '
                        'evolution mechanism (selection, crossover, mutation) to enable '
                        'continuous and open-ended scientific exploration. 2 Related Work Our '
                        'work is related to both AI-driven scientific discovery and '
                        'multi-agent systems. We briefly review these two topics within the '
                        'scope of LLM and the constraint of space. 2.1 AI for Scientific '
                        'Discovery Recent advances in LLMs have enabled AI sys- tems to '
                        'participate more deeply in scientific dis- covery (Zheng et al., '
                        '2025; Xiong et al., 2026). Early systems primarily assist with '
                        'literature min- ing (Smalheiser and Swanson, 1998; Hristovski et al., '
                        '2005) and experiment design (King et al., 2009; Tian et al., 2021), '
                        'while recent developments aim at more comprehensive support for the '
                        'scientific process.',
                'distance': 0.15300261974334717},
               {'chunk_id': 'https://openalex.org/W7166808050_chunk_8',
                'paper_id': 'https://openalex.org/W7166808050',
                'text': 'from topic formulation and literature exploration to hypothesis '
                        'generation, experiment simulation, paper writing, and even peer '
                        'review. However, existing approaches primarily focus on conducting a '
                        'single round of research under a fixed initial topic, overlooking the '
                        'evolutionary and cyclic nature of scientific discovery. This gap '
                        'highlights the need for frameworks that support iterative refinement '
                        'and problem reformulation across successive research cycles. 2.2 '
                        'Collaboration in Multi-Agent Systems Recent work has increasingly '
                        'turned to multi-agent systems (MAS) as a way to mitigate the '
                        'limitations of single LLMs, such as hallucination (Huang et al., '
                        '2025), weak long-horizon reasoning (Ferrag et al., 2025), and stale '
                        'knowledge (Zhang et al., 2023). MAS build on the paradigm of agentic '
                        'AI (Durante et al., 2024), organizing multiple LLM-based agents with '
                        'specialized roles and shared goals. Through coordinated planning, '
                        'division of labor, and mu- tual evaluation, these systems aim to '
                        'achieve more reliable and scalable cognitive performance than any '
                        'individual model (Zhou et al., 2024a; Li et al., 2025). A growing '
                        'line of studies applies MAS to scien- tific discovery, demonstrating '
                        'their potential in iter- ative and multi-step',
                'distance': 0.1601957082748413},
               {'chunk_id': 'https://openalex.org/W7166808050_chunk_2',
                'paper_id': 'https://openalex.org/W7166808050',
                'text': 'real-world research topics demonstrate that EvoSci significantly '
                        'outperforms strong baselines in LLM-based structured peer-review and '
                        'comparative rank- ing evaluations, achieving the highest overall '
                        'peer-review score (ICLR 4.90) and top ranking (Top-10 = 54). These '
                        'results suggest its supe- riority in both scientific idea generation '
                        'and continuous discovery. 1 Introduction With the breakthrough '
                        'development in knowledge representation (Pan et al., 2024), logical '
                        'reasoning (Ke et al., 2025; Xu et al., 2025), and complex multimodal '
                        'information integration (Han et al., 2025), LLMs are gradually '
                        'reshaping the paradigm of scientific research (Buehler, 2024). '
                        'Significant progress has already been made in traditional AI- powered '
                        'domains such as mathematical reasoning and theorem proving (Trinh et '
                        'al., 2024; Zhang and Xiong, 2025; Liu et al., 2025), automatic code '
                        'gen- eration (Li et al., 2023; Ren et al., 2023; He et al., 2024; '
                        'Yang et al., 2025), and complex data analysis (Sui et al., 2024). '
                        'Building on these successes, re- searchers are increasingly exploring '
                        'the potential of † Corresponding authors. LLMs across broader '
                        'scientific workflows, includ- ing idea and hypothesis generation from '
                        'large-scale scientific corpora (Kulkarni et',
                'distance': 0.1627855896949768},
               {'chunk_id': 'https://openalex.org/W7166808050_chunk_22',
                'paper_id': 'https://openalex.org/W7166808050',
                'text': 'challenging open research topics were selected from the task settings '
                        'introduced in AI Scientist (Lu et al., 2024). De- tailed task '
                        'settings and corresponding experimental prompts are provided in '
                        'Appendices B and F. 4.1 Evaluation Methodologies To comprehensively '
                        'assess the effectiveness and creativity of our multi-agent scientific '
                        'system, we adopted both qualitative and quantitative evalua- tions, '
                        'combining an expert-simulated review mech- anism with a '
                        'tournament-style idea ranking pro- cedure. In addition, we conducted '
                        'an additional experiment to further validate the effectiveness and '
                        'stability of the proposed meta-review mechanism (see Appendix E). '
                        'Multi-Reviewer + Meta-Reviewer Mechanism. Inspired by the AI '
                        'Scientist evaluation framework (Lu et al., 2024), we designed a '
                        'structured LLM- driven peer-review workflow that simulates aca- demic '
                        'conference reviewing. Reviewer agents in- dependently assess '
                        'generated ideas using prompts aligned with ICLR and NeurIPS review '
                        'templates, each incorporating a reflection mechanism to re- fine '
                        'their evaluations. A meta-reviewer agent then aggregates individual '
                        'reviews into a unified meta- review, enabling interpretable, '
                        'reproducible, and academically representative evaluation. '
                        'Tournament-Style Idea Ranking. In addition, we implemented a '
                        'comparative ranking procedure to evaluate idea quality.',
                'distance': 0.17635369300842285},
               {'chunk_id': 'https://openalex.org/W7166808050_chunk_38',
                'paper_id': 'https://openalex.org/W7166808050',
                'text': 'A key chal- lenge in achieving such continuous evolution lies in '
                        'establishing more objective and high-quality evalu- ation mechanisms '
                        'that allow LLM-based agents to better assess their own reasoning and '
                        'outputs, which is essential for truly effective self-improvement and '
                        'sustained innovation. Acknowledgments The present research was '
                        'supported by the National Key Research and Development Program of '
                        'China (Grant No. 2024YFE0203000). We also acknowl- edge support from '
                        'the State Key Laboratory of Ti- betan Intelligence (Grant No. '
                        '2025-ZJ-J08) and the Postdoctoral Fellowship Program of CPSF (Grant '
                        'No. GZC20251075). We thank the anonymous reviewers for their '
                        'insightful comments. References Jinheon Baek, Sujay Kumar Jauhar, '
                        'Silviu Cucerzan, and Sung Ju Hwang. 2024. Researchagent: Iter- ative '
                        'research idea generation over scientific liter- ature with large '
                        'language models. arXiv preprint arXiv:2404.07738. Markus J Buehler. '
                        '2024. Accelerating scientific dis- covery with generative knowledge '
                        'extraction, graph- based representation, and multimodal intelligent '
                        'graph reasoning. Machine Learning: Science and Technology, '
                        '5(3):035083. Gary Charness, Brian Jabarian, and John A. List. 2025. '
                        'The next generation of experimental research with LLMs. Nature Human '
                        'Behaviour, 9(5):833–835. Saaketh Desai, Sadhvikas Addamane, Jeffrey Y '
                        'Tsao,',
                'distance': 0.18020939826965332}]},
 {'analysis': {'problem': 'The paper investigates how Large Language Models (LLMs) can '
                          'transform scientific computing and research by automating data '
                          'preprocessing, literature review, knowledge integration, problem '
                          'solving, and scientific programming.',
               'methodology': 'It surveys and discusses the application of LLMs—particularly '
                              'through techniques such as Named Entity Recognition (NER), text '
                              'summarization, knowledge‑graph construction, and code '
                              'generation—to convert unstructured scientific text into '
                              'structured data, accelerate literature reviews, and support '
                              'simulation verification and code generation.',
               'dataset': 'The paper does not report the use of a specific experimental '
                          'dataset; it focuses on the general use of LLMs across scientific '
                          'domains such as computational chemistry, bioinformatics, and '
                          'climate science.',
               'results': 'The paper reports that LLMs can significantly reduce manual effort '
                          'in data extraction, literature summarization, and code generation, '
                          'thereby enhancing research efficiency and enabling faster '
                          'scientific discoveries across multiple fields.',
               'limitations': 'The paper identifies key limitations: (1) hallucinations '
                              'produced by LLMs, which can lead to inaccurate or fabricated '
                              'outputs; (2) data privacy concerns when handling sensitive '
                              'scientific information; and (3) the need for rigorous '
                              'validation of LLM outputs to ensure reliability.',
               'future_work': 'Future directions mentioned include: (1) developing robust data '
                              'security and privacy solutions for LLM usage; (2) improving '
                              'validation protocols to mitigate hallucinations; and (3) '
                              'advancing LLM capabilities to support interdisciplinary '
                              'research and cross‑domain knowledge synthesis.'},
  'evidence': [{'chunk_id': 'https://openalex.org/W7126221181_chunk_7',
                'paper_id': 'https://openalex.org/W7126221181',
                'text': 'and research summarization. For instance, LLMs are being used to '
                        'aggregate clinical trial records, research publications, and patents '
                        'to identify drug candidates and emerging research hotspots; Employing '
                        'LLMs to summarize existing research gaps and guide future research '
                        'directions [5]. Large language models can rapidly scan vast '
                        'scientific literature, extracting key information such as '
                        'methodologies, findings, and conclusions. This liberates scientists '
                        'from literature queries, enabling higher-level thinking. '
                        'Simultaneously, LLMs can automatically generate text summaries for '
                        'large volumes of papers, providing status updates or descriptive '
                        'overviews. This helps scientists achieve comprehensive knowledge '
                        'updates with minimal manual effort [5]. 265 Another potential '
                        'application involves automatically constructing knowledge graphs from '
                        'text— a powerful tool for managing and integrating complex scientific '
                        'domain information. LLMs can process literature into visual knowledge '
                        'graphs (e.g., depicting relationships between diseases, genes, and '
                        'molecules in biomedicine, or between compounds and reactions in '
                        'chemistry). Knowledge graphs themselves serve as potent scientific '
                        'discovery tools, enabling researchers to uncover patterns, trends, '
                        'and correlations across fragmented data sources [5]. 2.3. Problem '
                        'Solving and Theoretical Insights LLM applications have extended '
                        'beyond data preprocessing and literature',
                'distance': 0.18128323554992676},
               {'chunk_id': 'https://openalex.org/W7126221181_chunk_1',
                'paper_id': 'https://openalex.org/W7126221181',
                'text': '263 Collaborative Integration of Large Language Models and Computer '
                        'Algebra Systems for Simulation Verification and Code Generation Xuhui '
                        'Shi School of Computing, University of Leeds, Leeds, United Kingdom '
                        'sc23x2s@leeds.uk.com Abstract. This paper explores the transformative '
                        'role of Large Language Models (LLMs) in scientific computing and '
                        'research. As scientific datasets increasingly contain unstructured or '
                        'semi-structured text, preprocessing and structuring this data are '
                        'crucial challenges. LLMs have shown great potential in automating '
                        'data extraction, transforming raw text into structured formats for '
                        'computational models. By utilizing techniques such as Named Entity '
                        'Recognition (NER), LLMs can extract critical information like '
                        'chemical names, experimental conditions, and research findings, '
                        'significantly reducing manual efforts. In addition to data '
                        'preprocessing, LLMs facilitate literature reviews by rapidly scanning '
                        'vast amounts of research papers, summarizing key insights, and '
                        'constructing knowledge graphs to visualize relationships across '
                        'complex datasets. Furthermore, LLMs contribute to problem-solving by '
                        'generating theoretical insights and assisting in mathematical and '
                        'computational tasks. Finally, LLMs can support scientific programming '
                        'by automating code generation for data analysis and simulations. '
                        'These capabilities enhance efficiency, foster faster discoveries, and '
                        'lower the barrier to',
                'distance': 0.18813371658325195},
               {'chunk_id': 'https://openalex.org/W7126221181_chunk_6',
                'paper_id': 'https://openalex.org/W7126221181',
                'text': 'from free-text scientific articles and convert them into structured '
                        'data—such as through Named Entity Recognition (NER)—is crucial in '
                        'fields like computational chemistry and bioinformatics, where vast '
                        'amounts of unstructured text require parsing to construct structured '
                        'datasets [4]. Similarly, LLMs can streamline data conversion tasks. '
                        'For instance, in computational biology, LLMs can transform '
                        'experimental work and results into structured formats directly '
                        'compatible with simulation models. This process converts natural '
                        'language text into appropriate inputs for scientific workflows, '
                        'reducing manual effort and accelerating scientific processing. These '
                        'capabilities accelerate scientific work and lower the barrier to '
                        'handling large-scale, complex data. 2.2. Literature Review and '
                        'Knowledge Integration Scientists engage in an ongoing “race” for '
                        'knowledge by continuously acquiring new insights from millions of '
                        'current and historical publications. The sheer volume of published '
                        'literature makes it impractical for researchers to effectively track '
                        'the latest developments in their fields. LLMs can automate literature '
                        'surveys, information retrieval, and research summarization. For '
                        'instance, LLMs are being used to aggregate clinical trial records, '
                        'research publications, and patents to identify drug candidates and '
                        'emerging research hotspots; Employing LLMs to summarize',
                'distance': 0.1927291750907898},
               {'chunk_id': 'https://openalex.org/W7126221181_chunk_26',
                'paper_id': 'https://openalex.org/W7126221181',
                'text': 'to process and synthesize information from diverse fields will enable '
                        'scientists to tackle more complex, multi-faceted problems that '
                        'require knowledge from various domains. In conclusion, while there '
                        'are significant challenges to the adoption of LLMs in scientific '
                        'computing, the future looks promising. With continuous advancements '
                        'in technology, data privacy solutions, and policies that address the '
                        'socio-economic impacts of automation, LLMs will become indispensable '
                        'tools in accelerating scientific discovery. Through increased '
                        'efficiency, interdisciplinary collaboration, and enhanced accuracy, '
                        'LLMs are poised to revolutionize scientific research and computing, '
                        'making it more accessible and impactful than ever before. 4. '
                        'Conclusion In the ongoing advancement of scientific computing and '
                        'research, large language models (LLMs) are playing an increasingly '
                        'vital role. By automating data processing, literature analysis, and '
                        'code 270 generation, LLMs have significantly enhanced research '
                        'efficiency and propelled progress across multiple scientific fields. '
                        'However, despite their immense potential, LLMs still face numerous '
                        'challenges in scientific applications, particularly the phenomenon of '
                        '“hallucinations” and data privacy concerns. To maximize the '
                        'advantages of LLMs, researchers must rigorously validate their '
                        'outputs and ensure robust data security and privacy protections.',
                'distance': 0.23153281211853027},
               {'chunk_id': 'https://openalex.org/W7126221181_chunk_10',
                'paper_id': 'https://openalex.org/W7126221181',
                'text': 'optimization suggestions, and enable authors to swiftly correct '
                        'errors. This allows scientists to focus on core research questions '
                        'rather than excessive programming language issues [7]. 2.5. '
                        'Multidisciplinary Research and Cross-Domain Integration The trend '
                        'toward integrated research demands greater collaboration among '
                        'scholars from diverse disciplines. Interdisciplinary studies spanning '
                        'chemistry, physics, biology, and beyond require seamless knowledge '
                        'integration across fields. Large language models, trained on vast '
                        'datasets, can facilitate this cross-domain knowledge synthesis. LLMs '
                        'can also be applied to climate science. In climate science '
                        'applications, LLMs can analyze observational data from satellite '
                        'imagery, sensors, scientific literature, and other sources to extract '
                        'valuable insights and generate meaningful predictions. Their ability '
                        'to perform interdisciplinary analysis enables the generation of more '
                        'comprehensive research outcomes than those achievable within a single '
                        'discipline. Simultaneously, LLMs support interdisciplinary research, '
                        'effectively breaking down barriers between biology and chemistry, '
                        'chemistry and physics—fields often characterized by complex '
                        'overlapping knowledge. 266 3. Challenges and Future Prospects 3.1. '
                        'Challenges 3.1.1 Hallucination in Large Language Models When using '
                        "large language models (LLMs), such as OpenAI's GPT series, for "
                        'scientific computing, the phenomenon of “hallucination”',
                'distance': 0.2370438575744629}]}]
