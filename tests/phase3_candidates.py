"""Saved metadata of Phase 3 live-run candidates (data/runs/phase3-two-paper, 39 papers).

Copied verbatim from the saved state.json (git-ignored, so the tests embed it).
Keys are the candidates' 1-based retrieval positions in that run. RELEVANCE holds
the scores computed offline from these records by PaperRelevanceRanker
(topic_embedding_cosine_v2, BAAI/bge-small-en-v1.5) against TOPIC.
"""

TOPIC = 'Multi-agent LLM systems for scientific literature review and automated research gap discovery'

QUERIES = ['"multi-agent" AND "large language model" AND "literature review"',
 '"automated research gap" AND "LLM" AND "scientific"',
 '"collaborative agents" AND "knowledge extraction" AND "academic literature"',
 '"LLM" AND "gap detection" AND "research"',
 '"agent-based" AND "scientific review" AND "automated"']

RECORDS = {3: {'title': 'Large Language Model Applied in Multi-agent System—A Survey',
     'authors': ['Kaiwen Dong'],
     'year': 2024,
     'doi': '10.54254/2755-2721/109/20241330',
     'abstract': '<jats:p>The application of large language models (LLMs) in single-agent '
                 'systems within complex environments has proven successful, prompting a '
                 'growing interest in their use within multi-agent systems (MAS). Despite the '
                 'impressive capabilities of LLMs, it remains unclear how they can be '
                 'optimally integrated and utilized to empower agents in MAS. Understanding '
                 'how to effectively leverage the advantages of LLMs to enhance agent '
                 'performance is crucial. This survey provides a comprehensive overview of the '
                 'application of LLMs in MAS, focusing on their impact on agent cooperation, '
                 'reasoning, and adaptive abilities. Finally, we discuss future directions and '
                 'open questions in this evolving field.</jats:p>',
     'source': 'crossref',
     'source_id': '10.54254/2755-2721/109/20241330',
     'landing_url': 'https://doi.org/10.54254/2755-2721/109/20241330',
     'pdf_url': None},
 12: {'title': 'LLM-Driven Research Assistant for Automated Literature\n'
               'Analysis and Research Gap Detection',
      'authors': ['Bijoyini Bagchi',
                  'Kaushik Banerjee',
                  'Tunir De',
                  'Trina Chakraborty',
                  'Sayudh Chakraborty'],
      'year': 2026,
      'doi': '10.13052/rp-9788743817109a015',
      'abstract': None,
      'source': 'crossref',
      'source_id': '10.13052/rp-9788743817109a015',
      'landing_url': 'https://doi.org/10.13052/rp-9788743817109a015',
      'pdf_url': None},
 30: {'title': 'AIssistant: Human-AI Collaborative Review and Perspective Research Workflows '
               'in Data Science',
      'authors': ['Sasi Kiran Gaddipati', 'Farhana Keya', 'Gollam Rabby', 'Sören Auer'],
      'year': 2026,
      'doi': 'https://doi.org/10.1007/978-981-92-1468-6_35',
      'abstract': None,
      'source': 'openalex',
      'source_id': 'https://openalex.org/W4415315131',
      'landing_url': 'https://doi.org/10.1007/978-981-92-1468-6_35',
      'pdf_url': 'https://arxiv.org/pdf/2509.12282'},
 21: {'title': 'Comparative assessment of NaOH and NH4OH in alkaline extraction of lentil and '
               'chickpea sidestream proteins: impacts on protein quality and process '
               'sustainability',
      'authors': ['Eram Shaikh', 'A. Patti', 'A. Robinson', 'Amit Arora'],
      'year': 2026,
      'doi': '10.1007/s00217-026-05246-x',
      'abstract': None,
      'source': 'semantic_scholar',
      'source_id': 'cdc0249c7056a6325f799cdc29ae163bc4cf833a',
      'landing_url': 'https://www.semanticscholar.org/paper/cdc0249c7056a6325f799cdc29ae163bc4cf833a',
      'pdf_url': ''},
 34: {'title': 'Development of Reagents - Blowing Agents Based on Local Raw Materials and '
               'their Research',
      'authors': ['Yusupov S.K.', 'Yodgarov N.', 'F.M. Yusupov', 'Байматова Гулноза Ахмедовна'],
      'year': 2024,
      'doi': 'https://doi.org/10.35940/ijbsac.i0516.10090524',
      'abstract': 'The development and search for effective reagents and optimal technological '
                  'conditions is one of the main factors for improving the coal flotation '
                  'process. Solving these problems increases the technological and economic '
                  'efficiency of flotation enrichment. The article discusses methods for '
                  'producing foaming regents based on waste from the distillation process of '
                  'cottonseed oil - gossypol resin and methods of flotation of various classes '
                  'of fine coal. The advantages of this foaming agent include the availability '
                  'of raw materials, which are a product of domestic industry, produced at the '
                  'enterprises of the Kattakurgan MJK and amino alcohols (mono- , di- and '
                  'triethanolamine) , sufficient resources and relative low cost. Conducted '
                  'research to assess the effectiveness, surface activity and foaming ability '
                  'of the resulting reagents - in foaming agents, according to the proposed '
                  'technology of the studied technical products, the resulting product belongs '
                  'to the surface-active reagents and has high flotation activity. The '
                  'physico-chemical and foam-forming properties, foam stability, destruction '
                  'kinetics and other properties of flotation reagents - foaming agents SK-1 '
                  'and SK-2 - have been studied. The results of studies of the influence of '
                  'foaming reagents in the process of flotation and extraction of precious and '
                  'non-ferrous metals are presented. The optimal option for the effective '
                  'composition of composite chemical flotation reagents - foaming agents and '
                  'their parameters was selected. The results of scientific research conducted '
                  'at JSC NMMC on the use of analogues of the foaming agents SK-1 and SK-2 are '
                  'given in comparison with the reagents (Oxal) currently used at the plant. '
                  'The elemental composition of man-made waste samples using reagents SK-1 and '
                  'SK-2 was determined.',
      'source': 'openalex',
      'source_id': 'https://openalex.org/W4398189926',
      'landing_url': 'https://doi.org/10.35940/ijbsac.i0516.10090524',
      'pdf_url': 'https://www.ijbsac.org/wp-content/uploads/papers/v10i9/I051610090524.pdf'},
 18: {'title': 'Digitally enhanced maintenance: path towards continuous value creation and '
               'sustainability',
      'authors': ['John Ahmet Erkoyuncu', 'Fazel Ansari', 'Benoît Iung', 'Shozo Takata'],
      'year': 2026,
      'doi': 'https://doi.org/10.1016/j.cirp.2026.05.002',
      'abstract': 'Industrial maintenance plays a critical role in enabling complex assets '
                  'such as production equipment and more widely factories to be available for '
                  'use in a sustainable manner. With the emergence of digital technologies '
                  'such as digital twins, IoT, virtual and augmented reality, and machine '
                  'learning, significant opportunities are observed to transform the way in '
                  'which maintenance is delivered. It is anticipated that maintenance will '
                  'become more affordable, proactive, predictive, outcome driven and '
                  'sustainable in the future. This paper presents the state of the art and '
                  'industrial perspectives on digitally enhanced maintenance covering current '
                  'approaches and challenges. Furthermore, a comprehensive framework is '
                  'presented to structure future research directions by considering the '
                  'concept of a maintenance ecosystem. Finally, a strategic vision is set out '
                  'to convey the future of maintenance.',
      'source': 'openalex',
      'source_id': 'https://openalex.org/W7168129930',
      'landing_url': 'https://doi.org/10.1016/j.cirp.2026.05.002',
      'pdf_url': 'https://www.sciencedirect.com/science/article/pii/S0007850626001393/pdf'},
 22: {'title': 'Executive Summary: The Pediatric Extracorporeal Membrane Oxygenation '
               'Anticoagulation CollaborativE (PEACE) Consensus Conference*',
      'authors': ['Peta M. A. Alexander',
                  'M. Bembea',
                  'K. Cashen',
                  'I. Cheifetz',
                  'H. Dalton',
                  'A. Himebauch',
                  'O. Karam',
                  'K. Moynihan',
                  'M. Nellis',
                  'C. Ozment',
                  'L. Raman',
                  'N. Rintoul',
                  'A. Said',
                  'Arun Saini',
                  'Marie E. Steiner',
                  'R. Thiagarajan',
                  'K. Watt',
                  'A. Willems',
                  'N. Zantek',
                  'R. Barbaro',
                  'K. Steffen',
                  'A. Vogel',
                  'C. Almond',
                  'M. Anders',
                  'G. Annich',
                  'L. Brandão',
                  'W. Chandler',
                  'M. Delaney',
                  'R. DiGeronimo',
                  'S. Emani',
                  'S. Gadepalli',
                  'Alejandro V. García',
                  'B. Haileselassie',
                  'R. Hyslop',
                  'M. Kneyber',
                  'L. Baumann Kreuziger',
                  'J. Le',
                  'L. Loftis',
                  'A. McMichael',
                  'D. McMullan',
                  'P. Monagle',
                  'K. Nicol',
                  'M. Paden',
                  'J. Patregnani',
                  'J. Priest',
                  'L. Raffini',
                  'L. Ryerson',
                  'S. Sloan',
                  'J. Teruya',
                  'Andrew R. Yates',
                  'A. Gehred',
                  'E. Lyman',
                  'J. Muszynski'],
      'year': 2024,
      'doi': '10.1097/PCC.0000000000003480',
      'abstract': 'OBJECTIVES: To present recommendations and consensus statements with '
                  'supporting literature for the clinical management of neonates and children '
                  'supported with extracorporeal membrane oxygenation (ECMO) from the '
                  'Pediatric ECMO Anticoagulation CollaborativE (PEACE) consensus conference. '
                  'DATA SOURCES: Systematic review was performed using PubMed, Embase, and '
                  'Cochrane Library (CENTRAL) databases from January 1988 to May 2021, '
                  'followed by serial meetings of international, interprofessional experts in '
                  'the management ECMO for critically ill children. STUDY SELECTION: The '
                  'management of ECMO anticoagulation for critically ill children. DATA '
                  'EXTRACTION: Within each of eight subgroup, two authors reviewed all '
                  'citations independently, with a third independent reviewer resolving any '
                  'conflicts. DATA SYNTHESIS: A systematic review was conducted using MEDLINE, '
                  'Embase, and Cochrane Library databases, from January 1988 to May 2021. Each '
                  'panel developed evidence-based and, when evidence was insufficient, '
                  'expert-based statements for the clinical management of anticoagulation for '
                  'children supported with ECMO. These statements were reviewed and ratified '
                  'by 48 PEACE experts. Consensus was obtained using the Research and '
                  'Development/UCLA Appropriateness Method. Results were summarized using the '
                  'Grading of Recommendations Assessment, Development, and Evaluation method. '
                  'We developed 23 recommendations, 52 expert consensus statements, and 16 '
                  'good practice statements covering the management of ECMO anticoagulation in '
                  'three broad categories: general care and monitoring; perioperative care; '
                  'and nonprocedural bleeding or thrombosis. Gaps in knowledge and research '
                  'priorities were identified, along with three research focused good practice '
                  'statements. CONCLUSIONS: The 91 statements focused on clinical care will '
                  'form the basis for standardization and future clinical trials.',
      'source': 'semantic_scholar',
      'source_id': '5c8aa096889a969a16343c89ae371350d413a197',
      'landing_url': 'https://www.semanticscholar.org/paper/5c8aa096889a969a16343c89ae371350d413a197',
      'pdf_url': 'https://doi.org/10.1097/pcc.0000000000003480'},
 32: {'title': 'Blockchain System in Higher Education: Issues and Challenges — A Scientific '
               'Review',
      'authors': ['Umesh Sarkar'],
      'year': 2026,
      'doi': 'https://doi.org/10.30954/2322-0465.01.2026.9',
      'abstract': 'Blockchain technology offers powerful solutions to one of higher '
                  "education's most persistent challenges: the cumbersome and fraud-prone "
                  'process of credential verification.Traditional methods of authenticating '
                  'academic certificates are slow, manual, and resource-intensive.Blockchain '
                  'promises to solve this by creating a secure, decentralized, and immutable '
                  'digital ledger, enabling instant and permanent verification of degrees and '
                  'academic credits, independent of the originating institution.Despite its '
                  'promise, implementing blockchain in higher education introduces a series of '
                  'complex technical, organizational, and regulatory challenges.The most '
                  'immediate barriers are technical and financial -decentralized academic '
                  'networks require advanced digital infrastructure and skilled IT '
                  'professionals that many universities lack.High computational power and '
                  'energy consumption raise significant sustainability concerns.Systemic and '
                  'datagovernance issues present further hurdles, most notably the lack of '
                  'standardized protocols across diverse educational ecosystems.Privacy '
                  "regulations such as GDPR conflict directly with blockchain's immutability "
                  'principle.Furthermore, the scarcity of qualified blockchain developers and '
                  'resistance among institutional stakeholders continue to impede '
                  'adoption.This paper systematically reviews the core features and processes '
                  'of blockchain technology, its benefits for higher education, and the '
                  'multifaceted challenges that must be addressed for successful '
                  'implementation.A synthesis of current blockchain applications in education '
                  'is presented alongside a discussion of emerging solutions, supported by '
                  'comparative dataset analysis.The review concludes with actionable '
                  'recommendations for institutions, policymakers, and researchers to '
                  'facilitate a responsible and scalable transition toward blockchain-enabled '
                  'academic ecosystems.',
      'source': 'openalex',
      'source_id': 'https://openalex.org/W7214762920',
      'landing_url': 'https://doi.org/10.30954/2322-0465.01.2026.9',
      'pdf_url': 'https://doi.org/10.30954/2322-0465.01.2026.9'}}

RELEVANCE = {3: 0.7816, 12: 0.8898, 30: 0.7829, 21: 0.5174, 34: 0.5536, 18: 0.6148, 22: 0.5397, 32: 0.5910}
