# Phase I Synthesis: Governance Requirements and the Measurement Gap

Version 0.1 (partial), 2026-09-19
Status: partial. Four governance sources were retrieved (one only as a landing page). Four others were not. Section 2 lists exactly what is and is not covered.

Companions: `Assurance-Control-Matrix.md` (the control matrix), `requirements.yaml` (17 requirements with verbatim quotes), `term_coverage.md` (term counts), `Citation-Audit-Report.md` and `references.bib` (verified references), `sources/manifest.json` (source hashes).

## 1. Question and approach

Phase I asks whether the governance requirements that apply to deployed healthcare AI already specify what to measure after deployment, and where the Continuous Clinical Assurance (CCA) instrument adds specificity. The plan's four gaps supply the hypotheses to test: static versus continuous governance (Gap 1), principles versus measurements (Gap 2), model performance versus clinical operation (Gap 3), and human-in-the-loop versus effective human oversight (Gap 4).

Method, in order:

1. **Retrieve** primary documents and save the extracted text with a SHA-256 hash of the raw file (`sources/manifest.json`).
2. **Extract** requirements as verbatim quotes, each with a locator. `scripts/validate_requirements.py` rejects any quote that is not a verbatim substring of the saved text, after normalizing whitespace, case, and punctuation. It also rejects unknown indicator IDs. Six deliberate corruptions (a fabricated quote, a one-word alteration, an unknown indicator, a bad domain, a bad code, a duplicate ID) were each caught.
3. **Code** each requirement for measurability (five codes, defined in `requirements_schema.yaml`) and map it to CCA domains and indicators.
4. **Count** governance terms across the full retrieved texts (`scripts/term_coverage.py`) so that absence claims rest on counts.
5. **Verify** every cited reference against CrossRef, OpenAlex, Semantic Scholar, DataCite, or a publisher page (`scripts/verify_citations.py`).

Limitations of the method: one coder made all coding and mapping decisions, with no second reviewer. Term counts are regular-expression matches. They show whether a source uses a term, not how well it treats the concept, and contexts were read to interpret them.

## 2. Sources retrieved and not retrieved

| ID | Source | Retrieved as |
|---|---|---|
| S1 | NIST AI Risk Management Framework 1.0 (NIST AI 100-1), January 2023 | Full text |
| S2 | EU AI Act, Articles 14 and 72 | Text of two articles from an unofficial mirror. EUR-Lex blocked automated access. The regulation's official citation was not confirmed from the retrieved pages. |
| S3 | FDA guidance on predetermined change control plans for AI-enabled device software functions. The document states it was issued August 18, 2025 and originally issued December 4, 2024. | Full text |
| S4 | WHO, Ethics and governance of artificial intelligence for health (2021) | Landing page only. The page states six consensus principles and a set of recommendations. The principles themselves were not retrieved and are not listed here. |

Not retrieved: IMDRF Good Machine Learning Practice guiding principles (host unreachable); the FDA, Health Canada, and MHRA GMLP principles (page retrieved, principle text not recoverable); NIST AI 600-1, the generative AI profile; WHO guidance on large multimodal models; the official EU AI Act text; the full WHO 2021 guidance. The findings below therefore describe the retrieved sources, and coverage of the omitted ones remains open. In particular, the GMLP principles are a natural place to look for post-deployment monitoring language, and their absence here is a real limitation.

The EU AI Act mirror displays application dates for the high-risk provisions (2 December 2027 and 2 August 2028, citing Article 113(c)). These could not be checked against the official text, and this synthesis makes no claim about applicability dates.

## 3. Findings

### F1. Monitoring is required everywhere, but no retrieved requirement names a measure

Of 17 coded requirements, 9 require monitoring, assessment, or documentation without naming a measure, cadence, or threshold (`measure_unspecified`). The other codes are 4 `capability_unmeasured`, 2 `action_named`, 1 `trigger_unspecified`, and 1 `principles_only`. Representative text:

- NIST MEASURE 2.4 (R01): "The functionality and behavior of the AI system and its components – as identified in the MAP function – are monitored when in production."
- EU AI Act Art. 72(2) (R13): providers "shall actively and systematically collect, document and analyse relevant data" on performance "throughout their lifetime".

This supports Gap 2: the principle of post-deployment monitoring is well established, and the retrieved texts leave the measures to the implementer. NIST states this openly (R08): "Human judgment should be employed when deciding on the specific metrics related to AI trustworthiness characteristics and the precise threshold values for those metrics." That sentence also supports the instrument's design rule that margins are application-specific and set in the Validated Operating Envelope (VOE).

### F2. Human oversight is specified as a capability, and none of it is measured

Four requirements (R05, R09, R10, R11) concern human oversight. Each specifies what the overseer must be able to do, and none gives a way to measure whether that is achieved. NIST MAP 3.5 requires oversight processes to be "defined, assessed, and documented". EU Art. 14(4) requires that overseers can "properly understand the relevant capacities and limitations", "remain aware of the possible tendency of automatically relying or over-relying on the output", and "correctly interpret" the output. This supports Gap 4 at the level of the retrieved excerpts. It is the gap that the Effective Human Oversight Rate (EHOR) and the inappropriate-acceptance rate (H4) are designed to fill.

### F3. Calibration appears as a repair action, not a monitored signal

Across S1 to S3, "calibration" (excluding "recalibration") appears once, and it is a model-building task in NIST. "Recalibration" appears three times, all in NIST, as a management option ("Options may include recalibration, impact mitigation, or removal of the system from design, development, production, or use", R06) and an operations task (R07). Neither S2 nor S3 mentions either term. No retrieved requirement treats calibration error or confidence reliability as something to track. The retrieved texts thus name the remedy without naming the signal that would call for it. CCA pairs the two: U1 to U3 monitor calibration and their governance actions include recalibration. Whether calibration deterioration precedes other signals is the empirical question of research hypothesis RH2 and is untested here.

"Uncertainty" appears six times in NIST, and the contexts concern statistical uncertainty of measurement and inherent model limitations, for example "performance assessment methodologies with associated measures of uncertainty". None concerns whether stated confidence matches observed correctness.

### F4. Triggers have a precedent in change control, limited to re-training

The FDA guidance asks manufacturers to "identify any triggers for re-training (e.g., when the quantity of new data reaches a certain size or when a drift in data is observed over time)" (R15). This is the closest retrieved precedent for predefined triggers. It defines no drift metric or threshold, and the only action it names is re-training. CCA generalizes this to a graded green, yellow, orange, and red ladder with a predefined action at each level for every indicator.

### F5. Detection delay and near misses are absent from the retrieved text

"Near miss" and terms for detection time ("time to detect", "detection time") occur zero times in S1, S2, and S3. This is the count in the full text of NIST and the FDA guidance and in the two EU articles only. It shows that time to detection, the primary outcome of the plan, is not a stated requirement in these sources. It does not show that no governance source addresses it.

### F6. Lifecycle control is the best-developed area, but it is scoped to planned changes

The FDA guidance structures modifications into four Modification Protocol components: "data management practices, 2) re-training practices, 3) performance evaluation protocols, and 4) update procedures, including communication and transparency to users and real-world monitoring plans as applicable" (R14). It also asks for post-market surveillance plans "which may include real-world monitoring and notification requirements" (R16). This maps well onto CCA Domain E. The guidance addresses planned modifications by a manufacturer. It does not address detection of unplanned change in a deployed system, which is what L2 (unapproved prompt, configuration, and corpus changes) targets.

### F7. Automation bias is named in EU law and absent from the other two texts

The concept is named in Art. 14(4)(b), where the text uses both "automation bias" and "over-relying" (R10). Neither phrase occurs in NIST or in the FDA guidance. The two United States documents retrieved here do not name it.

### Gap status against the plan

| Plan gap | Status from retrieved sources |
|---|---|
| Gap 1, static versus continuous | Partly supported. Monitoring and lifetime data analysis are required (R01, R12, R13). None of the retrieved texts specifies what continuous evidence of assurance must consist of. |
| Gap 2, principles versus measurements | Supported (F1, F3, F4). |
| Gap 3, model performance versus clinical operation | Not assessed. The extracted passages do not test it. |
| Gap 4, human-in-the-loop versus effective oversight | Supported at capability level (F2). |

Indicator coverage: 17 of the 41 indicators are not mapped from any extracted requirement (P3, U2, U4, U6, G1, G2, G3, H6, O2, O3, O4, O5, G4, G5, L3, L5, G6). This reflects the extracted passages and the partial retrieval. It does not show that the sources omit them.

## 4. Literature checked so far

All 45 references passed verification for existence and metadata: 40 VALID (two or more databases agree on title, first author, and year) and 5 WARNING (one database only). The WARNING entries are `gretton2012` (the JMLR publisher page alone) and four of the project lead's own preprints, `yu2026operational`, `yu2026monitoring`, and `yu2026evaluator` (DataCite only) and `yu2026missingness` (CrossRef only). The audit is in `Citation-Audit-Report.md`. Database search budgets were partly exhausted during the run, and the report states which sources were unavailable.

**What verification covers.** It confirms that each paper exists with the stated title, first author, and year. It does not confirm what the paper found. No finding is attributed to any reference below, and each topical link comes from the title only. Claim-level checks against full text remain to be done before any sentence in a manuscript depends on a reference.

| Topic | Verified references (BibTeX keys) |
|---|---|
| Calibration and calibration drift | vancalster2019, davis2017, guo2017, steyerberg2010, naeini2015 |
| Dataset shift, drift detection, recurring local validation | finlayson2021, youssef2023, rabanser2019, gama2014, lu2019, feng2022 |
| External validation and deployment of a clinical model | wong2021, sendak2020 |
| Automation bias and human oversight | parasuraman2010, goddard2012, lyell2017, green2022, dratsch2023 |
| Reporting guidance | vasey2022 (DECIDE-AI; CrossRef lists a BMJ DOI and OpenAlex a Nature Medicine DOI for the same title and year, and both are kept as a note), collins2024 (TRIPOD+AI) |
| Data sources for Phase II | johnson2023 (MIMIC-IV), pollard2018 (eICU) |
| The project lead's related work (section 4.1) | yu2026operational, yu2026monitoring, yu2026deferral, yu2026evaluator, yu2025trust, yu2026missingness |
| Governance source with a DOI | nist2023 (NIST AI RMF 1.0) |
| Statistical and monitoring methods | see section 5 |

### 4.1 Related prior work by the project lead

`works.bib` has 65 entries by the project lead, some of them duplicates of the same work. Most are outside this project's scope. Six bear directly on it and are now verified references (the other entries were not assessed). Only the abstracts of four were read, and the rest were assessed from titles, so the descriptions below are limited to what those abstracts state.

| Reference | What the abstract or title says | Relevance to this program |
|---|---|---|
| yu2026operational | Defines the operational trajectory of a deployed system, the time-course of its measurable properties, as the unit of study after deployment (abstract read) | Close to the assurance state sequence A_t of the instrument. Positioning and terminology need a decision (`../Research-Alignment-Review.md`) |
| yu2026monitoring | A registered scoping review of post-deployment surveillance, risk assessment, and lifecycle management. The registration text cites a 2024 scoping review in which only 9 of 39 sources described clinically tested or implemented monitoring methods, with limited guidance on metrics and thresholds (registration text read in part) | The systematic evidence base for Gap 2. This synthesis is a targeted, single-coder extraction and does not replace it |
| yu2026deferral | Reliability-gated selective prediction on 84,013 MIMIC-IV ICU admissions across three tasks (sepsis, mortality, vasopressor need), with a temporal holdout (abstract read) | A candidate Tier 2 cohort and task set for Phase II (decision D8) |
| yu2026evaluator | Names evaluator dependence, where an automated judge shares systematic biases with the audited system and detection of failures falls (abstract read) | Relevant to G1 and G2, whose audits may be automated |
| yu2025trust | Confidence calibration and transparency for clinician trust in AI diagnostics (title only) | The prior program the plan says Domain B extends |
| yu2026missingness | Informative missingness as a clinical signal in ICU prediction models (title only) | Bears on the mechanism assumed in SC05 and SC06 (C4) |

Self-citation: these six are 6 of 45 references (13 percent). The project's guideline is to audit any proportion above 15 to 20 percent.

## 5. Methods citations resolved for the instrument specification

Existence and metadata are verified for each. Claim-level fit is not yet confirmed.

| Method used in the instrument | Candidate reference key |
|---|---|
| Maximum mean discrepancy | gretton2012 (WARNING, one source) |
| Wasserstein distance (earth mover's distance) | rubner2000 |
| False-discovery-rate control | benjamini1995 |
| Sequentially rejective multiple tests (Holm) | holm1979 |
| CUSUM | page1954 |
| EWMA | roberts1959 |
| Sequential probability ratio test | wald1945 |
| Expected calibration error | naeini2015 |
| Brier score | brier1950 |
| Calibration slope and intercept | steyerberg2010 |
| Bootstrap; block bootstrap for dependent data | efron1979; kunsch1989 |
| DeLong comparison of AUROC | delong1988 |
| Horvitz-Thompson estimation | horvitz1952 |
| Kaplan-Meier | kaplan1958 |
| Cox regression | cox1972 |
| Restricted mean survival time | royston2013 |
| Decision curve analysis | vickers2006 |

Still unresolved and marked `[CITATION NEEDED]`: the population stability index (no single authoritative source was searched), the Kolmogorov-Smirnov test (not searched), and every governance source in the "Not retrieved" list of section 2.

## 6. Phase I status and next steps

Delivered: a validated control matrix (partial), a verified reference set, and evidence that supports Gaps 2 and 4 and part of Gap 1 for the retrieved sources.

To complete Phase I:

1. Retrieve the omitted sources (IMDRF GMLP, FDA/Health Canada/MHRA GMLP, NIST AI 600-1, WHO 2021 and the WHO large-model guidance, official EU AI Act text) and extend `requirements.yaml`. The validator and matrix regenerate from it.
2. Have a second coder independently code the measurability of the 17 requirements, then report agreement.
3. Run claim-level checks against the full text of any reference a manuscript sentence will rely on.
4. Test Gap 3 directly, for example with a term count for workflow and operational failure language, before claiming support.
