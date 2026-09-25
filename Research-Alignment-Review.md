# Research Alignment Review

Version 0.1, 2026-09-19
Scope: the instrument (`indicators/`, `Indicator-Instrument.md`, `Instrument-Specification.md`), Phase I (`phase1/`), Phase II (`phase2/`), and the design spec, checked against `Continuous_Clinical_Assurance_Research_Plan.md` and `Evidence-and-measurement-matrix.md`.

## 1. Summary

- **Internal consistency.** After fixes, `scripts/check_consistency.py` reports 0 errors and 6 warnings. It checks file references, indicator, scenario, requirement, and reference IDs, section cross-references, stale terms, and numeric claims recomputed from the data. The warnings are expected (section 2).
- **Alignment with the plan.** The structure serves the plan's core claim, a translation from governance principle to risk to indicator to threshold to action. 19 of 20 matrix constructs now have an indicator, and the remaining one (evidence transparency) is out of scope by decision. One gap remains: the response half of the plan (governance interventions, RH5) is specified but not yet designed for testing.
- **Status.** Mostly design and specification. The Tier 1 simulator, the monitors for 25 indicators (13 model-level, the oversight indicators H1 to H7, and the workflow indicators O1, O2, O4, O5, and O6), and matched false-alarm calibration have been built and have run (180 tests, mutation-checked). Five exploratory studies have run (`phase2/M3-M4-Pilot.md`, `M5b-Reviewer-Pilot.md`, `M5c-RH3-Mixture.md`, `M5e-RH4-Workflow.md`, `M6-EHOR-Study.md`). No pre-specified hypothesis test has been run with the Holm family applied. "Covered" below means designed, not shown.
- **Largest risks.** Human interaction is simulated where the plan expects it to be observed (A1), and the pilot shows that CCA's advantage depends on the scenario mix (A10). Section 2a lists the advice applied. Section 4 ranks the rest.

## 2. Consistency defects found and fixed

| # | Defect | Fix |
|---|---|---|
| 1 | The instrument had no link to the Evidence and Measurement Matrix, so the principle-to-risk half of the plan's translation layer was missing from the records. | Added `matrix_construct` to all records. The renderer carries each construct's governance objective and clinical risk. A coverage table shows 19 of 20 constructs have indicators, and the last is recorded as out of scope. |
| 2 | Seven records (G1 to G6, L2) cited Phase II tests, but Phase II excludes them by design. | Reworded to a follow-on generative or agentic study. A check now enforces this. |
| 3 | `Instrument-Specification.md` said Phase II "tunes" the routing parameters, but Phase II fixes them in advance. | Corrected. The hysteresis j now has a default (2). |
| 4 | SC08 was tagged RH2 in the catalog, but the RH2 test never used it. SC03 is built so that calibration is the leading candidate, which favors RH2. | SC08 added to RH2 as the ecological check. |
| 5 | "Residual harm exposure" (Phase II) resembles the plan's secondary outcome "residual error or risk after intervention", which means something different. | Distinction stated in Phase II section 4.3. |
| 6 | Phase I counts were out of date after new references were verified. | Updated to 45 references (40 VALID, 5 WARNING). |
| 7 | G1 and G2 audits may be automated, with no mention of evaluator dependence. | A check for evaluator dependence added to both records. |
| 8 | The audit checker's own false positives (a quoted revision-log phrase, bare filenames). | Checker corrected. |

**Two errors of mine, disclosed.** First, my first read of `works.bib` counted 15 entries, but the file has 65, because my parser missed entries that share a line. That parse also attributed a DOI to the wrong paper. Both were corrected before any reference was added. Second, I suspected that the plan's manuscript list, numbered 7 to 15, was an artifact of my Word-to-markdown conversion. It is not: the Word original numbers it that way, because the list continues from an earlier list.

**Warnings that remain.** Six scenarios (SC05, SC12, SC15, SC18, SC21, SC24) are tagged for a hypothesis but have no harm-defined onset, so they are correctly outside the primary tests.

## 2a. Advice applied (2026-09-19)

| Item | What was done |
|---|---|
| D10 harm model | Residual harm exposure now counts all errors (weights 1, 7.1, 100, times 2.4 for a missed deterioration, elicited 2026-09-20 from two physicians, one of them the study's author), and the severity cutoff serves only EHOR. Kappa stays at 25 percent. Sub-threshold cells are reported by alert rate |
| D9 RH2 evidence | RH2 rests on scenarios where everything moves (SC08, later SC17 and SC25). SC03 was retired as redundant with SC08, and a new SC23 demonstrates the reporting-layer blind spot. Theta2 was amended |
| D11 build order | M3 and M4 were built first, for 12 model-level indicators (C3 was added later, with SC19). The reviewer layer (H1 to H6, SC10, SC11, SC13, the RH3 mixture test, and a first EHOR estimator study) followed. the workflow layer and the RH4 first pass have since been built. SC12, SC15, the Holm family, and the Tier 2 stream are next |
| A2 phase split | Accepted. The plan's markdown copy now says Phase II compares detection and Phase III adds interventions. The Word original in `archive/` was not changed |
| A3 matrix gaps | Indicator L5 (accountability completeness) and scenario SC24 were added. Evidence transparency is recorded as out of scope, with a reason, in the schema and the specification |
| A8 plan text | The plan's markdown copy now uses TAFD, numbers its manuscript list 1 to 9, and defines the operational trajectory. It keeps H1 to H5, and the instrument documents call them RH1 to RH5 |
| Operational trajectory | Adopted for the sequence of A_t, with one inline citation in the specification |
| A1 reviewer anchor | Your 2025 paper's abstract reports override rates by confidence band from 6,689 MIMIC-III cases. It is the right shape to anchor the reviewer model, but the abstract does not say whether those rates are observed clinician decisions or come from the framework's own rule, so the full text needs to be read first |

## 3. Traceability to the plan

Status terms: **Designed** means specified and ready to run, **Partial** means part of the requirement is unaddressed, and **Not yet** means no design exists.

### 3.1 Research questions

| Plan question | Addressed by | Status |
|---|---|---|
| Primary: operationalize governance as a continuous process that detects and responds to changes in reliability, uncertainty, human oversight, and operational risk | Instrument and specification (define and measure), Phase II (detect) | Partial: the respond half depends on Phase III, which is not yet designed |
| 1. What dimensions constitute CCA? | Five domains and six state components (mapped to 17 requirements in Phase I), and 19 of 20 matrix constructs | Partial: the set is proposed and mapped, but not independently validated |
| 2. Which indicators detect deterioration? | 41 candidate indicators, of which Phase II tests 34 | Designed |
| 3. Can calibration and uncertainty metrics be early-warning signals? | RH2 in Phase II: SC08, SC17, and SC25 (world drift, model regression, and compound drift), with SC20 supplementary, SC04 as the boundary case, and SC02 and SC23 as blind-spot demonstrations | Designed |
| 4. How can oversight effectiveness be measured, not documented? | EHOR with its companion IIR, the estimator study, RH3 mixture-stream test | Designed, with simulated reviewers (A1) |
| 5. Can interventions be linked to predefined thresholds? | `governance_action` at yellow, orange, and red for every indicator, and the trigger architecture | Partial: specified, with no evaluation design (RH5) |
| 6. Can the framework cover conventional, generative, and agentic AI? | `ai_types` tags, G1 to G6 and L2, the agentic trace | Partial: conceptual only. Phase II excludes 7 indicators, and NIST AI 600-1 was not retrieved |

### 3.2 Research gaps

| Plan gap | Evidence so far |
|---|---|
| 1. Static versus continuous governance | Partly supported: monitoring is required, but the retrieved sources say nothing on what continuous evidence must consist of (Phase I, R01, R12, R13) |
| 2. Principles versus measurements | Supported in the retrieved sources (F1, F3, F4) |
| 3. Model performance versus clinical operation | Not assessed in Phase I. Phase II tests detectability of workflow failures (RH4), which is not the same claim |
| 4. Human-in-the-loop versus effective oversight | Supported at capability level in the retrieved sources (F2) |

### 3.3 Hypotheses

| Plan | Here | Phase II test | Status |
|---|---|---|---|
| H1 continuous monitoring detects risk earlier | RH1 | theta1: mean relative reduction in restricted mean TAFD over 5 fair-test scenarios | Designed. Blind-spot scenarios are reported separately (D12 applied) |
| H2 calibration deteriorates before performance | RH2 | theta2: SC08, SC17, and SC25 (all metrics move). SC02 and SC23 are blind-spot demonstrations | Designed. The pilot shows calibration does not lead in SC08 |
| H3 oversight measures add information beyond documented review | RH3 | theta3: AUROC gain on mixture streams | Designed |
| H4 workflow indicators find failures model metrics miss | RH4 | theta4: detection difference on SC13, SC14, SC16, SC22 | Designed, and partly true by construction (Phase II section 9.1) |
| H5 risk-triggered interventions reduce persistence or severity | RH5 | none | Not yet |

### 3.4 Study phases

| Plan | Realized here |
|---|---|
| Phase I: synthesize FDA, NIST, WHO, IMDRF sources and peer-reviewed literature, and produce a Taxonomy and Control Matrix | Partial. 4 of 8 governance sources were retrieved (one only as a landing page), 17 requirements were extracted by a single coder, and a partial control matrix exists |
| Phase II: evaluate on tasks where predictions, confidence, outcomes, and human interaction can be observed, with controlled perturbations | Designed. Human interaction is simulated (A1) |
| Phase III: compare conventional monitoring with CCA, with thresholds triggering interventions. Outcomes are detection time, missed risk, persistent errors, and residual risk | Split. Here Phase II already compares detection between conventional monitoring and CCA (RH1 to RH4, including the plan's primary outcome). Phase III would add interventions (RH5). This differs from the plan (A2) |

### 3.5 Outcomes

The plan's primary outcome, time to detection, is TAFD here (`Instrument-Specification.md` section 7.1).

| Secondary outcome | Where addressed |
|---|---|
| False-positive governance alerts | Phase II false-alarm rate, matched across systems (section 7) |
| Missed-risk events | Phase II, reported per cell (section 9) |
| Calibration deterioration | Indicators U1 to U3, scenarios SC08 and SC17 |
| Inappropriate AI acceptance | H4, SC10 |
| Inappropriate overrides | H3, and the IIR reported with H5. SC12 |
| Workflow failures | O1 and O2, SC14 |
| Intervention latency | H6 measures reviewer latency. The plan may mean latency from alert to governance action, which belongs to Phase III. The term needs one definition |
| Residual error or risk after intervention | Phase III only |

### 3.6 Expected contributions

| Plan contribution | Status |
|---|---|
| Define CCA as a measurable lifecycle approach | Specified (`Instrument-Specification.md`) |
| A multidimensional framework connecting principles to observable indicators | Specified, with 19 of 20 matrix constructs covered and the last out of scope |
| Methods to detect post-deployment risk, including time to detection | Designed, not run |
| A risk-adaptive governance model linking deviations to interventions | Specified (trigger ladder), evaluation not designed |

### 3.7 Manuscript structure

| Plan section | Source material |
|---|---|
| Introduction | Plan, Phase I findings |
| Background | `phase1/Phase-I-Synthesis.md`, and the project lead's registered scoping review if it is completed |
| Conceptual framework | `Instrument-Specification.md` sections 3 to 6 |
| Measurement framework | `Indicator-Instrument.md`, `Instrument-Specification.md` sections 7 to 9 |
| Study design | `phase2/Phase-II-Simulation-Design.md` |
| Results | Nothing yet |
| Discussion, limitations | Threats sections of both specifications and this review |

### 3.8 Research program follow-ons (plan section 16)

| Follow-on | Coverage |
|---|---|
| Continuous Clinical Assurance (framework) | This program |
| Calibration as a Governance Signal | RH2 in Phase II |
| Effective Human Oversight | RH3 and the EHOR estimator study |
| Deployment Robustness and Temporal Governance Drift (longitudinal) | None. Phase II is simulated and short |
| Continuous Governance of Agentic Clinical AI | Indicators G4 to G6 and the trace tau only, with no empirical design |

## 4. Alignment risks and gaps, ranked (A9 was added after implementation began)

**A1. Human interaction is simulated (high).** The plan's Phase II calls for tasks where human interaction can be observed. Phase II here simulates reviewers, so RH3 and the EHOR conclusions are conditional on assumed reviewer behavior. To my knowledge neither MIMIC-IV nor eICU records clinician review of an AI output (please confirm). Options: state the results as detectability under assumptions (the current wording), add reader-study data to anchor the reviewer grid, or plan a small observational study.

**A2. Phase III scope differs from the plan (resolved).** The plan placed the conventional-versus-CCA comparison in Phase III together with interventions. The split adopted here puts detection in Phase II and interventions in Phase III. It was accepted, and the plan's markdown copy was edited to match (section 2a).

**A3. Matrix coverage (resolved).** Accountability completeness now has indicator L5. Evidence transparency is out of scope for predictive AI, with the reason recorded in `indicators/schema.yaml` and the specification.

**A4. The technology-agnostic claim is conceptual (medium).** Seven indicators (G1 to G6, L2) apply only to generative or agentic AI and have no empirical plan. The sources most relevant to generative AI (NIST AI 600-1, WHO guidance on large multimodal models) were not retrieved. RQ6 is answerable conceptually only.

**A5. Novelty positioning against your prior work (medium).** See section 5. The plan's contribution statements use "first" wording that I have not checked against your earlier preprints.

**A6. Phase I is partial and single-coded (medium).** It is a targeted requirement extraction and is not a systematic review, and it should be described that way. Your registered scoping review is the appropriate systematic evidence base.

**A7. Clinical parameters are placeholders (medium).** Severity weights, kappa, the false-alarm budget, and the RH1 margin (Phase II D2 to D5) are, apart from D3, D4, D5, and D23 (elicited from one rater, who is the study's author, or set by decision on 2026-09-20), proposals. The plan's H1 refers to clinically relevant risk. That claim will rest on clinician-set values. Only 2 of 40 indicators sit at evidence level 5, so clinical relevance in Phase II is defined by the simulation's harm model.

**A9. The harm model decides which cells have an onset (high, partly resolved).** With false positives counted, 17 of the 24 implemented cells have no onset at kappa = 0.25, because the baseline harm rose and dilutes relative changes. SC03 and SC05 became process scenarios by a rule fixed before any monitor ran. RH2 evidence comes from SC08 and SC09. RH1 and RH2 power still depends on the clinical values in decisions D3 and D4, which should be set before results are seen (`phase2/Phase-II-Simulation-Design.md` sections 2 and 16).

**A10. CCA's speed advantage is a label-lag effect (high, found in the pilot).** Across label lags 0, 2, and 8, CCA's reduction in time to detection is about zero at lag 0 and grows with the lag, because the conventional monitor detects at about the lag. The add-one arms show two label-free indicators (C2 and U5) carry the whole advantage. Where none responds (SC08), CCA is slower at every lag. The calibration arm is never earlier than the performance arm in the RH2 fair tests. Two implications: the plan's H1 holds only when outcome labels are delayed, and H2 is not supported in these simulations. Decisions D20 and D21 in the Phase II design address how to state them. The pilot is exploratory, with 150 replicates per cell and 12 of 34 indicators.

**A8. Plan text is out of step with the instrument documents (resolved in the markdown copy).** The markdown copy now uses TAFD, numbers its manuscript list 1 to 9, and states the phase split. The Word original in `archive/` still has the old text, so the two differ until it is updated. The plan keeps H1 to H5.

## 5. Relationship to your prior work

`works.bib` has 65 entries, some of them duplicates of the same work. The following are directly relevant, and six are now verified references (`phase1/Phase-I-Synthesis.md` section 4.1). I read the abstracts of four and only the titles of the others, so these notes are limited to that.

- **Operational trajectory** (`yu2026operational`, abstract read). Your preprint defines the operational trajectory as the time-course of a deployed system's measurable properties and proposes a trajectory predictability ratio. The sequence of the CCA assurance state A_t over the deployed life of a system appears to fit that definition, judging from the abstract. Decision for you: adopt "operational trajectory" as the name for the sequence (A_t) and cite the preprint, or keep the terms separate. Either way the overlap should be addressed before any novelty claim.
- **Registered scoping review** (`yu2026monitoring`, registration text read in part). It covers post-deployment surveillance, risk assessment, and lifecycle management. Its text says monitoring methods, action thresholds, and operational governance remain insufficiently standardized, which is Gap 2. The Phase I extraction should be positioned as a targeted complement.
- **Reliability-gated selective prediction** (`yu2026deferral`, abstract read). It uses a MIMIC-IV ICU cohort of 84,013 admissions with sepsis, mortality, and vasopressor tasks and a temporal holdout, and it combines confidence with data sufficiency, cohort similarity, temporal stability, and uncertainty consistency. It is a natural Tier 2 cohort (Phase II decision D8) and its dimensions overlap with CCA context and uncertainty indicators.
- **Evaluator dependence** (`yu2026evaluator`, abstract read). Relevant to G1 and G2 when audits are automated. A check is now in those records.
- **Confidence calibration and clinician trust** (`yu2025trust`, title only). The plan says Domain B extends a research program on confidence calibration and clinician trust. This paper is probably part of it, which I infer from its title alone.
- **Informative missingness in ICU models** (`yu2026missingness`, title only). It bears on how SC05 and SC06 should treat the missingness mechanism, and Phase II now says so.
- Titles suggest further overlap with agentic trajectory stability and agentic governance work. They were not assessed.

Self-citation stands at 6 of 45 references (13 percent), just under the project's 15 to 20 percent audit threshold. Adding more should be deliberate.

## 6. What has and has not been done

Done: the instrument and its records, the specification, the Phase I extraction and audit, the Phase II design and catalog, Phase II milestones M0, M1, M3, and part of M2, M4, M5, and M6 (180 tests, mutation-checked), five exploratory studies, and the scripts (four validators, a term counter, and a citation verifier), all passing.

Not done: any hypothesis test (the pilot is exploratory); the remaining perturbations, the reviewer and workflow layers, and M4 for the other window sizes and label lags; M5 to M7; Phase III design; the omitted governance sources; a second coder for Phase I; claim-level checks of references against full text; reading the full text of your 2025 trust paper; the Word original of the plan.

## 7. Decisions for you

1. Accept or reject the Phase II design changes in its revision log (30 changes, 18 alter the design you first reviewed).
2. Phase II D12: how theta1 treats scenarios where conventional metrics are blind. I recommend a primary set in which they can respond (SC08, SC09, SC17) and a supplementary blind-spot set. It is the harder test for CCA.
3. Phase II D13: how CCA combines many indicators at the yellow level, since the union rule costs sensitivity.
4. Phase II D3 and D4 (severity weights, kappa), which should be set by a clinician before results are seen, and D2, D5 to D8.
5. A1: whether the full text of your 2025 trust paper can anchor the reviewer model.
6. Whether to update the plan's Word original to match the markdown copy.
7. Whether to proceed with the reviewer layer next (Phase II D14).
8. Phase II D13, D14, and D18 to D19 were applied on your go-ahead (component-level budget split, label lags 0, 2, and 8, subgroup-aware onset, and a 1-window margin), along with D15 to D17. Decisions from the lag results, both accepted: D20 (state RH1 as conditional on the label lag) and D21 (report RH2 as a negative result, with a pre-specified rolling-calibration sensitivity). The window duration behind the margin is still a placeholder for you to set.

## 8. Re-running the checks

```
python scripts/validate_indicators.py --render
python scripts/validate_requirements.py
python scripts/validate_scenarios.py
python scripts/term_coverage.py
python scripts/validate_observables.py
python scripts/check_consistency.py
python -m pytest tests
python scripts/m2_expected_rhe.py
python scripts/m4_pilot.py
python scripts/verify_citations.py --only KEY   # needs network access
```
