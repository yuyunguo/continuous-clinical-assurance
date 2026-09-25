# Clinical AI Assurance Instrument Specification

Version 0.1 (draft), 2026-09-19
Companions: `Indicator-Instrument.md` (the 41 indicator records, generated from `indicators/*.yaml`), `Continuous_Clinical_Assurance_Research_Plan.md` (the research plan), `Evidence-and-measurement-matrix.md` (the matrix this instrument formalizes).

## 1. Purpose and scope

This specification defines the constructs, rules, and estimands that the 41 indicator records share. It answers four questions. What does "within the validated envelope" mean? How do indicators combine into a governance decision without a composite score? What counts as evidence? How are the two headline measures, Time to Assurance Failure Detection (TAFD) and the Effective Human Oversight Rate (EHOR), defined precisely enough to estimate?

The instrument is technology-agnostic. Predictive, generative, and agentic systems share one schema, and indicators carry `ai_types` tags. All 41 indicators are candidates. None has been validated, and thresholds are rules over margins that each application sets, not universal numbers.

## 2. Terminology alignment

| Research plan term | Term in this specification | Note |
|---|---|---|
| Time to Assurance Failure (TAF) | Time to Assurance Failure Detection (TAFD) | The plan's definition is a detection delay. The old name reads as time until failure occurs. |
| Time to Detection (primary outcome label) | TAFD | Same quantity. |
| EHOR = intercepted errors / errors requiring human detection | EHOR, as defined in section 7.2 | Denominator, numerator, estimator, and companion measure are now explicit. |
| Trace {I, R, P, T, A, O} | tau = (I, R, P, X, A, O) | T was used for both the trace and tool use. X is tool execution. |
| Orange: "significant deviation" | Orange: substantial deviation | "Significant" is reserved for statistical significance. |
| Research hypotheses H1-H5 | RH1-RH5 | The plan's H1-H5 collide with the oversight indicator IDs H1-H7. RH1 to RH5 map one to one onto the plan's H1 to H5. |

The research plan document is unchanged and still uses the old names.

## 3. Validated Operating Envelope (VOE)

### 3.1 Definition

The VOE is the documented set of conditions under which an AI system was validated and is approved to operate. It has five parts.

| Part | Contents |
|---|---|
| Population | Age range, condition, care setting, demographic composition, outcome prevalence |
| Data | Required variables, missingness limits, measurement ranges, data freshness |
| Performance | Validated sensitivity, specificity, calibration, error rates, and their intervals |
| Workflow | Intended user, intended workflow, required human review, escalation process, points of no return |
| System | Model version, prompt version, retrieval corpus, tools, agent permissions |

The governance question is not "is the model good?" but "is the system operating within the conditions under which it was validated?"

### 3.2 Setting and changing the VOE

The VOE is fixed before deployment from validation evidence, a clinical risk assessment, and the sign-off of a named accountable owner. It is versioned. A change to any part of it is a lifecycle event, monitored by L2 and controlled under L4. The VOE also fixes, for each indicator, the baseline, the margins (section 3.3), and the parameters the specification leaves open (severity cutoff s*, points of no return, audit sampling design).

### 3.3 Margins and indicator status

For indicator k in window t, let d_k(t) be the deviation of its estimate from the baseline, in the indicator's own units. The VOE sets three ordered margins per indicator, m_Y < m_O < m_R. Indicator status s_k(t) is:

- green: d_k(t) within m_Y (or no alert rule fired),
- yellow: m_Y exceeded,
- orange: m_O exceeded,
- red: m_R exceeded, or a sentinel rule fired.

Some rules are two-sided (H2) or use a floor rather than a deviation (P1, P2). The `alert_rule` field of each record states the exact form. Margins are chosen from clinical risk, not from statistical convenience alone.

## 4. Assurance state

### 4.1 Multidimensional state

Assurance at time t is the vector

`A_t = (C_t, P_t, U_t, H_t, O_t, L_t)`

Each component is itself a set of indicators. The 41 indicators distribute as follows.

| Component | Measurement class | Indicators | n |
|---|---|---|---|
| C, clinical-context validity | Validity | C1, C2, C3, C4, C5 | 5 |
| P, performance | Reliability | P1, P2, P3, P4, P5, P6, G1, G2, G3 | 9 |
| U, uncertainty and calibration | Uncertainty | U1, U2, U3, U4, U5, U6 | 6 |
| H, human oversight | Human control | H1, H2, H3, H4, H5, H6, H7 | 7 |
| O, operational safety | Operational safety | O1, O2, O3, O4, O5, O6, G4, G5 | 8 |
| L, lifecycle stability | Lifecycle stability | L1, L2, L3, L4, L5, G6 | 6 |

The sequence of A_t over the deployed life of a system is its operational trajectory, a term taken from the project lead's related work [yu2026operational]. The six components are a conceptual decomposition. They are not summed, weighted, or averaged. The plan's five domains relate to the components as follows: Domain A contains C, Domain B contains P and U, Domain C contains H, Domain D contains O, and Domain E contains L. The validator enforces this mapping.

Each record also links to one construct of `Evidence-and-measurement-matrix.md`, which supplies the governance objective and clinical risk the indicator operationalizes. 19 of the matrix's 20 constructs have at least one indicator. Evidence transparency has none by decision. Predictive models here return a score, not a source-attributed explanation, and explanation completeness has no agreed measure, so it is recorded as a limitation.

### 4.2 Governance status without a score

Governance needs one decision at a time, but the decision is an ordinal routing rule, not a score. The candidate rule has three parts.

1. Component status is the highest indicator status within the component.
2. System status is the highest component status that passes corroboration. An orange or red status passes when either (a) it persists for k consecutive windows, or (b) alerts of at least that level occur in two or more different components. Sentinel rules (for example, a highest-severity error in P5, or an executed out-of-scope action in G6) bypass corroboration.
3. Return to a lower status requires j consecutive windows within margin (hysteresis).

The parameters k and j and the corroboration rule are design candidates. Phase II fixes k = 2 with corroboration on, and j = 2, before any run, reports other settings as sensitivity analyses, each calibrated to the same false-alarm budget (section 10), and does not select among them on performance. The VOE records the values chosen for a deployment.

## 5. Trigger architecture

| Level | Meaning | Default governance response |
|---|---|---|
| Green | Within the validated envelope | Routine monitoring |
| Yellow | Early warning | Increase monitoring or human oversight |
| Orange | Substantial deviation | Restrict use; initiate investigation or revalidation |
| Red | Critical safety signal | Suspend or substantially restrict the function pending investigation |

Rules that apply to every indicator:

- Every alert level maps to a predefined action in the record's `governance_action`. An indicator without a predefined action is not admitted.
- Thresholds belong to the application. This is not a universal scoring system.
- A red status requires a decision by the named accountable owner within a time the VOE specifies.
- Actions taken after a trigger are logged, so that Phase III can evaluate whether interventions reduced the persistence or severity of deviations.

Worked example (illustrative, not empirical). A calibrated binary risk model has baseline ECE 0.04 and stable AUROC. ECE rises to 0.11 while AUROC stays within its validated range. Conventional monitoring reports acceptable performance. The instrument raises yellow on U1 and U3 (increase human review). If the drift persists, status becomes orange (recalibration and revalidation). If the drift is linked to clinically consequential errors, status becomes red (restrict or suspend the affected use case). All values are for demonstration.

## 6. Evidence hierarchy

| Level | Evidence | Indicators | n |
|---|---|---|---|
| 1 | System evidence: logs, outputs, prompts, versions, retrieval records | C1, C2, C4, C5, U5, G3, L2, L3, G6 | 9 |
| 2 | Performance evidence: accuracy, calibration, drift, robustness | P1, P2, P3, P4, P6, U1, U2, U3, U4, U6, G1, G2, L1 | 13 |
| 3 | Human-interaction evidence: acceptance, override, review, intervention | H1, H2, H3, H4, H5 | 5 |
| 4 | Workflow evidence: completion, escalation, delays | C3, H6, H7, O1, O2, O3, O4, O5, G4, G5, L4, L5 | 12 |
| 5 | Clinical outcome evidence: patient outcomes, safety events | P5, O6 | 2 |

Technical evidence does not automatically constitute clinical evidence. Two consequences follow.

First, only 2 of the 41 indicators sit at level 5. Outcome evidence is sparse, delayed, and under-reported, so it cannot serve as the only detector. It serves instead as the reference against which Phase II and Phase III judge whether lower-level alerts were clinically relevant. Second, every claim about detection benefit is stated at the level of evidence that supports it. A change in ECE (level 2) does not demonstrate patient benefit.

## 7. Two defined measures

### 7.1 Time to Assurance Failure Detection (TAFD)

**Definition.** TAFD is the elapsed time between the onset of a clinically meaningful deviation from the VOE and its first detection by the governance system.

**Onset (t0).** In Phase II simulation, t0 is the first window in which the expected residual harm exposure reaches (1 + kappa) times its baseline, computed from the simulator's truth (`phase2/Phase-II-Simulation-Design.md` section 4). It is distinct from the deviation start s0, when the perturbation begins. For an abrupt perturbation t0 equals s0, and for a ramped one t0 is later. In a real deployment the VOE specifies delta per application. In retrospective real data, t0 is an adjudicated change point and is labeled as such, with its uncertainty reported.

**Detection (t_d).** t_d is the first alert at or above level L* after t0. The primary analysis uses L* = yellow. The secondary analysis uses L* = orange, the first level that triggers a restriction, and is reported as time to actionable detection. Detection is defined on alert episodes. An episode starting before s0 is a false alarm. An episode starting from s0 up to t0 is an anticipatory detection with TAFD of 0 and a recorded lead time. An episode starting at or after t0 is a detection with TAFD equal to its start minus t0. An episode already active at s0 that began before it does not count as detection.

**Censoring.** If no alert occurs before the end of observation, TAFD is right-censored at the observation horizon. Analyses that average only detected cases are biased toward the better-performing system, so TAFD is analyzed as a time-to-event outcome.

**Estimands.** For each scenario, the CCA system and the conventional system (monitoring based primarily on model-performance metrics P1, P2, P4) are run on the same data stream. The comparison reports:

- the difference in restricted mean TAFD up to a horizon h,
- the hazard ratio for detection, stratified by scenario,
- the proportion detected by h.

**Matched false-alarm budget.** A shorter TAFD is uninformative if it comes from a more sensitive, noisier system. Both systems are calibrated to the same pre-onset false-alarm rate before TAFD is compared. The protocol fixes that rate. False-alarm rate, missed-risk events, and alert burden are reported alongside TAFD.

**Time scale.** TAFD is reported both in calendar time and in monitoring windows or case counts, because case volume differs across settings.

### 7.2 Effective Human Oversight Rate (EHOR)

**Purpose.** Human oversight is measured by what it intercepts, not by whether a human is documented in the loop.

**Terms.**

- *AI error requiring human detection* (R = 1): an AI output that reference-standard adjudication, blind to the reviewer's action, shows to deviate from the truth at severity at or above the VOE cutoff s*, such that acting on it unchanged would cause a clinically meaningful deviation in care.
- *Point of no return*: the workflow moment after which the erroneous output can no longer be prevented from affecting care. The VOE defines it for each workflow, and for each action class in agentic systems.
- *Intercepted* (I = 1): R = 1 and the reviewer rejected, corrected, or escalated the output before the point of no return. A correct detection after that point is recorded as late detection and does not count as interception.
- *Wrongly intervened* (Z = 1): a correct output that was reviewed and then rejected or changed.

**Estimator.** Errors caught by reviewers are visible by construction, while errors they missed surface only through later outcomes or audit. Counting only visible errors inflates EHOR. The estimate therefore comes from an independent audit of a probability sample of outputs. Each sampled output i has inclusion probability pi_i, and adjudication is blind to the reviewer's action. With weights w_i = 1/pi_i:

`EHOR = sum(w_i * I_i) / sum(w_i * R_i)`

The sampling design may oversample cases likely to be errors (for example, by confidence stratum) but the weights must reflect it. In the Tier 1 simulation, oversampling low-confidence outputs did not reliably reduce error compared with a flat sample (`phase2/M6-EHOR-Study.md`), so treat stratification as optional until real data show a gain.

**Companion measure.** Rejecting every output gives EHOR = 1. EHOR is therefore never reported alone. The inappropriate-intervention rate is

`IIR = sum(w_i * Z_i) / sum(w_i * V_i)`, where V_i = 1 for a correct output that was reviewed.

A rise in EHOR with a concurrent rise in IIR is flagged as indiscriminate rejection, not improved oversight.

**Decomposition.** Let Cov_R = P(reviewed before the point of no return | R = 1) and EHOR_c = P(intercepted | R = 1, reviewed in time). Then

`EHOR = Cov_R * EHOR_c`

EHOR_c is the conditional variant. It answers how well reviewers perform when they do review, and it must always be labeled as conditional. Cov_R is not the same as H1 review coverage. H1 is coverage over all outputs requiring review, while Cov_R is coverage over the erroneous ones, which is the quantity that matters for safety.

**Illustrative example (not empirical).** Of 100 AI errors requiring detection, 20 bypassed review and 80 were reviewed in time. Reviewers intercepted 72 of the 80. Then Cov_R = 0.80, EHOR_c = 0.90, and EHOR = 72 / 100 = 0.72. A definition that dropped the 20 bypassed errors would report 0.90 and hide the coverage failure. If reviewers also wrongly rejected 25 of 500 reviewed correct outputs, IIR = 0.05.

**Reporting requirements.**

1. EHOR with a confidence interval, stratified by severity, alongside IIR.
2. EHOR_c and Cov_R separately.
3. The audit design, the inclusion probabilities, and inter-rater agreement between adjudicators.
4. The count and fraction of sampled outputs that could not be adjudicated, with upper and lower bounds that treat them as all intercepted or all missed.

## 8. Agentic assurance trace

An agentic system executes a trajectory rather than producing a single output. The instrument represents it as

`tau = (I, R, P, X, A, O)`

for input, retrieval, planning or reasoning, tool execution, action, and observation. The governance system must be able to reconstruct tau for any decision (indicator L3). Each step carries its own risk.

| Step | Governance risk | Indicators |
|---|---|---|
| I, input | Out-of-scope or corrupted input | C1, C2, C3, C4, C5 |
| R, retrieval | Irrelevant or missing evidence | G3, G1 |
| P, planning or reasoning | Unsupported or failed reasoning | G2, G5 |
| X, tool execution | Unauthorized or failing tool use | G6, G5, L2 |
| A, action | Unsafe or unapproved action | G4, G6, H6, H7 |
| O, observation and outcome | Unrecovered failure; downstream harm | G5, O5, O6 |

Example trajectory: a patient question leads to record retrieval, interpretation of the condition, a search of clinical knowledge, a generated recommendation, a scheduled action, and a message to the patient. A failure at any step can invalidate a later step that looks correct, which is why G5 monitors recovery as well as failure.

## 9. Hypotheses, indicators, and comparators

| Research hypothesis (plan H1-H5) | Indicators tested | Comparator | Primary readout |
|---|---|---|---|
| RH1, continuous monitoring detects risk earlier | All 41, combined by the routing rule in section 4.2 (the 34 predictive-applicable indicators in Phase II) | Conventional monitoring on P1, P2, P4 | TAFD at matched false-alarm rate |
| RH2, calibration deteriorates before aggregate performance | U1, U2, U3 (label-dependent). U5, which is label-free, is reported separately | P1, P2, P4 | Lead time in TAFD under calibration drift at stable AUROC |
| RH3, effective oversight adds information beyond documented review | H1, H4, H5 with IIR | Documented presence of human review | Incremental explanation of missed-risk events |
| RH4, workflow indicators find failures that model metrics miss | O1 to O6, G4, G5 | P and U indicators only | Failures detected only by workflow indicators |
| RH5, risk-triggered interventions reduce persistence or severity | L1, L4, H7 and the `governance_action` fields | No predefined intervention | Persistence and severity of deviations after trigger (Phase III) |

## 10. Multiplicity and the false-alarm budget

Forty indicators monitored across many windows will produce false alarms unless the design controls them. The instrument uses four controls: false-discovery-rate control within each component, the corroboration rule and hysteresis in section 4.2, matched false-alarm calibration when comparing systems (section 7.1), and reporting of false alarms per 1,000 monitored windows for every configuration. The false-alarm budget is set in the VOE from the reviewer workload the application can absorb, because an alert nobody acts on has no safety value (O3).

## 11. Threats to validity

- Phase II is retrospective simulation. It is not prospective clinical deployment, and the two must be kept distinct in every report.
- Injected drift may be less realistic than natural drift. Scenarios should include real temporal shifts where data allow.
- Simulated reviewer behavior (bias, fatigue) rests on assumptions about human behavior that are not measured here.
- Outcome lag and adjudication cost limit how often levels 2, 3, and 5 can be measured. Label-free indicators (U5, C2) help but cannot replace outcome-based checks.
- Margins and severity cutoffs involve clinical judgment. Sensitivity analyses should vary them.
- Results from one site or one clinical task may not generalize. The worked example is one task, not evidence of coverage across tasks.
- Level 5 evidence is scarce (2 of 41 indicators), so claims of patient benefit need separate prospective evidence.
- Evidence transparency is not measured for predictive AI (section 4.1), so the transparency principle is covered only for generative and agentic systems, through groundedness.

## 12. Methods and sources: citation status

This instrument contains one inline citation, to the project lead's related work on operational trajectories [yu2026operational], and no others. Under the project's citation rules, no reference is written from memory. Phase I (`phase1/Phase-I-Synthesis.md`, section 5) verified the existence and metadata of a candidate reference for most methods below against CrossRef, OpenAlex, Semantic Scholar, DataCite, or a publisher page. Verification does not confirm that a reference supports the specific claim made here. Claim-level checks against full text are still required before any manuscript sentence cites them. Reference keys are in `phase1/references.bib`.

| Method or source | Status |
|---|---|
| Maximum mean discrepancy; Wasserstein (earth mover's) distance | Candidates verified (gretton2012, one source only; rubner2000) |
| False-discovery-rate control; Holm sequential rejection | Candidates verified (benjamini1995, holm1979) |
| CUSUM; EWMA; sequential probability ratio test | Candidates verified (page1954, roberts1959, wald1945) |
| Expected calibration error; Brier score; calibration slope and intercept | Candidates verified (naeini2015, brier1950, steyerberg2010) |
| Bootstrap; block bootstrap; DeLong comparison of AUROC | Candidates verified (efron1979, kunsch1989, delong1988) |
| Horvitz-Thompson estimation | Candidate verified (horvitz1952) |
| Kaplan-Meier; Cox regression; restricted mean survival time | Candidates verified (kaplan1958, cox1972, royston2013) |
| Decision curve analysis | Candidate verified (vickers2006) |
| NIST AI Risk Management Framework 1.0 | Verified (nist2023) and read in full text |
| FDA guidance on predetermined change control plans | Read in full text; issued August 18, 2025, originally December 4, 2024. It has no DOI and is cited by document. |
| EU AI Act, Articles 14 and 72 | Text read from an unofficial mirror only. `[CITATION NEEDED]` for the official source. |
| Population stability index | `[CITATION NEEDED]`. No source has been searched. |
| Kolmogorov-Smirnov test | `[CITATION NEEDED]`. Not searched. |
| IMDRF GMLP principles; FDA, Health Canada, and MHRA GMLP principles; NIST AI 600-1; WHO guidance (2021 full text and large multimodal models) | `[CITATION NEEDED]`. Not retrieved in Phase I. |

## 13. Reproducibility

Edit the records in `indicators/*.yaml`, then run:

```
python scripts/validate_indicators.py --render
```

The script checks that all 41 records are complete and consistent with `indicators/schema.yaml`, and regenerates `Indicator-Instrument.md`. The counts in sections 4 and 6 of this document were computed from the records on 2026-09-19 and must be refreshed if the records change.

Related artifacts, each generated or checked by a script:

- Phase I: `phase1/Phase-I-Synthesis.md`, `phase1/Assurance-Control-Matrix.md` (`scripts/validate_requirements.py`), `phase1/term_coverage.md` (`scripts/term_coverage.py`), `phase1/Citation-Audit-Report.md` and `phase1/references.bib` (`scripts/verify_citations.py`).
- Phase II: `phase2/Phase-II-Simulation-Design.md`, `phase2/scenarios.yaml`, and `phase2/Scenario-Coverage.md` (`scripts/validate_scenarios.py`).
