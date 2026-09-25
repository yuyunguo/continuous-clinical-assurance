# Indicator Instrument: Design Spec

Date: 2026-09-19
Status: approved for implementation (approach B; proofread revisions applied)
Sources: `Continuous_Clinical_Assurance_Research_Plan.md` (sections 4-6, 9-11, 17), `Evidence-and-measurement-matrix.md` (sections 2-10)

## 1. Purpose

Convert the Clinical AI Assurance Evidence and Measurement Matrix into a formal research instrument of 41 candidate indicators. Each indicator is specified precisely enough that Phase II (perturbation simulation) and Phase III (intervention experiment) can implement and test it without re-deriving definitions.

The instrument is technology-agnostic. Generative and agentic indicators are tagged with `ai_types`, not placed in a separate domain. One worked example (a calibrated binary risk model) shows thresholds and detection concretely.

## 2. Record schema

Each indicator is one YAML record. Controlled vocabularies live in `indicators/schema.yaml`.

| Field | Content |
|---|---|
| `id`, `name` | Stable identifier (e.g., `U1`) and name |
| `domain` | Plan domain: A Clinical Context, B Performance and Uncertainty, C Human Oversight, D Operational and Workflow, E Lifecycle |
| `state_component` | Component of A_t = (C_t, P_t, U_t, H_t, O_t, L_t) |
| `matrix_construct` | The measurable construct of `Evidence-and-measurement-matrix.md` that the indicator operationalizes. It carries the governance objective and clinical risk. |
| `evidence_level` | 1 system, 2 performance, 3 human interaction, 4 workflow, 5 clinical outcome |
| `ai_types` | Subset of predictive, generative, agentic |
| `definition` | Operational definition |
| `formula` | Mathematical formulation |
| `data_required` | Data elements needed |
| `frequency` | Measurement cadence |
| `baseline` | How the baseline is fixed within the Validated Operating Envelope (VOE) |
| `alert_rule` | Trigger logic using VOE-specified margins m_Y < m_O < m_R |
| `detection_method` | Statistical detection method |
| `governance_action` | Actions for yellow, orange, and red |
| `clinical_importance` | Why a deviation matters clinically |
| `validation` | How Phase II/III will test the indicator |

## 3. Trigger architecture

Green means within the validated envelope, with routine monitoring and no action field. Yellow is an early warning (increase monitoring or human oversight). Orange is a substantial deviation (restrict use, investigate, revalidate). Red is a critical safety signal (suspend or substantially restrict the function pending investigation).

Thresholds are not universal. Each application sets its margins in the VOE from risk and validated operating characteristics. The instrument gives the rule form. Numbers appear only in the labeled illustrative example.

## 4. Definitions decided during proofreading

**State component versus domain.** The plan has five domains, but the state vector has six components because Domain B splits into P and U. Every record carries both tags.

**EHOR (Effective Human Oversight Rate), revised.** Decided after review. The full definition is in record H5 and in `Instrument-Specification.md` section 7. In brief: EHOR is the design-weighted proportion of AI errors requiring human detection (adjudicated blind to reviewer action, severity at or above the VOE cutoff s*) that were intercepted before the workflow's point of no return. The denominator includes errors that bypassed review or were reviewed too late. Estimation uses an independent probability-sample audit, because counting only errors found through review inflates EHOR. EHOR is always reported with the inappropriate-intervention rate (IIR), since rejecting every output would otherwise give EHOR = 1. It decomposes as EHOR = Cov_R x EHOR_c, where EHOR_c is the conditional variant and is always labeled.

**TAFD (Time to Assurance Failure Detection), renamed.** Decided after review. The plan's "Time to Assurance Failure (TAF)" defines a detection delay, but the name reads as time until failure occurs. The metric is now Time to Assurance Failure Detection (TAFD), and the plan's primary-outcome label "Time to Detection" is the same quantity. TAFD is analyzed as a right-censored time-to-event outcome. The plan document itself is unchanged and still uses the old name; `Instrument-Specification.md` records the mapping.

**Trace notation.** The matrix wrote the agentic trace as {I, R, P, T, A, O}, with T used twice. It is now tau = (I, R, P, X, A, O), where X is tool execution.

## 5. Worked example (illustrative, not empirical)

A calibrated binary risk model with baseline ECE 0.04 and AUROC stable. Simulated calibration drift raises ECE to 0.11 while AUROC stays within its validated range. Conventional monitoring reports acceptable performance. The instrument raises yellow on U1 and U3 (increase human review), orange if drift persists (recalibration and revalidation), and red if linked to clinically consequential errors (restrict or suspend the use case). Values are for demonstration only.

## 6. Files

- `indicators/schema.yaml`: vocabularies and required fields.
- `indicators/domain_*.yaml`: the 41 records in six files (Domain B is split into `domain_B_performance.yaml` and `domain_B_uncertainty.yaml`). Domain counts: A 5, B 15, C 7, D 8, E 6.
- Generative and agentic indicators (G1-G6) sit in the domain of the risk they address: G1-G3 in B, G4-G5 in D, G6 in E. They carry `ai_types` tags, not a domain of their own.
- `Indicator-Instrument.md`: generated output; edit the YAML, not this file.
- `scripts/validate_indicators.py`: checks completeness and vocabulary, renders `Indicator-Instrument.md`.
- `Instrument-Specification.md`: the approach-C write-up (VOE, assurance state, trigger architecture, evidence hierarchy, TAFD, EHOR, agentic trace, hypothesis map).

Later work built on this instrument: `phase1/` (governance-source synthesis and verified references) and `phase2/` (simulation design and scenario catalog). Their scripts are listed in `Instrument-Specification.md` section 13.

Naming note: the plan's research hypotheses H1-H5 are written RH1-RH5 in the instrument documents, because H1-H7 are the human-oversight indicator IDs.

## 7. Constraints and non-goals

- No inline citations in the instrument, except one to the project lead's related work on operational trajectories. Method sources are added in Phase I after verification against CrossRef, OpenAlex, or Semantic Scholar. Unverified sources are marked `[CITATION NEEDED]`.
- No single composite governance score. A_t stays multidimensional.
- No claim that these indicators are validated. They are candidates until Phase II/III.

## 8. Verification

The validator must pass with zero errors: 41 records, unique IDs, every required field present, every vocabulary value valid, and yellow/orange/red present in every `governance_action`.
