# One follow-up question for decision D2 (the false-alarm budget)

## Purpose

This follow-up asks physicians to estimate how many false-alarm reviews a governance process could absorb for one AI tool over a year before people stop taking alerts seriously.



## D2. False-alarm budget

### 1. How many false-alarm reviews could one AI tool absorb over a year before people stop taking alerts seriously?

**Best guess:** About **one a month**.

**Plausible range:** About **one every six months to more often than monthly**.

**Primary operational assumption:** **12 false-alarm reviews per AI tool-year.**

For sensitivity analysis, evaluate the framework across a range of false-alarm budgets:

- **Low budget:** 2 false alarms per tool-year (about one every six months)
- **Moderate budget:** 4 false alarms per tool-year (about one every quarter)
- **Primary budget:** 12 false alarms per tool-year (about one per month)
- **High budget:** 24 false alarms per tool-year (about two per month / more often than monthly)

The study should report whether conclusions remain stable across this range rather than relying on a single false-alarm budget.

### 2. Does this number apply to one AI tool or several tools sharing the same governance process?

**Primary analysis:** One AI tool.

**Organizational sensitivity analysis:** Several AI tools sharing the same governance process, using approximately **5–10 tools** as a sensitivity scenario.

Where multiple tools share a common governance process, the study should distinguish between:

1. **Tool-level false-alarm budget** — the number of false-alarm reviews attributable to an individual AI tool.
2. **Governance-process budget** — the total false-alarm burden experienced by the people responsible for monitoring multiple AI tools.

This distinction is important because alert fatigue may depend on the aggregate burden placed on the governance team, not only on the false-alarm rate of an individual tool.

## Conversion to monitoring windows

The study converts the selected false-alarm budget into a rate per **1,000 monitoring windows**, where a monitoring window is **7–14 days** at the previously specified monitoring volumes.

The primary analysis uses **12 false alarms per AI tool-year** as the provisional false-alarm budget.

Results should also be reported across the sensitivity budgets of **2, 4, 12, and 24 false alarms per tool-year**.

## Interpretation

The false-alarm budget should be treated as an **operational design parameter** for evaluating governance monitoring approaches, not as a universal clinical threshold.

The final D2 value should ideally be derived from the actual physician responses collected by this elicitation. Until those responses are available, the values above should be explicitly labeled as **investigator assumptions / sensitivity-analysis scenarios**.
