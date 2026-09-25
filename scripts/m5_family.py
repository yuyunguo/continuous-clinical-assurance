"""Milestone M5 rehearsal: the four pre-specified tests and the Holm family (design section 9.1).

Usage: python scripts/m5_family.py
Needs `phase2/results/m5_rehearsal.npz` (RH1, RH2), `rh3.json`, `rh4_short.json` (RH4 with detection within 4 windows, decision D25), and `rh4.json` (the 40-window version, reported beside it). Writes phase2/M5-Family-Rehearsal.md.
A rehearsal, not a confirmatory result: D3, D4, D23, and the window duration are not fixed, and the pilots were seen first.
The primary label lag is 2 windows (decision D24, accepted 2026-09-20, from D20's "at least two windows"). Lags 0 and 8 are reported beside it.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_module.holm import holm  # noqa: E402
from src.eval_module.pooled import collect_diffs, pooled_bootstrap  # noqa: E402
from src.sim_module.config import false_alarm_budget, results_dir  # noqa: E402
from src.utils import log_environment  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
RESULTS = results_dir()
OUTPUT = RESULTS / "M5-Family-Rehearsal.md" if "PHASE2_RESULTS" in os.environ else ROOT / "phase2" / "M5-Family-Rehearsal.md"
RH1_POOL, RH2_POOL = ("SC07", "SC08", "SC09", "SC17", "SC19", "SC25"), ("SC08", "SC17", "SC25")
LAGS, PRIMARY_LAG, ALPHA = (0, 2, 8), 2, 0.05
BUDGET = false_alarm_budget()
PRIMARY_MARGIN = 2.0                 # windows (decision D23: delta = 1 baseline-window of harm)
MARGINS = (0.5, 1.0, PRIMARY_MARGIN)  # 1 window is one governance cycle at the elicited volume, and 0.5 is a further sensitivity


def main() -> int:
    """Compute the tests and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    store = dict(np.load(RESULTS / "m5_rehearsal.npz"))
    base = float(store["baseline_rhe"][0])
    rh3, rh4_long, rh4 = (json.loads((RESULTS / name).read_text()) for name in ("rh3.json", "rh4.json", "rh4_short.json"))
    rows, family = [], {}
    detail = ["| Lag | RH1 reduction [95% CI] | RH1 exposure avoided [95% CI] | RH2 reduction [95% CI] |", "|---|---|---|---|"]
    per_scenario = ["| Lag | " + " | ".join(RH1_POOL) + " | Cells in favor (CI above 0) | Cells inferior (CI below 0) |", "|" + "---|" * (len(RH1_POOL) + 3)]
    for lag in LAGS:
        rng = np.random.default_rng(7 + lag)
        rh1 = collect_diffs(store, RH1_POOL, lag, "conventional", "CCA (component split)")
        rh2 = collect_diffs(store, RH2_POOL, lag, "conventional", "calibration arm")
        exposure = collect_diffs(store, RH1_POOL, lag, "avoided", None)
        p1 = {m: pooled_bootstrap(rh1, m, rng) for m in MARGINS}
        cell_ci = [pooled_bootstrap({"c": [cell]}, 0.0, rng, n_boot=1000)[1:3] for cells in rh1.values() for cell in cells]
        by_scenario = [f"{np.mean([c.mean() for c in rh1[s]]):+.1f}" if rh1[s] else "n/a" for s in RH1_POOL]
        per_scenario.append(f"| {lag} | " + " | ".join(by_scenario) + f" | {sum(lo > 0 for lo, _ in cell_ci)} of {len(cell_ci)} | {sum(hi < 0 for _, hi in cell_ci)} of {len(cell_ci)} |")
        p_exposure = pooled_bootstrap(exposure, base, rng)          # delta = 1 baseline-window of harm (decision D23); the unit is already baseline-windows
        rh1_full = collect_diffs(store, RH1_POOL, lag, "conventional", "CCA (all testable)")
        p_full = pooled_bootstrap(rh1_full, PRIMARY_MARGIN, rng)
        p2 = pooled_bootstrap(rh2, 0.0, rng)
        ps = {f"RH1 (margin {PRIMARY_MARGIN:g} windows)": p1[PRIMARY_MARGIN][3], "RH2": p2[3], "RH3": rh3[str(lag)]["p"], "RH4 (detection within 4 windows, D25)": rh4[str(lag)]["p"]}
        adjusted = holm(ps, ALPHA)
        family[lag] = adjusted
        detail.append(f"| {lag} | {p1[1.0][0]:+.2f} [{p1[1.0][1]:+.2f}, {p1[1.0][2]:+.2f}] | {p_exposure[0]:+.2f} [{p_exposure[1]:+.2f}, {p_exposure[2]:+.2f}] | "
                      f"{p2[0]:+.2f} [{p2[1]:+.2f}, {p2[2]:+.2f}] |")
        for name, (adj, reject) in adjusted.items():
            rows.append(f"| {lag}{' (primary)' if lag == PRIMARY_LAG else ''} | {name} | {ps[name]:.4f} | {adj:.4f} | {'reject' if reject else 'not rejected'} |")
        window_ok, harm_ok = adjusted[f"RH1 (margin {PRIMARY_MARGIN:g} windows)"][1], p_exposure[3] <= ALPHA
        verdict = "consistent" if window_ok == harm_ok else "FRAGILE: the window test and the harm-unit test disagree"
        rows.append(f"| {lag}{' (primary)' if lag == PRIMARY_LAG else ''} | RH1 sensitivity: margin 0.5 / 1 window | {p1[0.5][3]:.4f} / {p1[1.0][3]:.4f} | not in family | |")
        rows.append(f"| {lag}{' (primary)' if lag == PRIMARY_LAG else ''} | RH4 sensitivity: 40-window horizon, {rh4_long[str(lag)]['estimate']:+.3f} (ceiling {np.exp(-BUDGET * 40 / 1000):.2f}) | {rh4_long[str(lag)]['p']:.4f} | not in family | |")
        rows.append(f"| {lag}{' (primary)' if lag == PRIMARY_LAG else ''} | RH1 sensitivity: CCA over all testable indicators, {p_full[0]:+.2f} [{p_full[1]:+.2f}, {p_full[2]:+.2f}] windows | {p_full[3]:.4f} | not in family | |")
        rows.append(f"| {lag}{' (primary)' if lag == PRIMARY_LAG else ''} | RH1 in harm units (exposure avoided against delta = 1 baseline-window) | {p_exposure[3]:.4f} | not in family | {verdict} |")
    lines = ["# Phase II M5 rehearsal: the four tests and the Holm family", "",
             "Generated by `scripts/m5_family.py`. **A rehearsal, not a confirmatory result.** D3, D4, D23, and the window duration are not fixed, and the pilots were seen before "
             "these seeds were chosen. RH1 and RH2 come from `scripts/m5_rehearsal.py` (200 replicates per cell, fresh seeds, fresh calibration null streams), RH3 from `M5c-RH3-Mixture.md` "
             "(equal mix, 1,000 fit and 1,000 evaluation streams), and RH4 from `M5e-RH4-Workflow.md` (200 replicates per cell, margin 0.20). Each hypothesis has one pooled estimand and a "
             f"one-sided bootstrap over replicates; Holm adjustment at a family-wise level of {ALPHA:g} over the four p-values, separately at each label lag. The primary lag is "
             f"{PRIMARY_LAG} windows (decision D24). A p-value of 0 means no bootstrap draw reached the null. The RH1 reduction is in windows against the D23 margin of 2 windows (delta = 1 baseline-window of harm), with 1 window (one governance cycle at the elicited volume) and 0.5 as sensitivity. RH1 is also tested in harm units (exposure avoided against one baseline-window of harm). It is outside the family, and if it disagrees with the window test at the same lag, RH1 is reported as fragile to the scale. "
             f"Environment: {log_environment()}", "",
             "## Holm family", "", "| Label lag | Test | p (one-sided) | Holm-adjusted p | Decision at 0.05 |", "|---|---|---|---|---|"] + rows + ["", "## Pooled estimates", ""] + detail + ["",
             "## RH1 by scenario and the supplementary criterion", "",
             "Mean reduction in windows per scenario (cells averaged), and the number of cells whose own 95 percent interval lies above zero (in favor of CCA) or below zero (inferior). "
             "The design's supplementary criterion, which is not error-controlled, asks for at least two thirds of cells in favor and fewer than one third inferior. The pooled mean gives "
             "each scenario equal weight, so one scenario can carry it.", ""] + per_scenario + [""]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", OUTPUT.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
