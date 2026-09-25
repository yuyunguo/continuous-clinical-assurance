"""Milestone M6 (first pass): EHOR estimator study (design section 10) under the reviewer scenarios SC10 and SC11.

Usage: python scripts/m6_ehor_study.py
Compares the design-weighted audit estimator with the naive comparator for audit fractions 1, 5, and 20 percent,
with and without stratification. Writes phase2/M6-EHOR-Study.md. Exploratory, Tier 1 synthetic data only.
"""

import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_module.ehor import audit_sample, ehor_ht, ehor_ht_se, ehor_truth, iir_ht, naive_ehor  # noqa: E402
from src.sim_module import DeployedModel, SyntheticWorld, load_config  # noqa: E402
from src.sim_module.review import sample_review  # noqa: E402
from src.sim_module.config import report_path  # noqa: E402
from src.utils import log_environment, set_seed  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
N_CASES = 20_000          # cases per replicate (about 20 windows of 1,000)
REPLICATES = 300
FRACTIONS = (0.01, 0.05, 0.20)
DISCOVERY = 0.3           # share of missed errors that later surface through outcome reports (assumption)
CONDITIONS: List[Tuple[str, Dict[str, float]]] = [
    ("baseline", {}), ("SC10 low (+10 pct)", {"automation_bias": 0.10}), ("SC10 medium (+25 pct)", {"automation_bias": 0.25}),
    ("SC10 high (+40 pct)", {"automation_bias": 0.40}), ("SC11 low (90 pct coverage)", {"coverage": 0.9}),
    ("SC11 medium (70 pct)", {"coverage": 0.7}), ("SC11 high (40 pct)", {"coverage": 0.4}),
]


def run_condition(world: SyntheticWorld, model: DeployedModel, cfg: Any, seed: int) -> Dict[str, Any]:
    """Estimator bias, RMSE, and interval coverage over replicates, for one reviewer condition."""
    rng = np.random.default_rng(seed)
    truths: List[float] = []
    naive: List[float] = []
    arms: Dict[Tuple[float, bool], List[Tuple[float, float]]] = {(f, s): [] for f in FRACTIONS for s in (False, True)}
    iir_err: List[float] = []
    for _ in range(REPLICATES):
        batch = model.apply(world.sample(N_CASES, rng))
        out = sample_review(batch, cfg, rng)
        truth = ehor_truth(out).ehor
        truths.append(truth)
        naive.append(naive_ehor(out, DISCOVERY))
        for (fraction, stratified), store in arms.items():
            _, w = audit_sample(out, batch, fraction, stratified, rng)
            store.append((ehor_ht(out, w) - truth, ehor_ht_se(out, w)))
            if fraction == 0.05 and stratified:
                iir_err.append(iir_ht(out, w) - iir_ht(out, np.ones(batch.n)))
    t = np.asarray(truths)
    row: Dict[str, Any] = {"truth": float(t.mean()), "naive_bias": float(np.mean(np.asarray(naive) - t)),
                           "naive_rmse": float(np.sqrt(np.mean((np.asarray(naive) - t) ** 2))), "iir_bias": float(np.mean(iir_err))}
    summary: Dict[Tuple[float, bool], Tuple[float, float, float]] = {}
    for key, store in arms.items():
        err = np.array([e for e, _ in store])
        se = np.array([s for _, s in store])
        summary[key] = (float(err.mean()), float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err) <= 1.96 * se)))
    row["arms"] = summary
    return row


def render(rows: Dict[str, Dict[str, Any]]) -> str:
    """Markdown report."""
    lines = ["# M6 (first pass): EHOR estimator study (exploratory)", "",
             f"Tier 1 synthetic data, {REPLICATES} replicates of {N_CASES:,} cases per condition. Truth is the realized EHOR of each replicate, "
             f"so bias and coverage are design-based. Naive comparator: intercepted errors over intercepted plus {DISCOVERY:.0%} of missed errors "
             "(an assumed discovery rate). Stratification is by displayed confidence, oversampling low-confidence outputs 6-fold against high. "
             "Reviewer settings are assumptions (design section 11).", "",
             "| Condition | Mean true EHOR | Naive bias | Naive RMSE | Audit % | Stratified | HT bias | HT RMSE | 95% CI coverage |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        for (fraction, stratified) in [(f, s) for f in FRACTIONS for s in (False, True)]:
            bias, rmse, cover = r["arms"][(fraction, stratified)]
            lines.append(f"| {name} | {r['truth']:.3f} | {r['naive_bias']:+.3f} | {r['naive_rmse']:.3f} | {fraction:.0%} | {'yes' if stratified else 'no'} | {bias:+.4f} | {rmse:.4f} | {cover:.2f} |")
    worst_iir = max(abs(r["iir_bias"]) for r in rows.values())
    lines += ["", f"IIR (5 percent stratified audit) bias, worst condition: {worst_iir:.4f}.", ""]
    return "\n".join(lines)


def main() -> None:
    """Run every reviewer condition and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    set_seed(cfg.seed)
    log_environment()
    world = SyntheticWorld(cfg.generator)
    model = DeployedModel(cfg.generator).fit(world, np.random.default_rng(cfg.seed))
    rows: Dict[str, Dict[str, Any]] = {}
    for i, (name, kw) in enumerate(CONDITIONS):
        rows[name] = run_condition(world, model, replace(cfg, reviewer=replace(cfg.reviewer, **kw)), cfg.seed + 1000 + i)
        logger.info("%s done", name)
    output = report_path("M6-EHOR-Study.md")
    output.write_text(render(rows), encoding="utf-8")
    logger.info("Wrote %s", output)


if __name__ == "__main__":
    main()
