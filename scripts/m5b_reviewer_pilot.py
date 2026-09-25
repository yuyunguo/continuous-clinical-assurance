"""Milestone M5b pilot: can oversight indicators (H1 to H6) detect the reviewer failures SC10, SC11, and SC13?

Usage: python scripts/m5b_reviewer_pilot.py
Writes phase2/M5b-Reviewer-Pilot.md. Exploratory, one seed, Tier 1 synthetic data. Reviewer parameters and the point of
no return (deadline of 5 median review times) are assumptions. In these scenarios the model does not change, so the
conventional monitor is blind by construction (design section 9.1, interpretation limit): the question is speed and false-alarm cost.
"""

import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_module.tafd import classify_replicate  # noqa: E402
from src.monitor_module import (INDICATOR_ORDER, build_stream, calibrate_threshold, episode_starts, progress_schedule,  # noqa: E402
                                stream_scores_multi, system_events)
from src.monitor_module.mixture import WindowRatio  # noqa: E402
from src.monitor_module.review_layer import REVIEW_ORDER, build_review, review_scores_multi  # noqa: E402
from src.sim_module import DeployedModel, SyntheticWorld, load_config, parse_level  # noqa: E402
from src.sim_module.reviewer_schedule import reviewer_schedule  # noqa: E402
from src.sim_module.config import false_alarm_budget  # noqa: E402
from src.utils import log_environment, set_seed  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "phase2" / "M5b-Reviewer-Pilot.md"
N_NULL, REPLICATES, LAGS, AUDIT, DEADLINE, COVERAGE = 800, 150, (0, 8), 0.2, 5.0, 0.95
BUDGET = false_alarm_budget()
K_MODEL, K_ALL = len(INDICATOR_ORDER), len(INDICATOR_ORDER) + len(REVIEW_ORDER)
ARMS: Dict[str, List[int]] = {"conventional (P, U, C)": list(range(K_MODEL)), "H1 only (documented review)": [K_MODEL],
                              "oversight (H1 to H6)": list(range(K_MODEL, K_ALL)), "conventional + H1": list(range(K_MODEL)) + [K_MODEL],
                              "conventional + oversight": list(range(K_ALL))}
MAIN = list(ARMS)
ARMS.update({name: [K_MODEL + k] for k, name in enumerate(REVIEW_ORDER)})   # each oversight indicator alone, for attribution
CELLS = [("SC10", "low", "abrupt"), ("SC10", "medium", "abrupt"), ("SC10", "high", "abrupt"), ("SC10", "high", "gradual"),
         ("SC11", "low", "abrupt"), ("SC11", "medium", "abrupt"), ("SC11", "high", "abrupt"), ("SC11", "high", "gradual"),
         ("SC13", "low", "abrupt"), ("SC13", "medium", "abrupt"), ("SC13", "high", "abrupt"), ("SC13", "high", "gradual")]


def stream_scores_all(stream: Any, cfg: Any, kind: Optional[str], level: float, progress: np.ndarray, rng: np.random.Generator) -> Dict[int, np.ndarray]:
    """Model-level and oversight scores side by side, per lag. Shape (T, 18)."""
    model_z = stream_scores_multi(stream, cfg, LAGS)
    review_z = review_scores_multi(stream, build_review(stream, cfg, kind, level, progress, rng), AUDIT, LAGS, rng)
    return {lag: np.concatenate([model_z[lag], review_z[lag]], axis=1) for lag in LAGS}


def null_scores(world: SyntheticWorld, model: DeployedModel, cfg: Any, seed: int) -> Dict[int, np.ndarray]:
    """Scores of N_NULL independent null streams, per lag. Shape (S, T, 18)."""
    zeros = np.zeros(cfg.stream.monitored_windows)
    per = [stream_scores_all(s, cfg, None, 0.0, zeros, np.random.default_rng(seed * 7919 + i))
           for i in range(N_NULL) for s in [build_stream(world, model, None, "abrupt", 0, cfg, np.random.default_rng(seed * 100_003 + i))]]
    return {lag: np.stack([p[lag] for p in per]) for lag in LAGS}


def ratio_at(ratio: WindowRatio, cfg: Any, kind: str, level: float, progress: float) -> float:
    """Expected RHE relative to baseline when the reviewer is at the given progress of the perturbation."""
    return ratio({k: float(v[0]) for k, v in reviewer_schedule(kind, level, np.array([progress]), cfg.reviewer).items()})


def main() -> int:
    """Run the pilot and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg0 = load_config()
    cfg = replace(cfg0, reviewer=replace(cfg0.reviewer, deadline=DEADLINE, coverage=COVERAGE))
    set_seed(cfg.seed)
    world = SyntheticWorld(cfg.generator, cfg.harm)
    model = DeployedModel(cfg.generator).fit(world, np.random.default_rng(cfg.seed))
    j, horizon, total = cfg.routing.hysteresis_j, cfg.stream.horizon, cfg.stream.monitored_windows
    scenarios = {s["id"]: s for s in yaml.safe_load((ROOT / "phase2" / "scenarios.yaml").read_text(encoding="utf-8"))}
    calib, evaluation = null_scores(world, model, cfg, 101), null_scores(world, model, cfg, 202)
    thr = {lag: {a: calibrate_threshold(calib[lag], cols, BUDGET, j) for a, cols in ARMS.items()} for lag in LAGS}
    cal_rows = []
    for lag in LAGS:
        for a, cols in ARMS.items():
            starts = episode_starts(system_events(evaluation[lag], cols, thr[lag][a]), j)
            cal_rows.append(f"| {lag} | {a} | {thr[lag][a]:.2f} | {starts.sum() / starts.size * 1000:.2f} |")
    pop = model.apply(world.sample(200_000, np.random.default_rng(11)))
    ratio = WindowRatio(cfg, pop)
    tables: Dict[int, List[str]] = {lag: [] for lag in LAGS}
    single: Dict[int, List[str]] = {lag: [] for lag in LAGS}
    for ci, (sid, level, shape) in enumerate(CELLS):
        spec = scenarios[sid]["perturbation"]
        value, kind = parse_level(spec["levels"][level]), spec["type"]
        full = ratio_at(ratio, cfg, kind, value, 1.0)
        lo, hi = cfg.stream.s0_low, (cfg.stream.s0_high_abrupt if shape == "abrupt" else cfg.stream.s0_high_ramped)
        rng_s0 = np.random.default_rng(3000 + ci)
        cache: Dict[float, float] = {}
        res: Dict[int, Dict[str, List[Tuple[int, bool]]]] = {lag: {a: [] for a in ARMS} for lag in LAGS}
        for r in range(REPLICATES):
            s0 = int(rng_s0.integers(lo, hi + 1))
            progress = progress_schedule(shape, s0, total, cfg.stream.ramp_windows)
            for p in set(np.round(progress[s0:], 6)):
                if p not in cache:
                    cache[p] = ratio_at(ratio, cfg, kind, value, float(p))
            onset = [t for t in range(s0, total) if cache[round(float(progress[t]), 6)] >= 1 + cfg.harm.kappa]
            if not onset or onset[0] + horizon > total:
                continue
            rng = np.random.default_rng(600_000 + ci * 1000 + r)
            zs = stream_scores_all(build_stream(world, model, None, shape, s0, cfg, rng), cfg, kind, value, progress, rng)
            for lag in LAGS:
                for a, cols in ARMS.items():
                    out = classify_replicate(system_events(zs[lag][None], cols, thr[lag][a])[0], s0, onset[0], horizon, j)
                    res[lag][a].append((int(out["tafd"]), not out["censored"]))
        n = len(res[LAGS[0]]["conventional (P, U, C)"])
        logger.info("cell %s %s %s: onset ratio at full %.2f, %d replicates", sid, level, shape, full, n)
        for lag in LAGS:
            fmt = lambda v: f"{np.mean([t for t, _ in v]):.1f} ({np.mean([d for _, d in v]):.0%})" if n else ""  # noqa: E731
            tables[lag].append(f"| {sid} | {level} | {shape} | {full:.2f} | {n} | " + " | ".join(fmt(res[lag][a]) for a in MAIN) + " |")
            single[lag].append(f"| {sid} | {level} | {shape} | " + " | ".join(fmt(res[lag][a]) for a in REVIEW_ORDER) + " |")
    header = "| Scenario | Level | Shape | Onset ratio at full effect | n | " + " | ".join(MAIN) + " |"
    lines = ["# Phase II M5b reviewer-failure pilot (exploratory)", "",
             "Generated by `scripts/m5b_reviewer_pilot.py`. Detection of SC10 (automation bias), SC11 (review bypass), and SC13 (queue delay) by oversight "
             f"indicators H1 to H6 against the conventional monitor, at a matched false-alarm budget of {BUDGET:g} per 1,000 windows (each arm calibrated on {N_NULL} null "
             f"streams and checked on {N_NULL} independent ones). Baseline review coverage {COVERAGE:.0%} (so that H1 has realistic noise). Point of no return: {DEADLINE:g} median review times. H5 uses a {AUDIT:.0%} audit of cases. The model does not change in these "
             "scenarios, so the conventional monitor cannot respond by construction (design section 9.1). This shows what the oversight layer can detect, how "
             "quickly, and at what false-alarm cost. It does not show that such failures occur in practice. Reviewer settings are assumptions. "
             f"Environment: {log_environment()}", "", "## Calibration", "", "| Lag | Arm | Threshold | Achieved per 1,000 windows |", "|---|---|---|---|"] + cal_rows + [""]
    lines += ["## Mean TAFD in windows (share detected within the horizon)", "",
              f"Undetected replicates count as the horizon of {horizon}. Onset is harm-defined (expected RHE at 1 + kappa of baseline); cells whose full-effect ratio is below "
              "1 + kappa have no onset and are dropped (n = 0).", ""]
    for lag in LAGS:
        lines += [f"### Label lag {lag} windows", "", header, "|" + "---|" * (5 + len(MAIN))] + tables[lag] + [""]
    lines += ["## Attribution: each oversight indicator alone", "", "Mean TAFD in windows (share detected), each indicator calibrated alone to the same budget.", ""]
    for lag in LAGS:
        lines += [f"### Label lag {lag} windows", "", "| Scenario | Level | Shape | " + " | ".join(REVIEW_ORDER) + " |", "|" + "---|" * (3 + len(REVIEW_ORDER))] + single[lag] + [""]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", OUTPUT.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
