"""Milestone M5 rehearsal: RH1 and RH2 cells with per-replicate storage on fresh seeds.

Usage: python scripts/m5_rehearsal.py
Runs the fair-test cells behind RH1 (SC07, SC08, SC09, SC17, SC19, SC25) and RH2 (SC08, SC17, SC25) for the conventional arm, the CCA component split (12 model-level
indicators in components C, P, U), the calibration arm, and a CCA arm over all testable indicators (adding the oversight H and workflow O components, each with an
equal share of the false-alarm budget, decision D13). It stores every replicate's TAFD and excess-exposure avoided in `phase2/results/m5_rehearsal.npz`, for `scripts/m5_family.py`. Exposure is in baseline-windows of harm of the stratum that defines onset (decision D18).
This is a rehearsal, not a confirmatory run: D3, D4, D23 and the window duration are not fixed, and the pilot results were seen before the seeds
were chosen. The seeds here differ from the pilot's (calibration nulls included).
"""

import logging
import os
import sys
from pathlib import Path
from dataclasses import replace
from typing import Any, Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m4_pilot import BUDGET, FLAT_ARMS, LAGS, arm_events, calibrate_all, catalog_cells, onset_index  # noqa: E402

from src.eval_module.tafd import classify_replicate, exposure_before_detection  # noqa: E402
from src.monitor_module import (INDICATOR_ORDER, build_stream, calibrate_hierarchical, component_groups, episode_starts, excess_ratio,  # noqa: E402
                                indicator_indices, null_scores_multi, progress_schedule, stream_scores_multi, system_events_hierarchical)
from src.monitor_module.review_layer import REVIEW_ORDER, build_review_params, review_scores_multi  # noqa: E402
from src.monitor_module.statistics import INDICATOR_COMPONENT  # noqa: E402
from src.monitor_module.workflow_layer import WORKFLOW_ORDER, build_workflow, workflow_scores_multi  # noqa: E402
from src.sim_module import HARM_POPULATION, DeployedModel, PerturbationConfig, PerturbationFactory, SyntheticWorld, load_config, parse_level, rhe_baselines  # noqa: E402
from src.sim_module.config import results_dir, scenarios_path  # noqa: E402
from src.sim_module.joint_schedule import joint_schedule  # noqa: E402
from src.utils import log_environment, set_seed  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
N_NULL, REPLICATES = int(os.environ.get("M5_NULL", 1000)), int(os.environ.get("M5_REPS", 200))   # the overrides are for smoke tests
RH1_POOL = ("SC07", "SC08", "SC09", "SC17", "SC19", "SC25")
REVIEWER_KEYS = ("coverage", "automation_bias", "false_reject", "latency_median")
WORKFLOW_KEYS = ("p_deliver", "tat_mult", "alert_dup", "verify")
KEEP = ("conventional", "CCA (component split)", "calibration arm")
FULL = "CCA (all testable)"
AUDIT = 0.2
STUDY_REVIEWER = {"deadline": 5.0, "coverage": 0.95}   # the placeholders of the reviewer-layer studies; used only to compute the H and O indicators
FULL_COMPONENT = {**INDICATOR_COMPONENT, **{h: "H" for h in (*REVIEW_ORDER, "H7")}, **{o: "O" for o in WORKFLOW_ORDER if o.startswith("O")}}
FULL_NAMES = (*INDICATOR_ORDER, *REVIEW_ORDER, *WORKFLOW_ORDER)
FULL_GROUPS: Dict[str, List[int]] = {}
for _k, _name in enumerate(FULL_NAMES):
    FULL_GROUPS.setdefault(FULL_COMPONENT[_name], []).append(_k)


def full_scores(stream: Any, cfg: Any, rng: np.random.Generator) -> Dict[int, np.ndarray]:
    """Model-level, oversight, and workflow scores side by side, per lag, with the reviewer and workflow at baseline. Shape (T, 25)."""
    par = joint_schedule("alert_duplication", 1.0, np.zeros(stream.y.shape[0]), cfg)
    review = build_review_params(stream, cfg, {k: par[k] for k in REVIEWER_KEYS}, rng)
    workflow = build_workflow(stream, review, cfg, {k: par[k] for k in WORKFLOW_KEYS}, rng)
    parts = (stream_scores_multi(stream, cfg, LAGS), review_scores_multi(stream, review, AUDIT, LAGS, rng), workflow_scores_multi(stream, workflow, LAGS))
    return {lag: np.concatenate([p[lag] for p in parts], axis=1) for lag in LAGS}


def null_full(world: SyntheticWorld, model: DeployedModel, cfg: Any, seed: int) -> Dict[int, np.ndarray]:
    """Full-set scores of N_NULL independent null streams, per lag. Shape (S, T, 25)."""
    per = [full_scores(build_stream(world, model, None, "abrupt", 0, cfg, np.random.default_rng(seed * 100_003 + i)), cfg, np.random.default_rng(seed * 7919 + i)) for i in range(N_NULL)]
    return {lag: np.stack([p[lag] for p in per]) for lag in LAGS}


def main() -> int:
    """Run the cells and store the replicates."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config()
    set_seed(cfg.seed + 1)
    world = SyntheticWorld(cfg.generator, cfg.harm)
    model = DeployedModel(cfg.generator).fit(world, np.random.default_rng(cfg.seed))
    j, horizon, total = cfg.routing.hysteresis_j, cfg.stream.horizon, cfg.stream.monitored_windows
    scenarios = {s["id"]: s for s in yaml.safe_load(scenarios_path().read_text(encoding="utf-8"))}
    cols = {name: indicator_indices(ids) for name, ids in FLAT_ARMS.items()}
    groups = component_groups(INDICATOR_ORDER)
    calib, evaluation = null_scores_multi(world, model, cfg, N_NULL, 303, LAGS), null_scores_multi(world, model, cfg, N_NULL, 404, LAGS)
    thresholds, _ = calibrate_all(calib, evaluation, cols, groups, j)
    baselines = rhe_baselines(model.apply(world.sample(HARM_POPULATION, np.random.default_rng(11))), cfg)
    cfg_full = replace(cfg, reviewer=replace(cfg.reviewer, **STUDY_REVIEWER))
    calib_full, eval_full = null_full(world, model, cfg_full, 303), null_full(world, model, cfg_full, 404)
    thr_full = {lag: calibrate_hierarchical(calib_full[lag], FULL_GROUPS, BUDGET, j) for lag in LAGS}
    for lag in LAGS:
        rate = episode_starts(system_events_hierarchical(eval_full[lag], FULL_GROUPS, thr_full[lag]), j).mean() * 1000
        logger.info("all-testable CCA, lag %d: %.2f alert episodes per 1,000 windows on independent null streams (budget %.1f)", lag, rate, BUDGET)
    store: Dict[str, np.ndarray] = {}
    cells = [c for c in catalog_cells(scenarios)["tafd"] if c[0] in RH1_POOL]
    for ci, (sid, level, shape) in enumerate(cells):
        spec = scenarios[sid]["perturbation"]
        pert = PerturbationFactory(spec["type"])(PerturbationConfig(level_value=parse_level(spec["levels"][level]), mode=spec.get("mode")))
        lo, hi = cfg.stream.s0_low, (cfg.stream.s0_high_abrupt if shape == "abrupt" else cfg.stream.s0_high_ramped)
        rng_s0 = np.random.default_rng(10_900 + ci)
        onset_cache: Dict[float, float] = {}
        excess_cache: Dict[float, float] = {}
        tafd: Dict[int, Dict[str, List[int]]] = {lag: {a: [] for a in (*KEEP, FULL)} for lag in LAGS}
        avoided: Dict[int, List[float]] = {lag: [] for lag in LAGS}
        avoided_full: Dict[int, List[float]] = {lag: [] for lag in LAGS}
        for r in range(REPLICATES):
            s0 = int(rng_s0.integers(lo, hi + 1))
            t0 = onset_index(pert, world, model, cfg, s0, shape, baselines, onset_cache)
            if t0 is None or t0 + horizon > total:
                continue
            progress = progress_schedule(shape, s0, total, cfg.stream.ramp_windows)
            for p in set(np.round(progress, 6)):
                if p not in excess_cache:
                    excess_cache[p] = excess_ratio(pert, world, model, cfg, float(p), baselines)
            excess = np.array([excess_cache[p] for p in np.round(progress, 6)])
            stream = build_stream(world, model, pert, shape, s0, cfg, np.random.default_rng(5_500_000 + ci * 1000 + r))
            zs, zf = stream_scores_multi(stream, cfg, LAGS), full_scores(stream, cfg_full, np.random.default_rng(8_500_000 + ci * 1000 + r))
            for lag in LAGS:
                for name in KEEP:
                    out = classify_replicate(arm_events(name, zs[lag][None], thresholds[lag][name], cols, groups)[0], s0, t0, horizon, j)
                    tafd[lag][name].append(int(out["tafd"]))
                full = classify_replicate(system_events_hierarchical(zf[lag][None], FULL_GROUPS, thr_full[lag])[0], s0, t0, horizon, j)
                tafd[lag][FULL].append(int(full["tafd"]))
                avoided_full[lag].append(exposure_before_detection(excess, t0, tafd[lag]["conventional"][-1]) - exposure_before_detection(excess, t0, tafd[lag][FULL][-1]))
                avoided[lag].append(exposure_before_detection(excess, t0, tafd[lag]["conventional"][-1]) - exposure_before_detection(excess, t0, tafd[lag]["CCA (component split)"][-1]))
        logger.info("cell %s %s %s: %d replicates", sid, level, shape, len(avoided[LAGS[0]]))
        for lag in LAGS:
            for name in (*KEEP, FULL):
                store[f"{sid}|{level}|{shape}|{lag}|{name}"] = np.array(tafd[lag][name], dtype=float)
            store[f"{sid}|{level}|{shape}|{lag}|avoided"] = np.array(avoided[lag], dtype=float)
            store[f"{sid}|{level}|{shape}|{lag}|avoided_full"] = np.array(avoided_full[lag], dtype=float)
    store["baseline_rhe"] = np.array([1.0])      # exposure is already in baseline-windows of the onset stratum, so the unit is 1
    output = results_dir() / "m5_rehearsal.npz"
    np.savez(output, **store)
    logger.info("Wrote %s (%s)", output, log_environment())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
