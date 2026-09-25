"""Milestone M5f: the pre-specified rolling-calibration sensitivity for RH2 (decision D21).

Usage: python scripts/m5f_rolling_rh2.py
Repeats the RH2 comparison (SC08, SC17, SC25) with a calibration arm built on the trailing four windows pooled (ECE, Brier, slope deviation),
beside the per-window calibration arm, against the performance arm (P1, P2, P4). Each arm is calibrated to the same false-alarm budget. Writes
phase2/M5f-Rolling-Calibration.md. Exploratory, fresh seeds. Reported whatever it shows, as decision D21 requires.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m4_pilot import LAGS, catalog_cells, onset_index  # noqa: E402

from src.eval_module.pooled import pooled_bootstrap  # noqa: E402
from src.eval_module.tafd import classify_replicate  # noqa: E402
from src.monitor_module import (INDICATOR_ORDER, build_stream, calibrate_threshold, episode_starts, indicator_indices,  # noqa: E402
                                stream_scores_multi, system_events)
from src.monitor_module.rolling import ROLLING_ORDER, rolling_scores_multi  # noqa: E402
from src.sim_module import HARM_POPULATION, DeployedModel, PerturbationConfig, PerturbationFactory, SyntheticWorld, load_config, parse_level, rhe_baselines  # noqa: E402
from src.sim_module.config import false_alarm_budget  # noqa: E402
from src.utils import log_environment, set_seed  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "phase2" / "M5f-Rolling-Calibration.md"
N_NULL, REPLICATES, SPAN = 1000, 200, 4
BUDGET = false_alarm_budget()
POOL = ("SC08", "SC17", "SC25")
K = len(INDICATOR_ORDER)
ARMS: Dict[str, List[int]] = {"performance (P1, P2, P4)": indicator_indices(["P1", "P2", "P4"]), "calibration, per window (U1, U2, U3)": indicator_indices(["U1", "U2", "U3"]),
                              "calibration, rolling four windows": [K + i for i in range(len(ROLLING_ORDER))]}
REF, PER_WINDOW, ROLLING = list(ARMS)


def scores(stream: Any, cfg: Any) -> Dict[int, np.ndarray]:
    """Model-level and rolling scores side by side, per lag. Shape (T, 15)."""
    base, roll = stream_scores_multi(stream, cfg, LAGS), rolling_scores_multi(stream, LAGS, SPAN)
    return {lag: np.concatenate([base[lag], roll[lag]], axis=1) for lag in LAGS}


def null_scores(world: SyntheticWorld, model: DeployedModel, cfg: Any, seed: int) -> Dict[int, np.ndarray]:
    """Scores of N_NULL independent null streams, per lag. Shape (S, T, 15)."""
    per = [scores(build_stream(world, model, None, "abrupt", 0, cfg, np.random.default_rng(seed * 100_003 + i)), cfg) for i in range(N_NULL)]
    return {lag: np.stack([p[lag] for p in per]) for lag in LAGS}


def main() -> int:
    """Run the cells and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config()
    set_seed(cfg.seed + 2)
    world = SyntheticWorld(cfg.generator, cfg.harm)
    model = DeployedModel(cfg.generator).fit(world, np.random.default_rng(cfg.seed))
    j, horizon, total = cfg.routing.hysteresis_j, cfg.stream.horizon, cfg.stream.monitored_windows
    scenarios = {s["id"]: s for s in yaml.safe_load((ROOT / "phase2" / "scenarios.yaml").read_text(encoding="utf-8"))}
    calib, evaluation = null_scores(world, model, cfg, 505), null_scores(world, model, cfg, 606)
    thr = {lag: {a: calibrate_threshold(calib[lag], cols, BUDGET, j) for a, cols in ARMS.items()} for lag in LAGS}
    cal_rows = [f"| {lag} | {a} | {thr[lag][a]:.2f} | {episode_starts(system_events(evaluation[lag], cols, thr[lag][a]), j).mean() * 1000:.2f} |" for lag in LAGS for a, cols in ARMS.items()]
    baselines = rhe_baselines(model.apply(world.sample(HARM_POPULATION, np.random.default_rng(11))), cfg)
    diffs: Dict[int, Dict[str, Dict[str, List[np.ndarray]]]] = {lag: {arm: {s: [] for s in POOL} for arm in (PER_WINDOW, ROLLING)} for lag in LAGS}
    table: Dict[int, List[str]] = {lag: [] for lag in LAGS}
    cells = [c for c in catalog_cells(scenarios)["tafd"] if c[0] in POOL]
    for ci, (sid, level, shape) in enumerate(cells):
        spec = scenarios[sid]["perturbation"]
        pert = PerturbationFactory(spec["type"])(PerturbationConfig(level_value=parse_level(spec["levels"][level]), mode=spec.get("mode")))
        lo, hi = cfg.stream.s0_low, (cfg.stream.s0_high_abrupt if shape == "abrupt" else cfg.stream.s0_high_ramped)
        rng_s0 = np.random.default_rng(12_900 + ci)
        cache: Dict[float, float] = {}
        tafd: Dict[int, Dict[str, List[int]]] = {lag: {a: [] for a in ARMS} for lag in LAGS}
        for r in range(REPLICATES):
            s0 = int(rng_s0.integers(lo, hi + 1))
            t0 = onset_index(pert, world, model, cfg, s0, shape, baselines, cache)
            if t0 is None or t0 + horizon > total:
                continue
            zs = scores(build_stream(world, model, pert, shape, s0, cfg, np.random.default_rng(6_500_000 + ci * 1000 + r)), cfg)
            for lag in LAGS:
                for a, cols in ARMS.items():
                    tafd[lag][a].append(int(classify_replicate(system_events(zs[lag][None], cols, thr[lag][a])[0], s0, t0, horizon, j)["tafd"]))
        n = len(tafd[LAGS[0]][REF])
        logger.info("cell %s %s %s: %d replicates", sid, level, shape, n)
        for lag in LAGS:
            if not n:
                continue
            arr = {k: np.array(v, dtype=float) for k, v in tafd[lag].items()}
            for arm in (PER_WINDOW, ROLLING):
                diffs[lag][arm][sid].append(arr[REF] - arr[arm])
            table[lag].append(f"| {sid} | {level} | {shape} | {n} | " + " | ".join(f"{arr[k].mean():.1f}" for k in ARMS) + " |")
    lines = ["# Phase II M5f: rolling four-window calibration sensitivity for RH2 (exploratory)", "",
             "Generated by `scripts/m5f_rolling_rh2.py`. The RH2 comparison with a calibration arm built on the cases of the last four windows pooled, beside the per-window calibration arm, "
             f"against the performance arm, at a matched false-alarm budget of {BUDGET:g} per 1,000 windows ({N_NULL} calibration and {N_NULL} evaluation null streams; fresh seeds). "
             f"{REPLICATES} replicates per cell, SC08, SC17, and SC25 with equal scenario weights. The reduction is the performance arm's mean TAFD minus the calibration arm's, in windows, so a "
             "positive value favors the calibration arm; the null is at most zero. Pre-specified in decision D21 and reported whatever it shows. "
             f"Environment: {log_environment()}", "", "## Calibration", "", "| Lag | Arm | Threshold | Achieved per 1,000 windows |", "|---|---|---|---|"] + cal_rows
    lines += ["", "## RH2 pooled reduction", "", "| Lag | Calibration arm | Reduction [95% CI] | p (one-sided) |", "|---|---|---|---|"]
    for lag in LAGS:
        for arm in (PER_WINDOW, ROLLING):
            est, lo_ci, hi_ci, p = pooled_bootstrap(diffs[lag][arm], 0.0, np.random.default_rng(9 + lag))
            lines.append(f"| {lag} | {arm} | {est:+.2f} [{lo_ci:+.2f}, {hi_ci:+.2f}] | {p:.3f} |")
    lines += ["", "## Mean TAFD by cell (windows; undetected count as the horizon)", ""]
    for lag in LAGS:
        lines += [f"### Label lag {lag} windows", "", "| Scenario | Level | Shape | n | " + " | ".join(ARMS) + " |", "|---|---|---|---|---|---|---|"] + table[lag] + [""]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", OUTPUT.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
