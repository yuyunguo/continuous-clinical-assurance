"""Milestones M3 and M4 pilot: matched false-alarm calibration and an exploratory detection comparison.

Usage: python scripts/m4_pilot.py
Writes phase2/M3-M4-Pilot.md. Exploratory only: it uses the 12 model-level indicators, the yellow
level only, and one seed. It does not test RH1 to RH4. It implements decisions D13 (component-level budget split), D14
(label lags 0, 2, and 8), D18 (subgroup-aware onset), and D19 (absolute reductions in windows against a margin).
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_module.tafd import bootstrap_difference, bootstrap_mean, classify_replicate, exposure_before_detection, rmst_difference  # noqa: E402
from src.monitor_module import (INDICATOR_ORDER, excess_ratio, build_stream, calibrate_hierarchical, calibrate_threshold, component_groups,  # noqa: E402
                                episode_starts, indicator_indices, null_scores_multi, progress_schedule, stream_scores_multi,
                                system_events, system_events_hierarchical)
from src.sim_module import (HARM_POPULATION, DeployedModel, PerturbationConfig, PerturbationFactory, SyntheticWorld, load_config, onset_ratio,  # noqa: E402
                            parse_level, registered_perturbations, rhe_baselines)
from src.sim_module.config import false_alarm_budget  # noqa: E402
from src.utils import log_environment, set_seed  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "phase2" / "M3-M4-Pilot.md"
N_NULL = 1000
REPLICATES = 150
BUDGET = false_alarm_budget()   # D2: alert episodes per 1,000 monitored windows, from the config (M5_BUDGET overrides it for the budget sweep)
LAGS = (0, 2, 8)
MARGIN_WINDOWS = 2.0     # decision D23: the primary margin, in windows (delta = 1 baseline-window of harm)
SENSITIVITY_MARGIN = 1.0  # one governance cycle at the elicited volume
TILT = {"P": 0.6, "U": 0.2, "C": 0.2}   # informational only, never selected
FLAT_ARMS = {"conventional": ["P1", "P2", "P4"], "conv + label-free": ["P1", "P2", "P4", "C2", "U5"],
             "conv + calibration": ["P1", "P2", "P4", "U1", "U2", "U3"], "CCA (union)": list(INDICATOR_ORDER),
             "calibration arm": ["U1", "U2", "U3"]}
SPLIT_ARMS = {"CCA (component split)": None, "CCA (tilted, informational)": TILT}
ARM_ORDER = ["conventional", "conv + label-free", "conv + calibration", "CCA (union)", "CCA (component split)", "CCA (tilted, informational)", "calibration arm"]
Thresholds = Union[float, Dict[str, float]]


def catalog_cells(scenarios: Dict[str, Any]) -> Dict[str, List[Tuple[str, str, str]]]:
    """TAFD cells (fair-test or blind-spot scenarios with a harm-defined onset) and alert-rate cells (harmless drift)."""
    implemented = set(registered_perturbations())
    tafd: List[Tuple[str, str, str]] = []
    alert: List[Tuple[str, str, str]] = []
    for sid, s in scenarios.items():
        if s["perturbation"]["type"] not in implemented or s["family"] == "control":
            continue
        shapes = s["perturbation"]["shapes"]
        if s["purpose"] in ("fair_test", "blind_spot") and s["primary_tafd"]:
            if "abrupt" in shapes:
                tafd += [(sid, level, "abrupt") for level in ("medium", "high")]
                if "gradual" in shapes and s["purpose"] == "fair_test":
                    tafd.append((sid, "high", "gradual"))
            else:                                    # a scenario that only ramps (SC19)
                tafd += [(sid, level, shapes[0]) for level in ("medium", "high")]
        elif s["purpose"] == "harmless_drift":
            alert.append((sid, "high", "abrupt"))
    return {"tafd": tafd, "alert": alert}


def onset_index(pert: Any, world: SyntheticWorld, model: DeployedModel, cfg: Any, s0: int, shape: str,
                baselines: Dict[str, float], cache: Dict[float, float]) -> Optional[int]:
    """First monitored window at which the onset ratio (decision D18) reaches 1 + kappa, or None."""
    progress = progress_schedule(shape, s0, cfg.stream.monitored_windows, cfg.stream.ramp_windows)
    for t in range(s0, len(progress)):
        p = round(float(progress[t]), 6)
        if p not in cache:
            cache[p] = onset_ratio(pert.generate(world, model, HARM_POPULATION, np.random.default_rng(31), p), cfg, baselines)
        if cache[p] >= 1 + cfg.harm.kappa:
            return t
    return None


def arm_events(name: str, z: np.ndarray, thresholds: Thresholds, cols: Dict[str, List[int]], groups: Dict[str, List[int]]) -> np.ndarray:
    """System events (S, T) for one arm."""
    if name in SPLIT_ARMS:
        return system_events_hierarchical(z, groups, thresholds)  # type: ignore[arg-type]
    return system_events(z, cols[name], thresholds)  # type: ignore[arg-type]


def calibrate_all(calib: Dict[int, np.ndarray], evaluation: Dict[int, np.ndarray], cols: Dict[str, List[int]],
                  groups: Dict[str, List[int]], j: int) -> Tuple[Dict[int, Dict[str, Thresholds]], List[str]]:
    """Per lag and arm, calibrate on the calibration streams and report the rate on the independent evaluation streams."""
    out: Dict[int, Dict[str, Thresholds]] = {}
    rows: List[str] = []
    for lag in LAGS:
        out[lag] = {}
        for name in ARM_ORDER:
            if name in SPLIT_ARMS:
                out[lag][name] = calibrate_hierarchical(calib[lag], groups, BUDGET, j, shares=SPLIT_ARMS[name])
            else:
                out[lag][name] = calibrate_threshold(calib[lag], cols[name], BUDGET, j)
            starts = episode_starts(arm_events(name, evaluation[lag], out[lag][name], cols, groups), j)
            windows = starts.size
            rate, se = starts.sum() / windows * 1000, np.sqrt(max(starts.sum(), 1)) / windows * 1000
            th = out[lag][name]
            shown = f"{th:.2f}" if isinstance(th, float) else ", ".join(f"{k} {v:.2f}" for k, v in th.items())
            rows.append(f"| {lag} | {name} | {shown} | {rate:.2f} | {se:.2f} | {'yes' if abs(rate - BUDGET) <= 3 * se else 'no'} |")
    return out, rows


def main() -> int:
    """Run the pilot and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config()
    set_seed(cfg.seed)
    world = SyntheticWorld(cfg.generator, cfg.harm)
    model = DeployedModel(cfg.generator).fit(world, np.random.default_rng(cfg.seed))
    j, horizon, total = cfg.routing.hysteresis_j, cfg.stream.horizon, cfg.stream.monitored_windows
    scenarios = {s["id"]: s for s in yaml.safe_load((ROOT / "phase2" / "scenarios.yaml").read_text(encoding="utf-8"))}
    cols = {name: indicator_indices(ids) for name, ids in FLAT_ARMS.items()}
    groups = component_groups(INDICATOR_ORDER)
    logger.info("Generating %d + %d null streams for lags %s", N_NULL, N_NULL, LAGS)
    calib, evaluation = null_scores_multi(world, model, cfg, N_NULL, 101, LAGS), null_scores_multi(world, model, cfg, N_NULL, 202, LAGS)
    thresholds, cal_rows = calibrate_all(calib, evaluation, cols, groups, j)
    baselines = rhe_baselines(model.apply(world.sample(HARM_POPULATION, np.random.default_rng(11))), cfg)
    lines = ["# Phase II M3 and M4 pilot (exploratory)", "",
             "Generated by `scripts/m4_pilot.py`. This is a pilot, not a test of RH1 to RH4. It uses the 12 model-level indicators, "
             "the yellow level only, CUSUM detectors with reference value 0.5, and one seed. It implements D13 (component-level budget split), D14 (label lags "
             f"{', '.join(map(str, LAGS))}), D18 (onset is the larger of the pooled ratio and the predefined subgroup's ratio), and D19 (reductions in windows "
             f"against a margin of {MARGIN_WINDOWS:g} windows (decision D23, with {SENSITIVITY_MARGIN:g} window as sensitivity)). Null streams: {N_NULL} to calibrate and {N_NULL} independent streams to evaluate, for each lag. "
             f"Replicates: {REPLICATES} per cell. The budget is {BUDGET:g} alert episodes per 1,000 windows. Harm weights, the reviewer model, kappa, and the "
             "window duration are placeholders. The tilted split is informational and was not selected.", "", f"Environment: {log_environment()}", "",
             "## Matched false-alarm calibration", "",
             "Rates are alert-episode starts per 1,000 monitored windows on independent evaluation streams. A system meets its budget when its rate is within 3 standard errors of the target.", "",
             "| Lag | System | Threshold | Achieved | SE | Meets budget |", "|---|---|---|---|---|---|"] + cal_rows + [""]

    tables: Dict[int, List[str]] = {lag: [] for lag in LAGS}
    cells = catalog_cells(scenarios)
    for ci, (sid, level, shape) in enumerate(cells["tafd"]):
        spec = scenarios[sid]["perturbation"]
        pert = PerturbationFactory(spec["type"])(PerturbationConfig(level_value=parse_level(spec["levels"][level]), mode=spec.get("mode")))
        lo, hi = cfg.stream.s0_low, (cfg.stream.s0_high_abrupt if shape == "abrupt" else cfg.stream.s0_high_ramped)
        cache: Dict[float, float] = {}
        rng_s0 = np.random.default_rng(900 + ci)
        tafd: Dict[int, Dict[str, List[int]]] = {lag: {a: [] for a in ARM_ORDER} for lag in LAGS}
        lead: Dict[int, Dict[str, List[int]]] = {lag: {a: [] for a in ARM_ORDER} for lag in LAGS}
        avoided: Dict[int, List[float]] = {lag: [] for lag in LAGS}
        excess_cache: Dict[float, float] = {}
        for r in range(REPLICATES):
            s0 = int(rng_s0.integers(lo, hi + 1))
            t0 = onset_index(pert, world, model, cfg, s0, shape, baselines, cache)
            if t0 is None or t0 + horizon > total:
                continue
            zs = stream_scores_multi(build_stream(world, model, pert, shape, s0, cfg, np.random.default_rng(500_000 + ci * 1000 + r)), cfg, LAGS)
            for p in set(np.round(progress_schedule(shape, s0, total, cfg.stream.ramp_windows), 6)):
                if p not in excess_cache:
                    excess_cache[p] = excess_ratio(pert, world, model, cfg, float(p), baselines)
            excess = np.array([excess_cache[p] for p in np.round(progress_schedule(shape, s0, total, cfg.stream.ramp_windows), 6)])
            for lag in LAGS:
                for name in ARM_ORDER:
                    out = classify_replicate(arm_events(name, zs[lag][None], thresholds[lag][name], cols, groups)[0], s0, t0, horizon, j)
                    tafd[lag][name].append(int(out["tafd"]))
                    lead[lag][name].append(int(out["lead_time"]))
                avoided[lag].append(exposure_before_detection(excess, t0, tafd[lag]["conventional"][-1]) - exposure_before_detection(excess, t0, tafd[lag]["CCA (component split)"][-1]))
        used = len(tafd[LAGS[0]]["conventional"])
        logger.info("cell %s %s %s: %d replicates", sid, level, shape, used)
        for lag in LAGS:
            if used == 0:
                tables[lag].append(f"| {sid} | {level} | {shape} | 0 | no onset or no room for the horizon | | | | | | | | | |")
                continue
            a = {k: np.array(v, dtype=float) for k, v in tafd[lag].items()}
            rng_b = np.random.default_rng(7)
            d_split, d_union, d_cal = (rmst_difference(a["conventional"], a[k]) for k in ("CCA (component split)", "CCA (union)", "calibration arm"))
            lo_s, hi_s = bootstrap_difference(a["conventional"], a["CCA (component split)"], rng_b)
            lo_c, hi_c = bootstrap_difference(a["conventional"], a["calibration arm"], rng_b)
            av = np.array(avoided[lag])
            lo_e, hi_e = bootstrap_mean(av, rng_b)
            tables[lag].append(
                f"| {sid} | {level} | {shape} | {used} | " + " | ".join(f"{a[k].mean():.1f}" for k in ("conventional", "conv + label-free", "conv + calibration", "CCA (union)", "CCA (component split)"))
                + f" | {d_split:+.1f} [{lo_s:+.1f}, {hi_s:+.1f}] | {av.mean():+.2f} [{lo_e:+.2f}, {hi_e:+.2f}] | {'yes' if lo_s >= MARGIN_WINDOWS else 'no'} | {d_union:+.1f} | {d_cal:+.1f} [{lo_c:+.1f}, {hi_c:+.1f}] | {'yes' if lo_s >= SENSITIVITY_MARGIN else 'no'} |")
    header = ("| Scenario | Level | Shape | n | Conventional | Conv + label-free | Conv + calibration | CCA (union) | CCA (split) | "
              "Reduction, split vs conventional [95% CI] | Exposure avoided in baseline-windows of harm, split vs conventional [95% CI] | Meets 2-window margin (D23) | Reduction, union | Reduction, calibration arm [95% CI] | Meets 1-window margin (sensitivity) |")
    lines += ["## Detection comparison", "",
              "Mean TAFD in windows (smaller is better, undetected replicates count as the horizon of "
              f"{horizon}). A reduction is conventional minus the arm, in windows, so a positive value means the arm detects earlier. The margin column asks whether "
              f"the lower end of the interval is at least {MARGIN_WINDOWS:g} windows (decision D23), or {SENSITIVITY_MARGIN:g} window in the last column. Replicates whose onset does not occur or leaves no room for the horizon are dropped.", ""]
    for lag in LAGS:
        lines += [f"### Label lag {lag} windows", "", header, "|" + "---|" * 15] + tables[lag] + [""]

    lines += ["## Alerts on harmless drift", "", "The share of replicates in which each system raised an alert within "
              f"{horizon} windows of the drift start, against the null rate (the share of {horizon}-window stretches of a null stream with an alert). "
              "The drift here never reaches the onset threshold.", "", "| Lag | Scenario | Level | Null: conventional | Null: CCA (split) | Conventional | CCA (union) | CCA (split) |", "|---|---|---|---|---|---|---|---|"]
    for ci, (sid, level, shape) in enumerate(cells["alert"]):
        spec = scenarios[sid]["perturbation"]
        pert = PerturbationFactory(spec["type"])(PerturbationConfig(level_value=parse_level(spec["levels"][level]), mode=spec.get("mode")))
        rng_s0 = np.random.default_rng(1900 + ci)
        hit = {lag: {a: 0 for a in ("conventional", "CCA (union)", "CCA (component split)")} for lag in LAGS}
        for r in range(REPLICATES):
            s0 = int(rng_s0.integers(cfg.stream.s0_low, cfg.stream.s0_high_abrupt + 1))
            zs = stream_scores_multi(build_stream(world, model, pert, shape, s0, cfg, np.random.default_rng(800_000 + ci * 1000 + r)), cfg, LAGS)
            for lag in LAGS:
                for name in hit[lag]:
                    hit[lag][name] += bool(arm_events(name, zs[lag][None], thresholds[lag][name], cols, groups)[0][s0:s0 + horizon].any())
        for lag in LAGS:
            null = {}
            for name in ("conventional", "CCA (component split)"):
                ev = arm_events(name, evaluation[lag], thresholds[lag][name], cols, groups)
                null[name] = float(np.stack([ev[:, s:s + horizon].any(axis=1) for s in range(ev.shape[1] - horizon + 1)]).mean())
            lines.append(f"| {lag} | {sid} | {level} | {null['conventional']:.2f} | {null['CCA (component split)']:.2f} | "
                         + " | ".join(f"{hit[lag][n] / REPLICATES:.2f}" for n in ("conventional", "CCA (union)", "CCA (component split)")) + " |")
        logger.info("alert cell %s %s", sid, level)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", OUTPUT.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
