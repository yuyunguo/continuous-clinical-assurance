"""Milestone M5e: the RH4 test (design section 9.1), first pass.

Usage: python scripts/m5e_rh4.py
theta4 is the mean over SC13, SC14, SC16, and SC22 of the workflow arm's (H7, O1 to O6) minus the model-level arm's (P, U, C) proportion of replicates
detected within the horizon, at a matched false-alarm budget. Null: theta4 is at most 0.20 (a placeholder margin). Writes phase2/M5e-RH4-Workflow.md.
Exploratory, one seed, Tier 1 synthetic data with assumed reviewer and workflow models. Not adjusted for the four-hypothesis family.
"""

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_module.tafd import classify_replicate  # noqa: E402
from src.monitor_module import INDICATOR_ORDER, build_stream, calibrate_threshold, episode_starts, progress_schedule, stream_scores_multi, system_events  # noqa: E402
from src.monitor_module.review_layer import build_review_params  # noqa: E402
from src.monitor_module.workflow_layer import WORKFLOW_ORDER, build_workflow, workflow_scores_multi  # noqa: E402
from src.sim_module import DeployedModel, SyntheticWorld, load_config, parse_level  # noqa: E402
from src.sim_module.joint_schedule import joint_schedule  # noqa: E402
from src.sim_module.workflow_harm import WorkflowRatio  # noqa: E402
from src.sim_module.config import false_alarm_budget, results_dir  # noqa: E402
from src.utils import log_environment, set_seed  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "phase2" / "M5e-RH4-Workflow.md"
N_NULL, REPLICATES, LAGS, DEADLINE, COVERAGE, MARGIN, N_BOOT = 600, 200, (0, 2, 8), 5.0, 0.95, 0.20, 2000
BUDGET = false_alarm_budget()
H_SHORT = 4      # decision D25: the primary RH4 horizon in windows. With a loose budget the 40-window horizon saturates (the model-level arm's false alarms alone detect most replicates)
HORIZONS = (2, 4, 8)   # 4 is primary (D25); 2 and 8 are sensitivities
K_MODEL, K_ALL = len(INDICATOR_ORDER), len(INDICATOR_ORDER) + len(WORKFLOW_ORDER)
ARMS: Dict[str, List[int]] = {"model-level (P, U, C)": list(range(K_MODEL)), "workflow (H7, O1 to O6)": list(range(K_MODEL, K_ALL))}
POOL = ("SC13", "SC14", "SC16", "SC22")
LEVELS = ("medium", "high")
REVIEWER_KEYS = ("coverage", "automation_bias", "false_reject", "latency_median")
WORKFLOW_KEYS = ("p_deliver", "tat_mult", "alert_dup", "verify")


def stream_scores(stream: Any, cfg: Any, par: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[int, np.ndarray]:
    """Model-level and workflow scores side by side, per lag. Shape (T, 19)."""
    review = build_review_params(stream, cfg, {k: par[k] for k in REVIEWER_KEYS}, rng)
    workflow = build_workflow(stream, review, cfg, {k: par[k] for k in WORKFLOW_KEYS}, rng)
    model_z, workflow_z = stream_scores_multi(stream, cfg, LAGS), workflow_scores_multi(stream, workflow, LAGS)
    return {lag: np.concatenate([model_z[lag], workflow_z[lag]], axis=1) for lag in LAGS}


def null_scores(world: SyntheticWorld, model: DeployedModel, cfg: Any, seed: int) -> Dict[int, np.ndarray]:
    """Scores of N_NULL independent null streams, per lag. Shape (S, T, 19)."""
    w = cfg.stream.baseline_windows + cfg.stream.monitored_windows
    par = joint_schedule("alert_duplication", 1.0, np.zeros(w), cfg)
    per = [stream_scores(build_stream(world, model, None, "abrupt", 0, cfg, np.random.default_rng(seed * 100_003 + i)), cfg, par, np.random.default_rng(seed * 7919 + i))
           for i in range(N_NULL)]
    return {lag: np.stack([p[lag] for p in per]) for lag in LAGS}


def theta4(cells: Dict[str, List[Tuple[np.ndarray, np.ndarray]]], rng: np.random.Generator) -> Tuple[float, float, float, float]:
    """Point estimate, 95 percent interval, and one-sided p (null: at most MARGIN). `cells` maps scenario to (workflow, model) detection indicators per cell."""
    def value(draw: bool) -> float:
        per = []
        for pairs in cells.values():
            diffs = []
            for wf, mo in pairs:
                idx = rng.integers(0, len(wf), len(wf)) if draw else np.arange(len(wf))
                diffs.append(wf[idx].mean() - mo[idx].mean())
            if diffs:
                per.append(np.mean(diffs))
        return float(np.mean(per)) if per else float("nan")
    boots = np.array([value(True) for _ in range(N_BOOT)])
    return value(False), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5)), float(np.nanmean(boots <= MARGIN))


def main() -> int:
    """Run the pilot and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg0 = load_config()
    cfg = replace(cfg0, reviewer=replace(cfg0.reviewer, deadline=DEADLINE, coverage=COVERAGE))
    set_seed(cfg.seed)
    world = SyntheticWorld(cfg.generator, cfg.harm)
    model = DeployedModel(cfg.generator).fit(world, np.random.default_rng(cfg.seed))
    j, horizon, total = cfg.routing.hysteresis_j, cfg.stream.horizon, cfg.stream.monitored_windows
    b = cfg.stream.baseline_windows
    scenarios = {s["id"]: s for s in yaml.safe_load((ROOT / "phase2" / "scenarios.yaml").read_text(encoding="utf-8"))}
    calib, evaluation = null_scores(world, model, cfg, 101), null_scores(world, model, cfg, 202)
    thr = {lag: {a: calibrate_threshold(calib[lag], cols, BUDGET, j) for a, cols in ARMS.items()} for lag in LAGS}
    cal_rows = [f"| {lag} | {a} | {thr[lag][a]:.2f} | " f"{episode_starts(system_events(evaluation[lag], cols, thr[lag][a]), j).mean() * 1000:.2f} |" for lag in LAGS for a, cols in ARMS.items()]
    ratio = WorkflowRatio(cfg, model.apply(world.sample(200_000, np.random.default_rng(11))))
    detect: Dict[int, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = {lag: {s: [] for s in POOL} for lag in LAGS}
    detect_short: Dict[int, Dict[int, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]]] = {h: {lag: {s: [] for s in POOL} for lag in LAGS} for h in HORIZONS}
    table: Dict[int, List[str]] = {lag: [] for lag in LAGS}
    for ci, (sid, level) in enumerate((s, lv) for s in POOL for lv in LEVELS):
        spec = scenarios[sid]["perturbation"]
        value, kind = parse_level(spec["levels"][level]), spec["type"]
        lo, hi = cfg.stream.s0_low, cfg.stream.s0_high_abrupt
        rng_s0, cache = np.random.default_rng(4000 + ci), {}
        res: Dict[int, Dict[str, List[Tuple[int, bool]]]] = {lag: {a: [] for a in ARMS} for lag in LAGS}
        full_ratio = ratio({k: float(v[0]) for k, v in joint_schedule(kind, value, np.array([1.0]), cfg).items()})
        for r in range(REPLICATES):
            s0 = int(rng_s0.integers(lo, hi + 1))
            progress = progress_schedule("abrupt", s0, total, cfg.stream.ramp_windows)
            for p in set(np.round(progress[s0:], 6)):
                if p not in cache:
                    cache[p] = ratio({k: float(v[0]) for k, v in joint_schedule(kind, value, np.array([float(p)]), cfg).items()})
            onset = [t for t in range(s0, total) if cache[round(float(progress[t]), 6)] >= 1 + cfg.harm.kappa]
            if not onset or onset[0] + horizon > total:
                continue
            par = joint_schedule(kind, value, np.concatenate([np.zeros(b), progress]), cfg)
            rng = np.random.default_rng(700_000 + ci * 1000 + r)
            zs = stream_scores(build_stream(world, model, None, "abrupt", s0, cfg, rng), cfg, par, rng)
            for lag in LAGS:
                for a, cols in ARMS.items():
                    out = classify_replicate(system_events(zs[lag][None], cols, thr[lag][a])[0], s0, onset[0], horizon, j)
                    res[lag][a].append((int(out["tafd"]), not out["censored"]))
        n = len(res[LAGS[0]][list(ARMS)[0]])
        logger.info("cell %s %s: onset ratio at full %.2f, %d replicates", sid, level, full_ratio, n)
        for lag in LAGS:
            if n:
                detect[lag][sid].append((np.array([d for _, d in res[lag]["workflow (H7, O1 to O6)"]], dtype=float), np.array([d for _, d in res[lag]["model-level (P, U, C)"]], dtype=float)))
                for h in HORIZONS:
                    detect_short[h][lag][sid].append((np.array([d and tf <= h for tf, d in res[lag]["workflow (H7, O1 to O6)"]], dtype=float), np.array([d and tf <= h for tf, d in res[lag]["model-level (P, U, C)"]], dtype=float)))
            cells = " | ".join(f"{np.mean([d for _, d in v]):.0%} ({np.mean([t for t, _ in v]):.1f})" if n else "" for v in res[lag].values())
            table[lag].append(f"| {sid} | {level} | {full_ratio:.2f} | {n} | {cells} |")
    lines = ["# Phase II M5e: RH4 workflow-versus-model test, first pass (exploratory)", "",
             "Generated by `scripts/m5e_rh4.py`. theta4 is the mean over SC13, SC14, SC16, and SC22 of the workflow arm's minus the model-level arm's proportion of replicates "
             f"detected within {horizon} windows of a harm-defined onset, each arm calibrated to {BUDGET:g} false alarms per 1,000 windows ({N_NULL} calibration and {N_NULL} evaluation "
             f"null streams). Null: theta4 is at most {MARGIN:g} (a placeholder margin). Medium and high levels, abrupt, {REPLICATES} replicates per cell, cells below the onset "
             "threshold dropped. Cells are averaged within a scenario and scenarios have equal weight. The interval and the one-sided p come from a bootstrap over replicates "
             "within cells. The p-value is not adjusted for the four-hypothesis family. Reviewer and workflow models are assumptions, and the model does not change in these scenarios, "
             "so the model-level arm cannot respond by construction (design section 9.1). A detection share for the model-level arm is its chance level. The point of no return is "
             f"{DEADLINE:g} median review times and baseline review coverage is {COVERAGE:.0%}. Environment: {log_environment()}", "",
             "## Calibration", "", "| Lag | Arm | Threshold | Achieved per 1,000 windows |", "|---|---|---|---|"] + cal_rows + ["", "## Result", "",
             "| Lag | theta4 | 95% CI | p (one-sided, null: at most the margin) | Rule met |", "|---|---|---|---|---|"]
    headline: Dict[str, Dict[str, float]] = {}
    for lag in LAGS:
        est, lo_ci, hi_ci, p = theta4(detect[lag], np.random.default_rng(5))
        headline[str(lag)] = {"estimate": est, "lo": lo_ci, "hi": hi_ci, "p": p}
        lines.append(f"| {lag} | {est:+.3f} | [{lo_ci:+.3f}, {hi_ci:+.3f}] | {p:.3f} | {'yes' if lo_ci > MARGIN else 'no'} |")
    horizons_out: Dict[str, Dict[str, Dict[str, float]]] = {}
    chance_full = 1 - np.exp(-BUDGET * horizon / 1000)
    lines += ["", f"## The primary RH4 test: detection within {H_SHORT} windows of onset (decision D25)", "",
              f"theta4 is bounded by one minus the model-level arm's chance detection rate. At the budget of {BUDGET:g} per 1,000 windows that chance rate is {chance_full:.0%} within the {horizon}-window horizon "
              f"(so the 40-window theta4 above cannot exceed {1 - chance_full:.2f}, and a margin of {MARGIN:g} cannot be rejected once the budget reaches about 45). A replicate counts as detected here only if the first "
              f"alarm comes within h windows of onset. The primary horizon is {H_SHORT}; 2 and 8 are sensitivities.", ""]
    for h in HORIZONS:
        chance_h = 1 - np.exp(-BUDGET * h / 1000)
        lines += [f"### h = {h} windows{' (primary)' if h == H_SHORT else ''}: chance detection {chance_h:.0%}, ceiling {1 - chance_h:.2f}", "",
                  "| Lag | theta4 | 95% CI | p (one-sided, null: at most the margin) | Rule met |", "|---|---|---|---|---|"]
        horizons_out[str(h)] = {}
        for lag in LAGS:
            e2, l2, h2, p2 = theta4(detect_short[h][lag], np.random.default_rng(6 + h))
            horizons_out[str(h)][str(lag)] = {"estimate": e2, "lo": l2, "hi": h2, "p": p2}
            lines.append(f"| {lag} | {e2:+.3f} | [{l2:+.3f}, {h2:+.3f}] | {p2:.3f} | {'yes' if l2 > MARGIN else 'no'} |")
        lines.append("")
    short = horizons_out[str(H_SHORT)]
    lines += ["", "## Cells: share detected within the horizon (mean TAFD in windows)", "", "Undetected replicates count as the horizon.", ""]
    for lag in LAGS:
        lines += [f"### Label lag {lag} windows", "", "| Scenario | Level | Onset ratio at full effect | n | " + " | ".join(ARMS) + " |", "|---|---|---|---|---|---|"] + table[lag] + [""]
    (results_dir() / "rh4.json").write_text(json.dumps(headline, indent=1), encoding="utf-8")
    (results_dir() / "rh4_short.json").write_text(json.dumps(short, indent=1), encoding="utf-8")
    (results_dir() / "rh4_horizons.json").write_text(json.dumps(horizons_out, indent=1), encoding="utf-8")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", OUTPUT.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
