"""Milestone M5g: the process scenarios SC12 (blanket rejection) and SC15 (duplicate alert burden), first pass. SC15 is also run with alert fatigue coupled to reviewers.

Usage: python scripts/m5g_process_scenarios.py
Neither scenario has a harm-defined onset (design section 5), so the clock starts at the drift start and TAFD is the time from it to the first alarm. Each system
is calibrated to the same false-alarm budget on independent null streams. SC12 asks whether oversight indicators notice indiscriminate rejection, which raises
EHOR and IIR together and must not be read as better oversight. SC15 asks whether the alert-burden indicator O3 notices duplicated alerts. Writes phase2/M5g-Process-Scenarios.md.
Exploratory, one seed, assumed reviewer and workflow models. The action-rate part of O3 is not simulated.
"""

import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_module.tafd import classify_replicate  # noqa: E402
from src.monitor_module import INDICATOR_ORDER, build_stream, calibrate_threshold, episode_starts, progress_schedule, stream_scores_multi, system_events  # noqa: E402
from src.monitor_module.review_layer import REVIEW_ORDER, build_review_params, review_scores_multi  # noqa: E402
from src.monitor_module.workflow_layer import WORKFLOW_ORDER, build_workflow, workflow_scores_multi  # noqa: E402
from src.sim_module import DeployedModel, SyntheticWorld, load_config, parse_level  # noqa: E402
from src.sim_module.config import scenarios_path  # noqa: E402
from src.sim_module.joint_schedule import joint_schedule  # noqa: E402
from src.sim_module.workflow_harm import WorkflowRatio  # noqa: E402
from src.sim_module.config import false_alarm_budget  # noqa: E402
from src.utils import log_environment, set_seed  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "phase2" / "M5g-Process-Scenarios.md"
N_NULL, REPLICATES, LAGS, AUDIT = int(os.environ.get("M5G_NULL", 800)), int(os.environ.get("M5G_REPS", 200)), (0, 8), 0.2
BUDGET = false_alarm_budget()
DEADLINE, COVERAGE = 5.0, 0.95
K_MODEL, K_REVIEW = len(INDICATOR_ORDER), len(REVIEW_ORDER)
NAMES = (*INDICATOR_ORDER, *REVIEW_ORDER, *WORKFLOW_ORDER)
def cols(*names: str) -> List[int]:
    """Column indices in the score array (model-level, then oversight, then workflow)."""
    return [NAMES.index(n) for n in names]


ARMS: Dict[str, List[int]] = {
    "model-level (P, U, C)": list(range(K_MODEL)), "oversight (H1 to H6, H2u)": list(range(K_MODEL, K_MODEL + K_REVIEW)),
    "H3 alone (appropriate override rate)": cols("H3"), "H2 downward arm alone (the first build)": cols("H2"), "H2 two-sided (H2 and H2u)": cols("H2", "H2u"),
    "workflow (H7, O1 to O6, O3a)": list(range(K_MODEL + K_REVIEW, len(NAMES))), "O3 burden alone": cols("O3"), "O3 action rate alone": cols("O3a")}
# label, scenario, arms, fatigue bias (extra automation bias from duplicated alerts; 0 in the primary)
CELLS: List[Tuple[str, str, List[str], float]] = [
    ("SC12", "SC12", ["model-level (P, U, C)", "oversight (H1 to H6, H2u)", "H3 alone (appropriate override rate)", "H2 downward arm alone (the first build)", "H2 two-sided (H2 and H2u)"], 0.0),
    ("SC15", "SC15", ["model-level (P, U, C)", "workflow (H7, O1 to O6, O3a)", "O3 burden alone", "O3 action rate alone"], 0.0),
    ("SC15 + fatigue", "SC15", ["model-level (P, U, C)", "oversight (H1 to H6, H2u)", "workflow (H7, O1 to O6, O3a)", "O3 action rate alone"], 0.3)]
REVIEWER_KEYS = ("coverage", "automation_bias", "false_reject", "latency_median")
WORKFLOW_KEYS = ("p_deliver", "tat_mult", "alert_dup", "verify")


def scores(stream: Any, cfg: Any, par: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[int, np.ndarray]:
    """Model-level, oversight, and workflow scores side by side, per lag. Shape (T, 26)."""
    review = build_review_params(stream, cfg, {k: par[k] for k in REVIEWER_KEYS}, rng)
    workflow = build_workflow(stream, review, cfg, {k: par[k] for k in WORKFLOW_KEYS}, rng)
    parts = (stream_scores_multi(stream, cfg, LAGS), review_scores_multi(stream, review, AUDIT, LAGS, rng), workflow_scores_multi(stream, workflow, LAGS))
    return {lag: np.concatenate([p[lag] for p in parts], axis=1) for lag in LAGS}


def null_scores(world: SyntheticWorld, model: DeployedModel, cfg: Any, seed: int) -> Dict[int, np.ndarray]:
    """Scores of N_NULL independent null streams, per lag. Shape (S, T, 26)."""
    par = joint_schedule("alert_duplication", 1.0, np.zeros(cfg.stream.baseline_windows + cfg.stream.monitored_windows), cfg)
    per = [scores(build_stream(world, model, None, "abrupt", 0, cfg, np.random.default_rng(seed * 100_003 + i)), cfg, par, np.random.default_rng(seed * 7919 + i)) for i in range(N_NULL)]
    return {lag: np.stack([p[lag] for p in per]) for lag in LAGS}


def main() -> int:
    """Run the cells and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg0 = load_config()
    cfg = replace(cfg0, reviewer=replace(cfg0.reviewer, deadline=DEADLINE, coverage=COVERAGE))
    set_seed(cfg.seed + 3)
    world = SyntheticWorld(cfg.generator, cfg.harm)
    model = DeployedModel(cfg.generator).fit(world, np.random.default_rng(cfg.seed))
    j, horizon, total, b = cfg.routing.hysteresis_j, cfg.stream.horizon, cfg.stream.monitored_windows, cfg.stream.baseline_windows
    scenarios = {s["id"]: s for s in yaml.safe_load(scenarios_path().read_text(encoding="utf-8"))}
    calib, evaluation = null_scores(world, model, cfg, 707), null_scores(world, model, cfg, 808)
    thr = {lag: {a: calibrate_threshold(calib[lag], c, BUDGET, j) for a, c in ARMS.items()} for lag in LAGS}
    cal_rows = [f"| {lag} | {a} | {thr[lag][a]:.2f} | {episode_starts(system_events(evaluation[lag], c, thr[lag][a]), j).mean() * 1000:.2f} |" for lag in LAGS for a, c in ARMS.items()]
    tables: Dict[int, List[str]] = {lag: [] for lag in LAGS}
    population = model.apply(world.sample(200_000, np.random.default_rng(11)))
    for label, sid, arm_names, fatigue_bias in CELLS:
        cfg_v = replace(cfg, workflow=replace(cfg.workflow, fatigue_bias=fatigue_bias))
        ratio = WorkflowRatio(cfg_v, population)
        spec = scenarios[sid]["perturbation"]
        for li, level in enumerate(("low", "medium", "high")):
            value, kind = parse_level(spec["levels"][level]), spec["type"]
            rng_s0 = np.random.default_rng(14_000 + 10 * int(sid[2:]) + li)
            res: Dict[int, Dict[str, List[Tuple[int, bool]]]] = {lag: {a: [] for a in arm_names} for lag in LAGS}
            for r in range(REPLICATES):
                s0 = int(rng_s0.integers(cfg.stream.s0_low, cfg.stream.s0_high_abrupt + 1))
                par = joint_schedule(kind, value, np.concatenate([np.zeros(b), progress_schedule("abrupt", s0, total, cfg.stream.ramp_windows)]), cfg_v)
                rng = np.random.default_rng(9_000_000 + int(sid[2:]) * 10_000 + li * 1000 + r)
                zs = scores(build_stream(world, model, None, "abrupt", s0, cfg, rng), cfg, par, rng)
                for lag in LAGS:
                    for a in arm_names:
                        out = classify_replicate(system_events(zs[lag][None], ARMS[a], thr[lag][a])[0], s0, s0, horizon, j)
                        res[lag][a].append((int(out["tafd"]), not out["censored"]))
            full_ratio = ratio({k: float(v[0]) for k, v in joint_schedule(kind, value, np.array([1.0]), cfg_v).items()})
            logger.info("%s %s done", label, level)
            for lag in LAGS:
                fmt = lambda v: f"{np.mean([d for _, d in v]):.0%} ({np.mean([t for t, _ in v]):.1f})"  # noqa: E731
                tables[lag].append(f"| {label} | {level} | {value:g} | {full_ratio:.2f} | " + " | ".join(f"{a}: {fmt(res[lag][a])}" for a in arm_names) + " |")
    lines = ["# Phase II M5g: process scenarios SC12 and SC15, first pass (exploratory)", "",
             f"Generated by `scripts/m5g_process_scenarios.py`. SC12 (blanket rejection: added probability of rejecting a correct output) and SC15 (alert duplication: alerts per alerted case) have no harm-defined onset, "
             f"so TAFD is the time from the drift start to the first alarm, with undetected replicates counted as the horizon of {horizon}. {REPLICATES} replicates per cell, each system calibrated to "
             f"{BUDGET:g} false alarms per 1,000 windows on {N_NULL} independent null streams. Share detected within the horizon, then mean TAFD in windows in parentheses. "
             f"A detection share for the model-level arm is its chance level, since neither scenario changes the model. Reviewer coverage {COVERAGE:.0%}, deadline {DEADLINE:g} median reviews, and the workflow model are assumptions. "
             f"O3 now has both halves: burden (O3) and an action rate that falls with duplication by a fatigue model (O3a). 'SC15 + fatigue' also lets duplicated alerts raise reviewers' automation bias (by up to 0.3 at full duplication), the indirect harm the catalog describes, which is off in the primary; the harm ratio shows its size (a value above 1 is added expected harm relative to baseline, and SC15 without fatigue has none by design). SC12's H2 is now charted in both directions (H2 downward and H2u upward), as its indicator record specifies. Environment: {log_environment()}", "", "## Calibration", "", "| Lag | Arm | Threshold | Achieved per 1,000 windows |", "|---|---|---|---|"] + cal_rows + [""]
    for lag in LAGS:
        lines += [f"## Label lag {lag} windows", "", "| Scenario | Level | Value | Harm ratio at full effect | Arms: share detected (mean TAFD) |", "|---|---|---|---|---|"] + tables[lag] + [""]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", OUTPUT.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
