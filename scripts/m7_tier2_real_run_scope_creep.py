"""SC19 (scope creep) on real data: a growing share of each monitored window is replaced with real, genuinely
out-of-scope cases from eICU-CRD (`scripts/tier2_extract_eicu_pool.py`) -- a separate hospital system a MIMIC-IV-
trained model has never seen. Every replaced case keeps its own real label; nothing about any outcome is invented
(`Tier2-Readiness.md` SS11). This also gives the C3 indicator real content for the first time in Tier 2: it is a
hardcoded constant 1.0 in every other real-data script here, since `stream.out_of_scope` was always `None`.

Usage: python scripts/m7_tier2_real_run_scope_creep.py --task {sepsis,readmission,mortality} [--target-ratio 1.15]
"""

import argparse
import functools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.m4_pilot import ARM_ORDER  # noqa: E402
from scripts.tier2_extract_features import NON_FEATURE_COLUMNS  # noqa: E402
from src.eval_module.tafd import bootstrap_difference, rmst_difference  # noqa: E402
from src.monitor_module import INDICATOR_ORDER, alert_rate, calibrate_threshold, indicator_indices  # noqa: E402
from src.sim_module.config import false_alarm_budget, load_config  # noqa: E402
from src.tier2_module import (RealDeployedModel, Tier2StreamConfig, assign_windows, scope_creep_and_rescore,  # noqa: E402
                              tier2_calibrate_arms, tier2_calibrate_onset, tier2_null_scores_multi,
                              tier2_onset_null_series_multi, tier2_ratio_for, tier2_replicate_tafd, tier2_solve_level)
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "phase2" / "results" / "tier2" / "features"
TRAIN_WINDOW_SHARE = 53 / 166
ONSET_TARGET_PER_1000 = 5.0
N_REPLICATES = 500   # matches m7_tier2_real_run_missingness.py's count -- this scenario is similarly weak-signal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, choices=("sepsis", "readmission", "mortality"))
    parser.add_argument("--target-ratio", type=float, default=1.15, help="D18 harm-ratio target for the scope-creep level")
    parser.add_argument("--window-size", type=int, default=500)
    args = parser.parse_args()

    path = FEATURES_DIR / f"{args.task}.parquet"
    pool_path = FEATURES_DIR / "eicu_out_of_scope_pool.parquet"
    if not path.exists():
        logger.error("missing %s -- run scripts/tier2_extract_features.py --task %s first", path, args.task)
        return 1
    if not pool_path.exists():
        logger.error("missing %s -- run scripts/tier2_extract_eicu_pool.py first", pool_path)
        return 1

    set_seed(42)
    rng = np.random.default_rng(42)
    phase2_cfg = load_config()
    harm = phase2_cfg.harm

    raw = pd.read_parquet(path)
    eicu_pool = pd.read_parquet(pool_path)
    feature_columns = tuple(c for c in raw.columns if c not in NON_FEATURE_COLUMNS)
    missing_in_pool = set(feature_columns) - set(eicu_pool.columns)
    if missing_in_pool:
        logger.error("eICU pool is missing feature columns the model needs: %s", sorted(missing_in_pool))
        return 1
    tier2_cfg = Tier2StreamConfig(feature_columns=feature_columns, window_size=args.window_size, target_sensitivity=0.8)
    logger.info("Task %s: %d real cases, %d real features; out-of-scope pool: %d real eICU cases", args.task,
               len(raw), len(feature_columns), len(eicu_pool))

    ordered = raw.sort_values("order_key", kind="stable").reset_index(drop=True)
    n_windows_total = len(ordered) // args.window_size
    n_baseline = max(1, round(n_windows_total * TRAIN_WINDOW_SHARE))
    train = ordered.iloc[: n_baseline * args.window_size]
    model = RealDeployedModel(tier2_cfg).fit(train, rng)
    scored = model.apply(ordered)
    windowed = assign_windows(scored, tier2_cfg)
    n_windows = windowed["window"].nunique()
    logger.info("%d windows (%d cases each): %d fit-period, %d monitored", n_windows, args.window_size, n_baseline, n_windows - n_baseline)

    lag = phase2_cfg.stream.label_lag
    j = phase2_cfg.routing.hysteresis_j
    budget = false_alarm_budget()
    n_null_windows_out = min(64, n_windows)
    null_baseline = min(16, n_null_windows_out - 1)
    block_size = min(8, n_null_windows_out - null_baseline)
    calib = tier2_null_scores_multi(windowed, tier2_cfg, phase2_cfg, harm, streams=300, seed=101, lags=(lag,),
                                    n_windows_out=n_null_windows_out, block_size=block_size, n_baseline=null_baseline)[lag]
    evaluation = tier2_null_scores_multi(windowed, tier2_cfg, phase2_cfg, harm, streams=300, seed=202, lags=(lag,),
                                         n_windows_out=n_null_windows_out, block_size=block_size, n_baseline=null_baseline)[lag]
    cols = indicator_indices(list(INDICATOR_ORDER))
    threshold = calibrate_threshold(calib, cols, budget, j)
    achieved = alert_rate(evaluation, cols, threshold, j)
    logger.info("Real-data matched calibration at budget %.0f/1,000 windows: threshold %.2f, achieved %.2f", budget, threshold, achieved)

    arm_results = tier2_calibrate_arms(calib, evaluation, budget, j)
    for name, (arm_threshold, rate, se) in arm_results.items():
        shown = f"{arm_threshold:.2f}" if isinstance(arm_threshold, float) else ", ".join(f"{k} {v:.2f}" for k, v in arm_threshold.items())
        logger.info("Arm %-28s threshold %-20s achieved %6.2f (SE %.2f)", name, shown, rate, se)

    # Solve the scope-creep share on the real population itself, targeting args.target_ratio. `ratio_at` must be
    # deterministic given the same `share` for `brentq`'s root search (mirroring the missingness/level_solving
    # scripts' pattern): a FRESH, fixed-seed rng per call.
    scored_pool = model.apply(eicu_pool)
    replace_columns = list(feature_columns) + [tier2_cfg.label_column, tier2_cfg.subgroup_column]

    def ratio_at(share: float) -> float:
        replace_rng = np.random.default_rng(11)

        def perturb(pop: pd.DataFrame) -> pd.DataFrame:
            modified = pop.copy()
            k = round(share * len(pop))
            if k <= 0:
                return model.apply(modified)
            replace_index = replace_rng.choice(modified.index.to_numpy(), size=k, replace=False)
            draw_positions = replace_rng.integers(0, len(scored_pool), size=k)
            modified.loc[replace_index, replace_columns] = scored_pool.iloc[draw_positions][replace_columns].to_numpy()
            return model.apply(modified)
        return tier2_ratio_for(scored, tier2_cfg, phase2_cfg, perturb)

    solution = tier2_solve_level(ratio_at, target_ratio=args.target_ratio, neutral=0.0, cap=1.0)
    if not solution.attainable or solution.value is None:
        logger.error("target ratio %.2f is unattainable even at share=1.0 (max ratio reached: %.3f)", args.target_ratio, solution.max_ratio)
        return 1
    share = solution.value
    logger.info("Solved scope-creep level: share=%.4f gives D18 ratio %.3f (target %.2f)", share, solution.ratio, args.target_ratio)

    onset_calib = tier2_onset_null_series_multi(windowed, tier2_cfg, phase2_cfg, harm, streams=300, seed=501,
                                                n_windows_out=n_null_windows_out, block_size=block_size, n_baseline=null_baseline)
    onset_threshold = tier2_calibrate_onset(onset_calib, ONSET_TARGET_PER_1000, j)
    logger.info("Calibrated onset threshold (target %.0f/1,000 windows): %.2f", ONSET_TARGET_PER_1000, onset_threshold)

    perturb_fn = functools.partial(scope_creep_and_rescore, cfg=tier2_cfg, model=model, out_of_scope_pool=eicu_pool,
                                   level_value=share, shape="abrupt", n_baseline=null_baseline, ramp_windows=1,
                                   rng=np.random.default_rng(909))
    horizon = min(20, n_null_windows_out - null_baseline - 1)
    tafd = tier2_replicate_tafd(windowed, tier2_cfg, phase2_cfg, harm, arm_results, lag=lag, perturb_fn=perturb_fn,
                                n_replicates=N_REPLICATES, n_windows_out=n_null_windows_out, n_baseline=null_baseline,
                                block_size=block_size, horizon=horizon, onset_threshold=onset_threshold,
                                rng=np.random.default_rng(303))
    used = len(tafd["conventional"])
    logger.info("Real-data replicate run (scope creep, share=%.4f): %d of %d replicates used", share, used, N_REPLICATES)
    if used >= 5:
        rng_boot = np.random.default_rng(7)
        for name in ARM_ORDER:
            if name == "conventional":
                continue
            reduction = rmst_difference(tafd["conventional"], tafd[name])
            lo, hi = bootstrap_difference(tafd["conventional"], tafd[name], rng_boot)
            logger.info("Reduction vs conventional, %-28s %+.2f windows [%+.2f, %+.2f] (95%% CI)", name, reduction, lo, hi)
    else:
        logger.warning("too few replicates survived (%d) for a meaningful comparison", used)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
