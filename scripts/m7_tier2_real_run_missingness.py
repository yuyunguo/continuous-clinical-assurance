"""Diagnostic companion to `m7_tier2_real_run.py`: the same real-data replicate pipeline, but with a genuinely
INPUT-affecting perturbation (SC05/SC06, generalized to a correlated column group) instead of SC17 version
regression, to check whether the label-free/CCA arms' disadvantage under SC17 (see `m7_tier2_real_run.py`'s
docstring) is specific to a model-only perturbation or holds for real data generally.

**Why a column group, not a single column.** A single-column real missingness diagnostic on the sepsis extraction
reached only a 1.04 D18 ratio even at prob=1.0 (always missing) -- the same feature-dilution finding that ruled out
SC07 rescale. Masking an entire correlated column group at once (e.g. one vital's mean/min/max/last/count together,
as one sensor or device dropping out would in practice) reached 1.25 -- short of Tier 1's "medium" (1.35) target but
above "low" (1.15), so this script solves against 1.15 and reports the ratio actually reached.

Usage: python scripts/m7_tier2_real_run_missingness.py --task {sepsis,readmission,mortality} [--vital-prefix vital_heart_rate_]
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
from src.eval_module.tafd import arm_reductions_with_holm_correction  # noqa: E402
from src.monitor_module import INDICATOR_ORDER, alert_rate, calibrate_threshold, indicator_indices  # noqa: E402
from src.sim_module.config import false_alarm_budget, load_config  # noqa: E402
from src.tier2_module import (RealDeployedModel, Tier2StreamConfig, assign_windows, group_missingness_and_rescore,  # noqa: E402
                              tier2_calibrate_arms, tier2_calibrate_onset, tier2_null_scores_multi,
                              tier2_onset_null_series_multi, tier2_ratio_for, tier2_replicate_tafd, tier2_solve_level)
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "phase2" / "results" / "tier2" / "features"
TRAIN_WINDOW_SHARE = 53 / 166
ONSET_TARGET_PER_1000 = 5.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, choices=("sepsis", "readmission", "mortality"))
    parser.add_argument("--vital-prefix", default="vital_heart_rate_", help="column-name prefix defining the correlated group that goes missing together")
    parser.add_argument("--target-ratio", type=float, default=1.15, help="D18 harm-ratio target for the missingness level (1.15, Tier 1's 'low', is the attainable ceiling for most task/group combinations; 1.35 is generally out of reach)")
    parser.add_argument("--window-size", type=int, default=500)
    args = parser.parse_args()

    path = FEATURES_DIR / f"{args.task}.parquet"
    if not path.exists():
        logger.error("missing %s -- run scripts/tier2_extract_features.py --task %s first", path, args.task)
        return 1

    set_seed(42)
    rng = np.random.default_rng(42)
    phase2_cfg = load_config()
    harm = phase2_cfg.harm

    raw = pd.read_parquet(path)
    feature_columns = tuple(c for c in raw.columns if c not in NON_FEATURE_COLUMNS)
    group_columns = tuple(c for c in feature_columns if c.startswith(args.vital_prefix))
    if not group_columns:
        logger.error("no feature columns start with %r", args.vital_prefix)
        return 1
    tier2_cfg = Tier2StreamConfig(feature_columns=feature_columns, window_size=args.window_size, target_sensitivity=0.8)
    logger.info("Task %s: %d real cases, %d real features, %d in the missingness group %s", args.task, len(raw),
               len(feature_columns), len(group_columns), group_columns)

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

    # Solve the group-missingness probability on the real population itself, targeting args.target_ratio (1.15, the
    # attainable target for this perturbation by default, per the module docstring). `ratio_at` must be deterministic
    # given the same `prob` for `brentq`'s root search to converge (mirroring `level_solving._ratio_at`'s
    # determinism, which has no randomness at all): a FRESH, fixed-seed rng per call, not a shared external one that
    # would advance and give a different, non-monotonic random draw on every evaluation.
    def ratio_at(prob: float) -> float:
        mask_rng = np.random.default_rng(11)

        def perturb(pop: pd.DataFrame) -> pd.DataFrame:
            modified = pop.copy()
            mask = mask_rng.random(len(pop)) < prob
            for c in group_columns:
                modified.loc[mask, c] = model.impute_means[c]
            return model.apply(modified)
        return tier2_ratio_for(scored, tier2_cfg, phase2_cfg, perturb)

    solution = tier2_solve_level(ratio_at, target_ratio=args.target_ratio, neutral=0.0, cap=1.0)
    if not solution.attainable or solution.value is None:
        logger.error("target ratio %.2f is unattainable even at prob=1.0 (max ratio reached: %.3f)", args.target_ratio, solution.max_ratio)
        return 1
    prob = solution.value
    logger.info("Solved group-missingness level: prob=%.4f gives D18 ratio %.3f (target %.2f)", prob, solution.ratio, args.target_ratio)

    onset_calib = tier2_onset_null_series_multi(windowed, tier2_cfg, phase2_cfg, harm, streams=300, seed=501,
                                                n_windows_out=n_null_windows_out, block_size=block_size, n_baseline=null_baseline)
    onset_threshold = tier2_calibrate_onset(onset_calib, ONSET_TARGET_PER_1000, j)
    logger.info("Calibrated onset threshold (target %.0f/1,000 windows): %.2f", ONSET_TARGET_PER_1000, onset_threshold)

    perturb_fn = functools.partial(group_missingness_and_rescore, cfg=tier2_cfg, model=model, columns=group_columns,
                                   level_value=prob, shape="abrupt", n_baseline=null_baseline, ramp_windows=1,
                                   rng=np.random.default_rng(909))
    horizon = min(20, n_null_windows_out - null_baseline - 1)
    # 500, not 150 (m7_tier2_real_run.py's SC17 count): this weaker, harder-to-detect scenario (D18 ratio 1.15, vs.
    # SC17's 1.35) yields far fewer surviving replicates per draw under the deliberately strict calibrated onset
    # rule (~13%, vs. SC17's ~40-50%) -- a larger n_replicates recovers a comparable absolute sample size without
    # loosening the onset threshold itself, mirroring the same fix already applied to a synthetic test earlier this
    # session for the identical reason (stricter rule -> fewer survivors -> more draws needed, not a looser rule).
    n_replicates = 500
    tafd = tier2_replicate_tafd(windowed, tier2_cfg, phase2_cfg, harm, arm_results, lag=lag, perturb_fn=perturb_fn,
                                n_replicates=n_replicates, n_windows_out=n_null_windows_out, n_baseline=null_baseline,
                                block_size=block_size, horizon=horizon, onset_threshold=onset_threshold,
                                rng=np.random.default_rng(303))
    used = len(tafd["conventional"])
    logger.info("Real-data replicate run (group missingness on %s, prob=%.4f): %d of %d replicates used", group_columns, prob, used, n_replicates)
    if used >= 5:
        results = arm_reductions_with_holm_correction(tafd, ARM_ORDER, "conventional",
                                                       ci_rng=np.random.default_rng(7), p_rng=np.random.default_rng(701))
        for name, r in results.items():
            sig = "*" if r["significant"] else " "
            logger.info("Reduction vs conventional, %-28s %+.2f windows [%+.2f, %+.2f] (95%% CI) p=%.4f%s", name,
                       r["reduction"], r["lo"], r["hi"], r["p"], sig)
    else:
        logger.warning("too few replicates survived (%d) for a meaningful comparison", used)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
