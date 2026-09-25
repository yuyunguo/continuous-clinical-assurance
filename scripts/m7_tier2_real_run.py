"""The Tier 2 real-data run: the same pipeline `m7_tier2_demo.py` demonstrates on the synthetic stand-in, now on
the real MIMIC-IV extraction (`scripts/tier2_extract_features.py`'s output).

Reads only the local Parquet file `phase2/results/tier2/features/<task>.parquet` — never the live database. No
row-level value (a feature, a label, a prediction) is ever printed; every reported number here is an aggregate
(a count, a rate, a mean, a threshold, a bootstrap interval). See `Tier2-Readiness.md` for what "real data" means
here and its limitations (one dataset, one task at a time, one injected scenario per run).

**Perturbation: SC17 version regression, not SC07 rescale.** The first real-data run used SC07 (rescale one
feature) at Tier 1's own magnitude and found it produced no detectable real harm change on real, 115-feature data
at any level (checked up to a 200x rescale) — a single feature's influence is diluted across many correlated real
features after regularization. SC17 (a fraction of the deployed model's weight vector replaced) is not diluted the
same way and was confirmed attainable. Its level is solved on the real population itself (`tier2_solve_level`),
against Tier 1's own SC17 harm target (1.35, its "medium" level in `phase2/scenarios.yaml`), not a magnitude
carried over unchanged from the synthetic stand-in.

**Onset: calibrated, not naive.** `tier2_onset_index`/`tier2_onset_index_rolling`'s single-window threshold test
false-fires on ~100% of pure-null real replicates (a multiple-comparisons problem). Onset here is calibrated the
same way the monitoring arms already are (`tier2_calibrate_onset`), at a stricter target (5, not the monitoring
budget's 30, per 1,000 windows) since ground-truth onset should be a rare, confident judgment.

Usage: python scripts/m7_tier2_real_run.py --task {sepsis,readmission,mortality} [--target-ratio 1.35]
"""

import argparse
import functools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.m4_pilot import ARM_ORDER  # noqa: E402
from scripts.tier2_extract_features import NON_FEATURE_COLUMNS  # noqa: E402
from src.eval_module.tafd import arm_reductions_with_holm_correction  # noqa: E402
from src.monitor_module import INDICATOR_ORDER, alert_rate, calibrate_threshold, indicator_indices  # noqa: E402
from src.sim_module.config import false_alarm_budget, load_config  # noqa: E402
from src.tier2_module import (RealDeployedModel, Tier2StreamConfig, assign_windows, tier2_calibrate_arms,  # noqa: E402
                              tier2_calibrate_onset, tier2_null_scores_multi, tier2_onset_null_series_multi,
                              tier2_ratio_for, tier2_replicate_tafd, tier2_solve_level, version_regression_and_rescore)
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "phase2" / "results" / "tier2" / "features"
TRAIN_WINDOW_SHARE = 53 / 166   # matches the 2008-2010 fit period's share of total windows (Tier2-Readiness.md SS6)
ONSET_TARGET_PER_1000 = 5.0     # stricter than the 30/1,000 monitoring budget: onset should be a rare judgment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, choices=("sepsis", "readmission", "mortality"))
    parser.add_argument("--target-ratio", type=float, default=1.35, help="D18 harm-ratio target for the SC17 level (Tier 1's own 'medium' is 1.35)")
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
    tier2_cfg = Tier2StreamConfig(feature_columns=feature_columns, window_size=args.window_size, target_sensitivity=0.8)
    logger.info("Task %s: %d real cases, %d real features, pooled prevalence %.3f", args.task, len(raw),
               len(feature_columns), raw["label"].mean())

    ordered = raw.sort_values("order_key", kind="stable").reset_index(drop=True)
    n_windows_total = len(ordered) // args.window_size
    n_baseline = max(1, round(n_windows_total * TRAIN_WINDOW_SHARE))
    train = ordered.iloc[: n_baseline * args.window_size]
    model = RealDeployedModel(tier2_cfg).fit(train, rng)
    scored = model.apply(ordered)
    windowed = assign_windows(scored, tier2_cfg)
    n_windows = windowed["window"].nunique()
    logger.info("%d windows (%d cases each): %d fit-period, %d monitored", n_windows, args.window_size, n_baseline, n_windows - n_baseline)

    monitored_mask = windowed["window"] >= n_baseline
    auroc = roc_auc_score(windowed.loc[monitored_mask, "label"], windowed.loc[monitored_mask, "p_hat"])
    sensitivity = windowed.loc[monitored_mask & (windowed["label"] == 1), "decision"].mean()
    logger.info("Deployed model on held-out monitored windows: AUROC %.3f, sensitivity at threshold %.3f", auroc, sensitivity)

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

    # Solve the SC17 (version regression) level on the real population itself, targeting Tier 1's own harm ratio,
    # instead of reusing a magnitude picked for the synthetic stand-in.
    solution = tier2_solve_level(lambda f: tier2_ratio_for(scored, tier2_cfg, phase2_cfg, lambda pop: model.with_regression(f, seed=1).apply(pop)),
                                 target_ratio=args.target_ratio, neutral=0.0, cap=1.0)
    if not solution.attainable or solution.value is None:
        logger.error("target ratio %.2f is unattainable even at full regression (max ratio reached: %.3f)",
                     args.target_ratio, solution.max_ratio)
        return 1
    fraction = solution.value
    logger.info("Solved SC17 level: fraction=%.4f gives D18 ratio %.3f (target %.2f)", fraction, solution.ratio, args.target_ratio)

    # Calibrate onset (see the module docstring): stricter than the monitoring budget, since ground-truth onset
    # should be a rare, confident judgment, not a routine alert.
    onset_calib = tier2_onset_null_series_multi(windowed, tier2_cfg, phase2_cfg, harm, streams=300, seed=501,
                                                n_windows_out=n_null_windows_out, block_size=block_size, n_baseline=null_baseline)
    onset_threshold = tier2_calibrate_onset(onset_calib, ONSET_TARGET_PER_1000, j)
    logger.info("Calibrated onset threshold (target %.0f/1,000 windows): %.2f", ONSET_TARGET_PER_1000, onset_threshold)

    perturb_fn = functools.partial(version_regression_and_rescore, cfg=tier2_cfg, model=model, level_value=fraction,
                                   shape="abrupt", n_baseline=null_baseline, ramp_windows=1)
    horizon = min(20, n_null_windows_out - null_baseline - 1)
    tafd = tier2_replicate_tafd(windowed, tier2_cfg, phase2_cfg, harm, arm_results, lag=lag, perturb_fn=perturb_fn,
                                n_replicates=150, n_windows_out=n_null_windows_out, n_baseline=null_baseline,
                                block_size=block_size, horizon=horizon, onset_threshold=onset_threshold,
                                rng=np.random.default_rng(303))
    used = len(tafd["conventional"])
    logger.info("Real-data replicate run (SC17 version regression, fraction=%.4f): %d of 150 replicates used", fraction, used)
    if used >= 5:
        # ci_rng=default_rng(7) matches every prior real-data run's own CI computation exactly (bit-identical, not
        # just statistically equivalent); p_rng is a separate, independent bootstrap draw used only for the
        # Holm-Bonferroni significance decision (manuscript Methods Sec. 2.6) -- it does not affect the CI at all.
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
