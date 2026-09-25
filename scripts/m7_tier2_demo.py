"""End-to-end demo of the Tier 2 stream builder on the synthetic stand-in.

Not a real-data run: it never touches MIMIC-IV or eICU. It shows that a cohort table with the schema
`scripts/tier2_extract_features.py` produces (order key, label, subgroup, feature columns) flows through
`Tier2StreamConfig` -> `RealDeployedModel` -> `assign_windows` -> `build_stream_data` into a `StreamData`, the
same shape `src.monitor_module` and `src.eval_module` already consume for Tier 1 — including matched false-alarm
calibration (via the block-bootstrap null streams) and detection on an injected input-only perturbation. RH1-4
evaluation (the "M5" half) and outcome-affecting scenarios (SC08, SC19) are still the next milestone.

Usage: python scripts/m7_tier2_demo.py
"""

import functools
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.m4_pilot import ARM_ORDER, FLAT_ARMS, SPLIT_ARMS  # noqa: E402
from src.eval_module.tafd import bootstrap_difference, rmst_difference  # noqa: E402
from src.monitor_module import (INDICATOR_ORDER, alert_rate, calibrate_threshold, component_groups, episode_starts,  # noqa: E402
                                indicator_indices, stream_scores_multi, system_events, system_events_hierarchical)
from src.sim_module.config import HarmConfig, false_alarm_budget, load_config  # noqa: E402
from src.tier2_module import (RealDeployedModel, Tier2StreamConfig, assign_windows, block_bootstrap_stream,  # noqa: E402
                              build_stream_data, make_synthetic_cohort, rescale_and_rescore, tier2_calibrate_arms,
                              tier2_calibrate_onset, tier2_null_scores_multi, tier2_onset_index_calibrated,
                              tier2_onset_null_series_multi, tier2_replicate_tafd)
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = tuple(f"f{i}" for i in range(8))


def main() -> None:
    set_seed(42)
    rng = np.random.default_rng(42)
    phase2_cfg = load_config()
    harm: HarmConfig = phase2_cfg.harm
    tier2_cfg = Tier2StreamConfig(feature_columns=FEATURE_COLUMNS, window_size=500, target_sensitivity=0.8)

    cohort = make_synthetic_cohort(n=50_000, feature_columns=FEATURE_COLUMNS, rng=rng, prevalence=0.15)
    train, rest = cohort.iloc[:10_000], cohort.iloc[10_000:]
    model = RealDeployedModel(tier2_cfg).fit(train, rng)
    scored = model.apply(rest)
    windowed = assign_windows(scored, tier2_cfg)
    n_windows = windowed["window"].nunique()
    n_baseline = max(1, n_windows // 5)
    stream = build_stream_data(windowed, tier2_cfg, harm, n_baseline=n_baseline, rng=rng)

    logger.info("Synthetic Tier 2 demo: %d windows (%d baseline, %d monitored), %d cases/window",
               n_windows, n_baseline, n_windows - n_baseline, tier2_cfg.window_size)
    logger.info("Pooled prevalence: %.3f, pooled sensitivity at threshold: %.3f",
               float(stream.y.mean()), float(stream.decision[stream.y == 1].mean()))
    logger.info("Severity mix (all windows): %s", [float((stream.severity == c).mean()) for c in (1, 2, 3)])

    # An input-only perturbation (SC07 rescale), injected onto the same fixed cohort from window s0=0: real labels
    # are untouched, only the model's input and its rescoring change.
    perturbed = rescale_and_rescore(windowed, tier2_cfg, model, column="f0", level_value=3.0, shape="abrupt",
                                    s0=0, n_baseline=n_baseline, ramp_windows=1)
    perturbed_stream = build_stream_data(perturbed, tier2_cfg, harm, n_baseline=n_baseline, rng=rng)
    logger.info("After SC07 rescale (x3 on f0, full effect): pooled sensitivity %.3f -> %.3f",
               float(stream.decision[stream.y == 1].mean()), float(perturbed_stream.decision[perturbed_stream.y == 1].mean()))

    # A block-bootstrap null stream for false-alarm calibration: resampled from the same real (here, synthetic-stand-in)
    # windows, since a single real dataset cannot be redrawn like the synthetic generator.
    null_stream = block_bootstrap_stream(windowed, tier2_cfg, harm, n_windows_out=64, block_size=8, rng=rng)
    logger.info("Block-bootstrap null stream: %d windows, pooled prevalence %.3f (vs. real %.3f)",
               null_stream.y.shape[0], float(null_stream.y.mean()), float(stream.y.mean()))

    # Matched false-alarm calibration (the "M4" half of the M4/M5 arm wiring), reusing src.monitor_module unchanged:
    # calibrate the CUSUM threshold on one batch of block-bootstrap null streams, check it on an independent batch.
    lag = phase2_cfg.stream.label_lag   # decision D24
    j = phase2_cfg.routing.hysteresis_j
    budget = false_alarm_budget()       # decision D2
    cols = indicator_indices(list(INDICATOR_ORDER))
    calib = tier2_null_scores_multi(windowed, tier2_cfg, phase2_cfg, harm, streams=300, seed=101, lags=(lag,),
                                    n_windows_out=64, block_size=8, n_baseline=16)[lag]
    evaluation = tier2_null_scores_multi(windowed, tier2_cfg, phase2_cfg, harm, streams=300, seed=202, lags=(lag,),
                                         n_windows_out=64, block_size=8, n_baseline=16)[lag]
    threshold = calibrate_threshold(calib, cols, budget, j)
    achieved = alert_rate(evaluation, cols, threshold, j)
    logger.info("Calibrated threshold at budget %.0f/1,000 windows: %.2f (achieved %.2f on an independent evaluation batch)",
               budget, threshold, achieved)

    # Detection: does the calibrated system raise an alert episode on the SC07-perturbed stream, and when?
    z = stream_scores_multi(perturbed_stream, phase2_cfg, (lag,))[lag]
    events = system_events(z[None], cols, threshold)
    starts = episode_starts(events, j)[0]
    first_alert = int(np.argmax(starts)) if starts.any() else None
    logger.info("First alert-episode window on the perturbed stream (onset at monitored window 0): %s", first_alert)

    # Multi-arm calibration: all seven M4-pilot arms (conventional, label-free, calibration, CCA union, CCA
    # component-split, tilted, calibration-only), reusing scripts.m4_pilot's exact arm definitions so Tier 1 and
    # Tier 2 compare the same indicator sets. Each arm's own calibrated threshold and its first alert on the
    # perturbed stream, so the comparison Tier 1's M4/M5 pilot makes is now visible on Tier 2 architecture too.
    arm_results = tier2_calibrate_arms(calib, evaluation, budget, j)
    arm_groups = component_groups(INDICATOR_ORDER)
    for name, (arm_threshold, rate, se) in arm_results.items():
        shown = f"{arm_threshold:.2f}" if isinstance(arm_threshold, float) else ", ".join(f"{k} {v:.2f}" for k, v in arm_threshold.items())
        if name in SPLIT_ARMS:
            arm_events = system_events_hierarchical(z[None], arm_groups, arm_threshold)  # type: ignore[arg-type]
        else:
            arm_events = system_events(z[None], indicator_indices(FLAT_ARMS[name]), arm_threshold)  # type: ignore[arg-type]
        arm_starts = episode_starts(arm_events, j)[0]
        arm_first = int(np.argmax(arm_starts)) if arm_starts.any() else None
        logger.info("Arm %-28s threshold %-20s achieved %6.2f (SE %.2f), first alert on perturbed stream: %s",
                   name, shown, rate, se, arm_first)

    # Real onset detection (the "M5" half of the arm wiring), CALIBRATED: investigation on real MIMIC-IV data found
    # the naive single-window threshold rule false-fires on ~100% of pure-null replicates (a multiple-comparisons
    # problem -- scanning many correlated single-window tests against a fixed ratio threshold). The fix reuses the
    # SAME CUSUM + calibrate_threshold machinery already used for the monitoring arms, applied to the D18 harm-ratio
    # series instead (see onset_calibration.py). Calibrated here at a stricter target than the monitoring budget
    # (5, not 30, per 1,000 windows): ground-truth onset should be a rare, confident judgment, not a routine alert.
    onset_calib = tier2_onset_null_series_multi(windowed, tier2_cfg, phase2_cfg, harm, streams=300, seed=401,
                                                n_windows_out=64, block_size=8, n_baseline=16)
    onset_threshold = tier2_calibrate_onset(onset_calib, target_per_1000=5.0, j=j)
    onset = tier2_onset_index_calibrated(perturbed_stream, phase2_cfg, onset_threshold, j)
    logger.info("Calibrated onset threshold: %.2f. Real onset window on the perturbed stream: %s", onset_threshold, onset)
    if onset is not None and first_alert is not None:
        logger.info("TAFD (time after first detection): %d windows", first_alert - onset)

    # Multi-replicate statistics (the full "M5" comparison): 150 replicates, mirroring m4_pilot.py's own REPLICATES.
    # Each replicate is a fresh block-bootstrap resample of the real windows with the SC07 rescale injected from a
    # randomly chosen onset window, classified by src.eval_module.tafd.classify_replicate -- reused unchanged, with
    # no Tier 2-specific version needed. Onset uses the SAME calibrated rule as above, not the naive one.
    replicate_perturb_fn = functools.partial(rescale_and_rescore, cfg=tier2_cfg, model=model, column="f0",
                                             level_value=3.0, shape="abrupt", n_baseline=16, ramp_windows=1)
    tafd = tier2_replicate_tafd(windowed, tier2_cfg, phase2_cfg, harm, arm_results, lag=lag,
                                perturb_fn=replicate_perturb_fn, n_replicates=150, n_windows_out=64, n_baseline=16,
                                block_size=8, horizon=20, onset_threshold=onset_threshold, rng=np.random.default_rng(303))
    used = len(tafd["conventional"])
    logger.info("Multi-replicate TAFD comparison: %d of 150 replicates used (rest dropped: no real onset, or onset "
               "too close to the horizon)", used)
    if used >= 5:
        rng_boot = np.random.default_rng(7)
        for name in ARM_ORDER:
            if name == "conventional":
                continue
            reduction = rmst_difference(tafd["conventional"], tafd[name])
            lo, hi = bootstrap_difference(tafd["conventional"], tafd[name], rng_boot)
            logger.info("Reduction vs conventional, %-28s %+.2f windows [%+.2f, %+.2f] (95%% CI)", name, reduction, lo, hi)


if __name__ == "__main__":
    main()
