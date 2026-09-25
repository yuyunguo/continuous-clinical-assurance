"""SC08 (concept drift) on real data: the real, naturally-occurring 2020-2022 mortality rise, used as a found
natural experiment instead of an injected perturbation.

**Why no injection.** SC08 works by changing the true label-generating process (a relative sensitivity loss on the
synthetic stand-in). Injecting that onto an already-observed real cohort would mean asserting "this patient's real
outcome would have been different under a changed disease process" -- fabricating a counterfactual on a real
patient, a scientific-integrity line the study author chose not to cross (`Tier2-Readiness.md` SS3/SS10). Instead,
this script runs the calibrated onset detector and the monitoring arms directly on the REAL, chronologically
ordered mortality stream -- no resampling, no injected perturbation -- and asks whether the real 2020-2022 mortality
rise (8.75% to 11.1% by calendar-period aggregate, `Tier2-Readiness.md` SS6 / `aggregates_mimiciv.json`) is itself
detectable as a real D18 onset, and which arms catch it first.

**A real ordering bug, found and fixed.** `order_key` (`icu_intime`) is shifted independently per patient by
MIMIC-IV's de-identification (confirmed empirically: a run against `order_key` alone showed a "calendar range" of
2110 to 2214 -- a 100-plus-year spread for a decade of real admissions). MIMIC-IV's own documentation is explicit
that shifted dates are NOT comparable across patients, and names `anchor_year_group` (a real, de-identification-safe
3-year calendar bucket) as the sanctioned field for exactly this kind of cross-patient temporal analysis. This
script windows by a COMPOSITE key: primarily `anchor_year_group` (so no window ever mixes cases from different real
calendar periods -- the property that actually matters), with `order_key` only as a same-bucket tiebreak (its
within-bucket ordering is not claimed to be meaningful, only deterministic). Requires `anchor_year_group` in the
extracted Parquet -- re-run `scripts/tier2_extract_features.py` after this session's update to `NON_FEATURE_COLUMNS`
if your existing file predates it.

**The monitored period is the 2020-2022 bucket specifically** (not an arbitrary calendar-agnostic split): training
on every earlier bucket (2008-2019) and monitoring only within 2020-2022 directly targets the known drift period,
rather than reusing the other scripts' fit-period share (calibrated to match the earlier paper's own split, not to
isolate this drift).

**A genuine limitation, stated up front.** This produces ONE real trajectory, not a replicate distribution: there
is only one real calendar history, so there is no bootstrap CI here, unlike every other real-data result in this
project. The false-alarm calibration (null-stream block bootstrap) and the onset-detection threshold are still
calibrated the usual way; only the SIGNAL side (the actual drift being tested) is a single, found, real event, not
an injected, repeatable one. And even the composite ordering is coarse: a handful of windows straddling the
2019-2020 boundary mix cases from both real periods (global 500-case chunking on a correctly bucket-ordered
sequence doesn't land exactly on every bucket edge) -- reported as-is, not hidden.

Usage: python scripts/m7_tier2_natural_drift.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.m4_pilot import FLAT_ARMS, SPLIT_ARMS  # noqa: E402
from scripts.tier2_extract_features import NON_FEATURE_COLUMNS  # noqa: E402
from src.monitor_module import (INDICATOR_ORDER, component_groups, episode_starts, indicator_indices,  # noqa: E402
                                stream_scores_multi, system_events, system_events_hierarchical)
from src.sim_module.config import false_alarm_budget, load_config  # noqa: E402
from src.tier2_module import (RealDeployedModel, Tier2StreamConfig, assign_windows, build_stream_data,  # noqa: E402
                              tier2_calibrate_arms, tier2_calibrate_onset, tier2_null_scores_multi,
                              tier2_onset_index_calibrated, tier2_onset_null_series_multi)
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "phase2" / "results" / "tier2" / "features"
ONSET_TARGET_PER_1000 = 5.0
# Chronological order of MIMIC-IV's de-identification buckets for this cohort (confirmed via aggregate query,
# `aggregates_mimiciv.json`'s `cohort_by_period`); the drift under test is specifically the last one.
BUCKET_ORDER = ("2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022")
MONITORED_BUCKET = "2020 - 2022"


def add_composite_order(df: pd.DataFrame) -> pd.DataFrame:
    """A single sortable column, `combined_order`: primarily `anchor_year_group`'s chronological rank (so no window
    ever mixes cases from two different real calendar periods), `order_key` only as a same-bucket tiebreak (not
    claimed to be meaningful on its own -- see module docstring)."""
    unknown = set(df["anchor_year_group"]) - set(BUCKET_ORDER)
    if unknown:
        raise ValueError(f"anchor_year_group has values not in BUCKET_ORDER: {sorted(unknown)}")
    out = df.copy()
    bucket_rank = out["anchor_year_group"].map({b: i for i, b in enumerate(BUCKET_ORDER)})
    within_bucket_rank = out.groupby("anchor_year_group")["order_key"].rank(method="first")
    within_bucket_count = out.groupby("anchor_year_group")["order_key"].transform("count")
    out["combined_order"] = bucket_rank + (within_bucket_rank - 1) / within_bucket_count
    return out


def main() -> int:
    path = FEATURES_DIR / "mortality.parquet"
    if not path.exists():
        logger.error("missing %s -- run scripts/tier2_extract_features.py --task mortality first", path)
        return 1

    set_seed(42)
    rng = np.random.default_rng(42)
    phase2_cfg = load_config()
    harm = phase2_cfg.harm

    raw = pd.read_parquet(path)
    if "anchor_year_group" not in raw.columns:
        logger.error("%s has no anchor_year_group column -- re-run scripts/tier2_extract_features.py (this "
                     "session's NON_FEATURE_COLUMNS update added it; see the module docstring)", path)
        return 1
    feature_columns = tuple(c for c in raw.columns if c not in NON_FEATURE_COLUMNS)
    ordered = add_composite_order(raw).sort_values("combined_order", kind="stable").reset_index(drop=True)
    tier2_cfg = Tier2StreamConfig(feature_columns=feature_columns, window_size=500, target_sensitivity=0.8,
                                  order_column="combined_order")

    logger.info("Cases per calendar bucket (chronological order):")
    for bucket in BUCKET_ORDER:
        counts = ordered[ordered["anchor_year_group"] == bucket]
        logger.info("  %-14s %6d cases, mortality %.3f", bucket, len(counts), counts["label"].mean())

    n_windows_total = len(ordered) // 500
    # `n_baseline`: the number of complete 500-case windows entirely before the monitored bucket begins (since
    # `ordered` is already sorted by `combined_order`, the monitored bucket's first case's position tells us exactly
    # where training ends). The window CONTAINING that boundary (if the bucket doesn't start on a window edge)
    # becomes the first monitored window, per the module docstring's stated coarseness.
    first_monitored_position = int(ordered.index[ordered["anchor_year_group"] == MONITORED_BUCKET][0])
    n_baseline = max(1, min(first_monitored_position // 500, n_windows_total - 1))
    train = ordered.iloc[: n_baseline * 500]
    model = RealDeployedModel(tier2_cfg).fit(train, rng)
    scored = model.apply(ordered)
    windowed = assign_windows(scored, tier2_cfg)
    n_windows = windowed["window"].nunique()
    logger.info("%d windows (%d cases each): %d training (pre-%s), %d monitored (%s onward)", n_windows, 500,
               n_baseline, MONITORED_BUCKET, n_windows - n_baseline, MONITORED_BUCKET)

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
    arm_results = tier2_calibrate_arms(calib, evaluation, budget, j)

    onset_calib = tier2_onset_null_series_multi(windowed, tier2_cfg, phase2_cfg, harm, streams=300, seed=501,
                                                n_windows_out=n_null_windows_out, block_size=block_size, n_baseline=null_baseline)
    onset_threshold = tier2_calibrate_onset(onset_calib, ONSET_TARGET_PER_1000, j)
    logger.info("Calibrated onset threshold (target %.0f/1,000 windows): %.2f", ONSET_TARGET_PER_1000, onset_threshold)

    # The real, unresampled, unperturbed stream -- the actual (bucket-valid) calendar history, not a synthetic replicate.
    real_stream = build_stream_data(windowed, tier2_cfg, harm, n_baseline=n_baseline, rng=rng)
    real_onset = tier2_onset_index_calibrated(real_stream, phase2_cfg, onset_threshold, j)
    if real_onset is None:
        logger.warning("No real onset detected on the natural mortality stream at this calibration -- the real "
                       "2020-2022 rise did not cross the calibrated D18 threshold. Reporting as a null result, not "
                       "an error: a real, modest drift is not guaranteed to be large enough to register.")
        return 0
    logger.info("Real onset detected at monitored window %d (of %d)", real_onset, n_windows - n_baseline)

    z = stream_scores_multi(real_stream, phase2_cfg, (lag,))[lag]
    groups = component_groups(INDICATOR_ORDER)
    logger.info("Single real trajectory (n=1, no replicate CI -- see module docstring):")
    for name, (threshold, _rate, _se) in arm_results.items():
        if name in SPLIT_ARMS:
            events = system_events_hierarchical(z[None], groups, threshold)  # type: ignore[arg-type]
        else:
            events = system_events(z[None], indicator_indices(FLAT_ARMS[name]), threshold)  # type: ignore[arg-type]
        starts = episode_starts(events, j)[0]
        first_alert = int(np.argmax(starts)) if starts.any() else None
        tafd = (first_alert - real_onset) if first_alert is not None else None
        logger.info("Arm %-28s first alert: %-6s TAFD: %s", name, first_alert, tafd if tafd is not None else "no alert")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
