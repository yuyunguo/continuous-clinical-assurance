"""Tier 2 real-data stream builder: synthetic stand-in, real deployed model, windowing, and StreamData assembly.

Row-level MIMIC-IV and eICU data stays on the study author's machine (see `Tier2-Readiness.md`); this package is
developed and tested only against the synthetic stand-in in `synthetic_stand_in.py`. `scripts/tier2_extract_features.py`
is written for the study author to run locally; nothing in this package executes a database query.
"""

from .arms import tier2_calibrate_arms
from .calibrate import tier2_null_scores_multi
from .config import Tier2StreamConfig
from .harm import batch_from_windows, tier2_onset_index, tier2_onset_index_rolling, tier2_rhe_baselines
from .level_solving import LevelSolution, tier2_ratio_for, tier2_solve_level, tier2_solve_rescale_level
from .onset_calibration import tier2_calibrate_onset, tier2_onset_index_calibrated, tier2_onset_null_series_multi, tier2_onset_series
from .perturb import (group_missingness_and_rescore, missingness_and_rescore, rescale_and_rescore,
                      scope_creep_and_rescore, version_regression_and_rescore, window_progress)
from .real_model import RealDeployedModel
from .replicates import tier2_replicate_tafd
from .resample import block_bootstrap_stream, block_bootstrap_windows
from .stream import build_stream_data
from .synthetic_stand_in import make_synthetic_cohort
from .windowing import assign_windows

__all__ = ["LevelSolution", "RealDeployedModel", "Tier2StreamConfig", "assign_windows", "batch_from_windows",
          "block_bootstrap_stream", "block_bootstrap_windows", "build_stream_data", "group_missingness_and_rescore",
          "make_synthetic_cohort", "missingness_and_rescore", "rescale_and_rescore", "tier2_calibrate_arms", "tier2_calibrate_onset",
          "tier2_null_scores_multi", "tier2_onset_index", "tier2_onset_index_calibrated", "tier2_onset_index_rolling",
          "tier2_onset_null_series_multi", "tier2_onset_series", "tier2_replicate_tafd", "tier2_rhe_baselines",
          "tier2_ratio_for", "tier2_solve_level", "tier2_solve_rescale_level", "scope_creep_and_rescore",
          "version_regression_and_rescore", "window_progress"]
