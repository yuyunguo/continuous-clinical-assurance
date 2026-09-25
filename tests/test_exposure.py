"""Excess harm exposure accrued before detection (advice on decision D19: report windows and harm)."""

import numpy as np

from src.eval_module.tafd import exposure_before_detection
from src.monitor_module.streams import pooled_excess
from src.sim_module import HARM_POPULATION, PerturbationConfig, PerturbationFactory, baselines_for


def test_exposure_sums_undetected_windows():
    excess = np.array([0.0, 0.0, 2.0, 3.0, 5.0, 5.0, 5.0])
    assert exposure_before_detection(excess, t0=2, tafd=0) == 0.0
    assert exposure_before_detection(excess, t0=2, tafd=1) == 2.0
    assert exposure_before_detection(excess, t0=2, tafd=3) == 10.0
    assert exposure_before_detection(excess, t0=2, tafd=99) == 20.0   # capped at the end of the stream


def test_pooled_excess_zero_without_drift_and_positive_with(world, model, cfg):
    bases = baselines_for(world, model, cfg, HARM_POPULATION)
    none = PerturbationFactory("none")(PerturbationConfig(level_value=0.0))
    assert abs(pooled_excess(none, world, model, cfg, 1.0, bases)) < 0.01 * bases["pooled"]
    shift = PerturbationFactory("outcome_model_change")(PerturbationConfig(level_value=0.195))
    assert pooled_excess(shift, world, model, cfg, 1.0, bases) > pooled_excess(shift, world, model, cfg, 0.5, bases) > 0.0


def test_pooled_excess_uses_the_pooled_baseline_and_never_goes_negative(world, model, cfg):
    shift = PerturbationFactory("outcome_model_change")(PerturbationConfig(level_value=0.195))
    assert pooled_excess(shift, world, model, cfg, 1.0, {"pooled": 1e9, "subgroup": 0.0}) == 0.0
    assert pooled_excess(shift, world, model, cfg, 1.0, {"pooled": 0.0, "subgroup": 1e9}) > 10.0


def test_bootstrap_mean_interval_covers_the_mean_and_is_degenerate_for_constants():
    from src.eval_module.tafd import bootstrap_mean
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, 400)
    lo, hi = bootstrap_mean(x, np.random.default_rng(1))
    assert lo < x.mean() < hi and hi - lo < 1.0
    assert bootstrap_mean(np.full(50, 3.0), np.random.default_rng(1)) == (3.0, 3.0)


def test_excess_ratio_matches_the_pooled_excess_for_pooled_mechanisms_and_uses_the_subgroup_for_sc09(world, model, cfg):
    from src.monitor_module.streams import excess_ratio
    from src.sim_module import baselines_for
    bases = baselines_for(world, model, cfg, HARM_POPULATION)
    none = PerturbationFactory("none")(PerturbationConfig(level_value=0.0))
    assert excess_ratio(none, world, model, cfg, 1.0, bases) < 0.02
    pooled = PerturbationFactory("outcome_model_change")(PerturbationConfig(level_value=0.288))
    assert abs(excess_ratio(pooled, world, model, cfg, 1.0, bases) - pooled_excess(pooled, world, model, cfg, 1.0, bases) / bases["pooled"]) < 0.03
    sub = PerturbationFactory("subgroup_outcome_model_change")(PerturbationConfig(level_value=0.305))
    ratio, pooled_only = excess_ratio(sub, world, model, cfg, 1.0, bases), pooled_excess(sub, world, model, cfg, 1.0, bases) / bases["pooled"]
    assert abs(ratio - 0.71) < 0.06 and pooled_only < 0.15          # the subgroup's own baseline gives the catalog's 1.71, the pooled cohort barely notices
    assert excess_ratio(sub, world, model, cfg, 0.5, bases) < ratio  # rises with progress
    inflated = {"pooled": 1e9, "subgroup": 1e9}
    assert excess_ratio(sub, world, model, cfg, 1.0, inflated) == 0.0      # a ratio below 1 is not negative exposure
