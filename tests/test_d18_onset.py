"""D18: onset is when pooled harm or the harm of a predefined subgroup reaches (1 + kappa) times its own baseline."""

from dataclasses import replace

import numpy as np
import pytest

from src.sim_module import HARM_POPULATION, PerturbationConfig, PerturbationFactory, expected_rhe, onset_ratio, rhe_baselines

N = HARM_POPULATION      # the elicited weights make expected harm heavy-tailed, so the check needs a large sample


@pytest.fixture(scope="module")
def baselines(world, model, cfg):
    return rhe_baselines(model.apply(world.sample(N, np.random.default_rng(3))), cfg)


def subgroup_drift(world, model, level=0.305):   # SC09 high: the catalog level that targets a subgroup onset ratio of 1.71
    return PerturbationFactory("subgroup_outcome_model_change")(PerturbationConfig(level_value=level)).generate(
        world, model, N, np.random.default_rng(3), 1.0)


def test_baselines_cover_pooled_and_subgroup(baselines):
    assert set(baselines) == {"pooled", "subgroup"} and baselines["pooled"] > 0 and baselines["subgroup"] > 0


def test_subgroup_harm_is_seen_when_pooled_harm_is_not(world, model, cfg, baselines):
    batch = subgroup_drift(world, model)
    pooled = expected_rhe(batch, cfg) / baselines["pooled"]
    assert pooled < 1.15                                            # the pooled cohort barely notices
    assert onset_ratio(batch, cfg, baselines) > 1.5                 # the subgroup does
    assert onset_ratio(batch, cfg, baselines) >= pooled


def test_a_subgroup_below_the_minimum_size_is_ignored(world, model, cfg, baselines):
    """Keep only 10 percent of the truly drifted subgroup, so it is about 1 percent of cases but still clearly harmed."""
    batch = subgroup_drift(world, model)
    keep = np.random.default_rng(0).random(batch.n) < 0.1
    tiny = replace(batch, g=((batch.g == 1) & keep).astype(int))
    pooled = expected_rhe(tiny, cfg) / baselines["pooled"]
    assert tiny.g.mean() < cfg.harm.min_subgroup_share and pooled < 1.15
    assert onset_ratio(tiny, cfg, baselines) == pytest.approx(pooled)                     # ignored: too small to count
    relaxed = replace(cfg, harm=replace(cfg.harm, min_subgroup_share=0.005))
    assert onset_ratio(tiny, relaxed, baselines) > 1 + cfg.harm.kappa                       # counted once the rule allows it, and above the onset threshold


def test_the_subgroup_ratio_uses_the_subgroups_own_baseline(world, model, cfg, baselines):
    """SC09 at its top level targets a ratio of 1.71. The pooled baseline would give about 1.89."""
    assert abs(onset_ratio(subgroup_drift(world, model), cfg, baselines) - 1.71) < 0.06


def test_onset_ratio_never_falls_below_the_pooled_ratio(world, model, cfg, baselines):
    pert = PerturbationFactory("class_prior_shift")(PerturbationConfig(level_value=2.0))
    batch = pert.generate(world, model, N, np.random.default_rng(3), 1.0)
    assert onset_ratio(batch, cfg, baselines) >= expected_rhe(batch, cfg) / baselines["pooled"] - 1e-12


def test_select_keeps_all_fields_aligned(world, model):
    batch = model.apply(world.sample(1000, np.random.default_rng(1)))
    sub = batch.select(batch.g == 1)
    assert sub.n == int((batch.g == 1).sum()) and sub.x_model.shape[0] == sub.n and sub.decision.shape[0] == sub.n
