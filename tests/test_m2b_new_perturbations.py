"""SC09 subgroup drift, SC17 version regression, and the compound scenario."""

import numpy as np
import pytest

from src.eval_module.metrics import auroc, sensitivity
from src.sim_module import AXES, PerturbationConfig, PerturbationFactory

N = 400_000


def gen(name, level, world, model, progress=1.0, seed=21, mode=None):
    return PerturbationFactory(name)(PerturbationConfig(level_value=level, mode=mode)).generate(
        world, model, N, np.random.default_rng(seed), progress)


@pytest.fixture(scope="module")
def base(world, model):
    return model.apply(world.sample(N, np.random.default_rng(21)))


def test_new_types_have_harm_axes():
    assert {"subgroup_outcome_model_change", "version_swap_with_regression", "covariate_and_concept_drift"} <= set(AXES)


def test_subgroup_drift_changes_only_the_subgroup(world, model, base):
    out = gen("subgroup_outcome_model_change", 0.30, world, model)
    g1, g0 = out.g == 1, out.g == 0
    base_se1, base_se0 = sensitivity(base.y[base.g == 1], base.decision[base.g == 1]), sensitivity(base.y[base.g == 0], base.decision[base.g == 0])
    se1, se0 = sensitivity(out.y[g1], out.decision[g1]), sensitivity(out.y[g0], out.decision[g0])
    assert abs((1 - se1 / base_se1) - 0.30) < 0.04          # the requested relative loss, inside the subgroup
    assert abs(se0 - base_se0) < 0.01                        # everyone else is untouched
    assert abs(out.y.mean() - base.y.mean()) < 0.005 and abs(out.y[g1].mean() - base.y[base.g == 1].mean()) < 0.01
    assert abs(out.x.mean(axis=0) - base.x.mean(axis=0)).max() < 0.01   # P(x) unchanged


def test_subgroup_drift_hits_the_requested_loss_and_prevalence_tightly_near_the_cap(world, model, base):
    """Near the cap the pooled and subgroup relationships differ most, so a solver that ignores the subgroup shows up here."""
    out = gen("subgroup_outcome_model_change", 0.45, world, model)
    g1 = out.g == 1
    base_se1 = sensitivity(base.y[base.g == 1], base.decision[base.g == 1])
    assert abs((1 - sensitivity(out.y[g1], out.decision[g1]) / base_se1) - 0.45) < 0.02
    assert abs(out.y[g1].mean() - base.y[base.g == 1].mean()) < 0.006


def test_subgroup_drift_rejects_an_unreachable_loss(world, model):
    with pytest.raises(ValueError, match="maximum"):
        PerturbationFactory("subgroup_outcome_model_change")(PerturbationConfig(level_value=0.6)).generate(
            world, model, 1000, np.random.default_rng(0), 1.0)


def test_version_regression_leaves_the_world_unchanged_and_lowers_auroc(world, model, base):
    aurocs = []
    for f in (0.3, 0.6, 0.9):
        out = gen("version_swap_with_regression", f, world, model)
        assert np.array_equal(out.y, base.y) and np.array_equal(out.x, base.x)   # same cases, only the model changed
        aurocs.append(auroc(out.y, out.reported_p))
    assert auroc(base.y, base.reported_p) > aurocs[0] > aurocs[1] > aurocs[2]


def test_version_regression_at_zero_is_the_identity_and_does_not_mutate_the_model(world, model, base):
    coef, threshold = model._clf.coef_.copy(), model.threshold
    out = gen("version_swap_with_regression", 0.0, world, model)
    assert np.allclose(out.p_hat, base.p_hat)
    gen("version_swap_with_regression", 0.8, world, model)
    assert np.array_equal(model._clf.coef_, coef) and model.threshold == threshold


def test_compound_shifts_inputs_and_weakens_the_outcome_link(world, model, base):
    out = gen("covariate_and_concept_drift", 0.25, world, model)
    shift = (out.x.mean(axis=0) - base.x.mean(axis=0)) @ world.direction
    assert abs(shift - 0.5) < 0.05                                        # the fixed 0.5 SD covariate shift
    assert auroc(out.y, out.reported_p) < auroc(base.y, base.reported_p) - 0.08           # the concept drift weakens discrimination
    alone = gen("outcome_model_change", 0.25, world, model)
    assert sensitivity(out.y, out.decision) > sensitivity(alone.y, alone.decision) + 0.05   # the upward shift in scores offsets part of the loss
    none = gen("covariate_and_concept_drift", 0.25, world, model, progress=0.0)
    assert abs((none.x.mean(axis=0) - base.x.mean(axis=0)) @ world.direction) < 0.02
