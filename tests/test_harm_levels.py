"""Harm-scaled levels: solve each perturbation's parameter for a target expected-RHE ratio."""

import pytest

from src.sim_module.harm_levels import AXES, ratio_at, solve_level

N = 100_000


def test_axes_cover_the_implemented_perturbation_types():
    assert {"class_prior_shift", "outcome_model_change", "covariate_mean_shift", "mcar_missingness", "variable_rescale",
            "shrink_scores_to_base_rate_recalibrated", "slope_and_intercept_drift"} <= set(AXES)


def test_solved_level_reproduces_the_target_ratio(world, model, cfg):
    sol = solve_level("class_prior_shift", 1.5, world, model, cfg, n=N)
    assert sol.attainable and abs(sol.ratio - 1.5) < 0.03
    check = ratio_at("class_prior_shift", sol.value, world, model, cfg, n=N)
    assert abs(check - 1.5) < 0.03


def test_higher_targets_need_larger_effects(world, model, cfg):
    values = [solve_level("outcome_model_change", t, world, model, cfg, n=N).value for t in (1.15, 1.35, 1.75)]
    assert values[0] < values[1] < values[2]


def test_direction_is_handled_for_parameters_that_decrease(world, model, cfg):
    """Slope drift gets stronger as the slope falls, so a larger target gives a smaller slope."""
    a = solve_level("slope_and_intercept_drift", 1.15, world, model, cfg, n=N, mode="outcome_link")
    b = solve_level("slope_and_intercept_drift", 1.35, world, model, cfg, n=N, mode="outcome_link")
    assert a.attainable and b.attainable and b.value < a.value < 1.0


def test_unattainable_targets_are_reported_with_the_best_ratio_reached(world, model, cfg):
    sol = solve_level("mcar_missingness", 1.75, world, model, cfg, n=N)
    assert not sol.attainable and sol.value is None and sol.max_ratio < 1.75
    reporting_only = solve_level("slope_and_intercept_drift", 1.15, world, model, cfg, n=N, mode="output_transform")
    assert not reporting_only.attainable


def test_solution_is_deterministic(world, model, cfg):
    a = solve_level("variable_rescale", 1.35, world, model, cfg, n=N)
    b = solve_level("variable_rescale", 1.35, world, model, cfg, n=N)
    assert a.value == b.value


def test_unknown_type_raises(world, model, cfg):
    with pytest.raises(KeyError):
        solve_level("nope", 1.2, world, model, cfg, n=N)
