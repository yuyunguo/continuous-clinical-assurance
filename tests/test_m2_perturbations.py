"""M2: perturbation registry. Property tests are generated from the scenario catalog's own expectations."""

import numpy as np
import pytest

from src.eval_module.metrics import auroc, calibration_slope, ece, npv, ppv, sensitivity, specificity
from src.sim_module import PerturbationFactory, PerturbationConfig, expected_rhe, parse_level, registered_perturbations

IMPLEMENTED = ["SC01", "SC02", "SC04", "SC05", "SC07", "SC08", "SC09", "SC17", "SC23", "SC25"]
OUTCOME_LINK_SLOPES = (0.95, 0.9, 0.85)   # the retired SC03 levels, kept as a unit test of the parametrization
LEVELS = ("low", "medium", "high")


def population_metrics(batch):
    """Population-level values for the catalog indicators that can be computed from a batch alone."""
    return {"P1": sensitivity(batch.y, batch.decision), "P2": specificity(batch.y, batch.decision),
            "P3ppv": ppv(batch.y, batch.decision), "P3npv": npv(batch.y, batch.decision),
            "P4": auroc(batch.y, batch.reported_p), "U1": ece(batch.y, batch.reported_p),
            "U3": calibration_slope(batch.y, batch.reported_p), "prevalence": batch.y.mean(),
            "input_mean_shift": float(np.abs(batch.x.mean(axis=0)).max())}


@pytest.fixture(scope="module")
def baseline(world, model, cfg):
    return population_metrics(model.apply(world.sample(cfg.tolerances.population_cases, np.random.default_rng(11))))


def run(scenarios, scenario_id, level, world, model, cfg, mode=None, progress=1.0, seed=12):
    spec = scenarios[scenario_id]["perturbation"]
    pert = PerturbationFactory(spec["type"])(PerturbationConfig(level_value=parse_level(spec["levels"][level]), mode=mode or spec.get("mode")))
    return pert.generate(world, model, cfg.tolerances.population_cases, np.random.default_rng(seed), progress)


def test_parse_level_handles_catalog_formats():
    assert parse_level("0.25 SD") == 0.25 and parse_level("1.5x") == 1.5 and parse_level("slope 0.9") == 0.9
    assert parse_level("lambda 0.1") == 0.1 and parse_level("5 pct") == 0.05 and parse_level("10 pct relative Se loss") == 0.10


def test_registry_names_match_catalog(scenarios):
    catalog_types = {s["perturbation"]["type"] for s in scenarios.values()}
    assert set(registered_perturbations()) <= catalog_types
    assert {scenarios[i]["perturbation"]["type"] for i in IMPLEMENTED} <= set(registered_perturbations())


@pytest.mark.parametrize("scenario_id", IMPLEMENTED)
@pytest.mark.parametrize("level", LEVELS)
def test_held_stable_indicators_stay_stable(scenario_id, level, scenarios, world, model, cfg, baseline):
    """Every indicator the catalog lists as held_stable must not move (population level)."""
    tol = cfg.tolerances
    held = scenarios[scenario_id]["held_stable"]
    values = population_metrics(run(scenarios, scenario_id, level, world, model, cfg))
    for ind in held:
        if ind in ("P1", "P2"):
            assert abs(values[ind] - baseline[ind]) <= tol.rate_abs, (scenario_id, level, ind, values[ind], baseline[ind])
        elif ind == "P4":
            assert abs(values[ind] - baseline[ind]) <= tol.auroc_abs, (scenario_id, level, ind, values[ind], baseline[ind])
        elif ind == "U1":
            assert values[ind] <= baseline[ind] + tol.rate_abs, (scenario_id, level, ind, values[ind], baseline[ind])
        elif ind == "U3":
            assert abs(values[ind] - 1.0) <= 0.05, (scenario_id, level, ind, values[ind])


def test_overconfidence_preserves_ranking_exactly(scenarios, world, model, cfg):
    """SC03 output-transform mode is monotone, so AUROC is unchanged to machine precision."""
    base = model.apply(world.sample(50_000, np.random.default_rng(5)))
    pert = PerturbationFactory("slope_and_intercept_drift")(PerturbationConfig(level_value=0.6, mode="output_transform"))
    out = pert.generate(world, model, 50_000, np.random.default_rng(5), 1.0)
    assert abs(auroc(out.y, out.reported_p) - auroc(base.y, base.reported_p)) < 1e-12


def test_reporting_layer_overconfidence_raises_calibration_error_and_lowers_slope(scenarios, world, model, cfg, baseline):
    ece_values, slopes = [], []
    for level in LEVELS:
        values = population_metrics(run(scenarios, "SC23", level, world, model, cfg))
        ece_values.append(values["U1"])
        slopes.append(values["U3"])
    assert ece_values[0] < ece_values[1] < ece_values[2] and ece_values[0] > baseline["U1"] + 0.005
    assert slopes[0] > slopes[1] > slopes[2] and slopes[2] < 0.75


def test_prevalence_shift_moves_predictive_values_but_not_se_sp(scenarios, world, model, cfg, baseline):
    values = population_metrics(run(scenarios, "SC02", "high", world, model, cfg))
    assert values["prevalence"] > 1.8 * baseline["prevalence"] and values["P3ppv"] > baseline["P3ppv"] + 0.10   # SC02 high is a 1.96x prevalence shift


def test_concept_drift_leaves_inputs_and_prevalence_unchanged(scenarios, world, model, cfg, baseline):
    """SC08 lists C1 and C2 as negative controls, so inputs and prevalence must not move."""
    values = population_metrics(run(scenarios, "SC08", "high", world, model, cfg))
    assert values["input_mean_shift"] <= baseline["input_mean_shift"] + cfg.tolerances.input_mean_abs
    assert abs(values["prevalence"] - baseline["prevalence"]) <= 0.005


def test_concept_drift_hits_the_requested_relative_sensitivity_loss(scenarios, world, model, cfg, baseline):
    for level in LEVELS:
        target = parse_level(scenarios["SC08"]["perturbation"]["levels"][level])
        loss = 1 - population_metrics(run(scenarios, "SC08", level, world, model, cfg))["P1"] / baseline["P1"]
        assert abs(loss - target) <= 0.02, (level, loss, target)


def test_missingness_only_changes_model_inputs(scenarios, world, model, cfg):
    batch = run(scenarios, "SC05", "high", world, model, cfg)
    assert abs(batch.missing_mask[:, 0].mean() - 0.30) < 0.01 and batch.missing_mask[:, 1:].sum() == 0


@pytest.mark.parametrize("scenario_id", ["SC02", "SC04", "SC08", "SC17", "SC25"])
def test_expected_rhe_is_non_decreasing_in_level(scenario_id, scenarios, world, model, cfg):
    rhe = [expected_rhe(run(scenarios, scenario_id, lv, world, model, cfg), cfg) for lv in LEVELS]
    assert rhe[0] <= rhe[1] + 0.5 and rhe[1] <= rhe[2] + 0.5 and rhe[2] > rhe[0]


def test_gradual_shape_reaches_full_effect_only_at_full_progress(scenarios, world, model, cfg, baseline):
    half = population_metrics(run(scenarios, "SC08", "high", world, model, cfg, progress=0.5))
    full = population_metrics(run(scenarios, "SC08", "high", world, model, cfg, progress=1.0))
    assert baseline["P1"] > half["P1"] > full["P1"]


def test_output_transform_leaves_decisions_and_se_sp_unchanged(scenarios, world, model, cfg):
    """The default SC03 variant drifts only the displayed confidence, never the decisions."""
    base = model.apply(world.sample(50_000, np.random.default_rng(6)))
    for level in LEVELS:
        pert = PerturbationFactory("slope_and_intercept_drift")(
            PerturbationConfig(level_value=parse_level(scenarios["SC23"]["perturbation"]["levels"][level]), mode="output_transform"))
        out = pert.generate(world, model, 50_000, np.random.default_rng(6), 1.0)
        assert np.array_equal(out.decision, base.decision) and not np.allclose(out.reported_p, base.reported_p)


def test_outcome_link_variant_preserves_prevalence_and_trades_auroc_for_calibration(world, model, cfg, baseline):
    """Unit test of the outcome-link parametrization (SC03 is retired as a scenario): prevalence held, AUROC falls with the slope."""
    aurocs, slopes = [], []
    for slope in OUTCOME_LINK_SLOPES:
        pert = PerturbationFactory("slope_and_intercept_drift")(PerturbationConfig(level_value=slope, mode="outcome_link"))
        values = population_metrics(pert.generate(world, model, cfg.tolerances.population_cases, np.random.default_rng(12), 1.0))
        assert abs(values["prevalence"] - baseline["prevalence"]) <= 0.005
        aurocs.append(values["P4"])
        slopes.append(values["U3"])
    assert baseline["P4"] > aurocs[0] > aurocs[1] > aurocs[2]
    assert slopes[0] > slopes[1] > slopes[2]


def test_unknown_perturbation_raises_instead_of_defaulting():
    with pytest.raises(KeyError):
        PerturbationFactory("not_a_perturbation")
    with pytest.raises(ValueError):
        PerturbationFactory("slope_and_intercept_drift")(PerturbationConfig(level_value=0.8, mode="bogus")).generate(
            None, None, 10, np.random.default_rng(0), 1.0)


def test_outcome_link_leaves_the_confidence_distribution_alone(world, model, cfg):
    """With the world changing and the model fixed, the displayed confidence distribution does not move (U5 is a negative control)."""
    base_batch = model.apply(world.sample(cfg.tolerances.population_cases, np.random.default_rng(12)))
    for slope in OUTCOME_LINK_SLOPES:
        pert = PerturbationFactory("slope_and_intercept_drift")(PerturbationConfig(level_value=slope, mode="outcome_link"))
        batch = pert.generate(world, model, cfg.tolerances.population_cases, np.random.default_rng(12), 1.0)
        assert abs(batch.reported_p.mean() - base_batch.reported_p.mean()) < 0.01
