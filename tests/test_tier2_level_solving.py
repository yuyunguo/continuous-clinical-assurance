"""Harm-scaled perturbation levels for Tier 2, mirroring `src.sim_module.harm_levels`'s approach: solve a
perturbation's parameter for a target D18 ratio (max of pooled and subgroup expected-RHE ratios), instead of
reusing a magnitude picked for the synthetic stand-in's much lower-dimensional world unchanged.

Investigation found the reused magnitude (level_value=3.0, close to Tier 1's own SC07 "medium", 3.51x) produced no
detectable harm change on real, 115-feature data at all -- a single rescaled feature's influence is diluted across
many correlated real features after regularization, unlike the synthetic world's few, weighted-by-design features.
"""

import numpy as np
import pytest

from src.sim_module.config import load_config
from src.tier2_module.config import Tier2StreamConfig
from src.tier2_module.level_solving import tier2_solve_rescale_level
from src.tier2_module.real_model import RealDeployedModel
from src.tier2_module.synthetic_stand_in import make_synthetic_cohort

FEATURES = ("f0", "f1", "f2")


@pytest.fixture
def phase2_cfg():
    return load_config()


@pytest.fixture
def cfg():
    return Tier2StreamConfig(feature_columns=FEATURES, target_sensitivity=0.8)


@pytest.fixture
def fitted_population(cfg):
    df = make_synthetic_cohort(n=20_000, feature_columns=FEATURES, rng=np.random.default_rng(0), prevalence=0.25)
    model = RealDeployedModel(cfg).fit(df.iloc[:5000], np.random.default_rng(1))
    return model, model.apply(df)


def test_solve_rescale_level_finds_a_higher_level_for_a_higher_target(cfg, phase2_cfg, fitted_population):
    """target_ratio=1.45, not the higher value this test used before `RealDeployedModel` started winsorizing:
    winsorization clips a rescaled column to the training cohort's fit-period bounds, so beyond some rescale factor
    the model sees an identical (clipped) value regardless of how much larger `level_value` grows -- the ratio
    plateaus (an expected side effect, and a second, independent reason real SC07 rescale is a weak real-data
    perturbation, alongside the feature-dilution finding in this module's own docstring)."""
    model, population = fitted_population
    low = tier2_solve_rescale_level(population, cfg, phase2_cfg, model, column="f0", target_ratio=1.15, cap=20.0)
    high = tier2_solve_rescale_level(population, cfg, phase2_cfg, model, column="f0", target_ratio=1.45, cap=20.0)
    assert low.attainable and high.attainable
    assert high.value > low.value


def test_solve_rescale_level_solution_actually_achieves_close_to_the_target_ratio(cfg, phase2_cfg, fitted_population):
    model, population = fitted_population
    solution = tier2_solve_rescale_level(population, cfg, phase2_cfg, model, column="f0", target_ratio=1.35, cap=20.0)
    assert solution.attainable
    assert abs(solution.ratio - 1.35) < 0.02


def test_solve_rescale_level_reports_unattainable_when_the_cap_is_too_low(cfg, phase2_cfg, fitted_population):
    model, population = fitted_population
    solution = tier2_solve_rescale_level(population, cfg, phase2_cfg, model, column="f0", target_ratio=100.0, cap=1.5)
    assert not solution.attainable
    assert solution.value is None
    assert solution.max_ratio < 100.0


def test_solve_rescale_level_at_neutral_value_one_gives_ratio_close_to_one(cfg, phase2_cfg, fitted_population):
    """A sanity check on the solver's own ratio function: no rescale (level_value=1.0, the neutral/no-op value)
    must reproduce the population's own baseline ratio, close to 1."""
    from src.tier2_module.level_solving import _ratio_at
    model, population = fitted_population
    ratio = _ratio_at(population, cfg, phase2_cfg, model, column="f0", level_value=1.0)
    assert abs(ratio - 1.0) < 0.05


def test_solve_rescale_level_treats_the_cap_exactly_reaching_the_target_as_attainable(cfg, phase2_cfg, fitted_population, monkeypatch):
    """The boundary case: if the grid's maximum ratio lands exactly on the target, that must count as attainable
    (`ratios.max() >= target`), not unattainable (`>`)."""
    import src.tier2_module.level_solving as level_solving
    model, population = fitted_population
    grid = np.linspace(1.0, 5.0, level_solving.GRID_POINTS)
    monkeypatch.setattr(level_solving, "_ratio_at", lambda *a, **k: 1.35 if a[-1] == grid[-1] else 1.0)
    solution = tier2_solve_rescale_level(population, cfg, phase2_cfg, model, column="f0", target_ratio=1.35, cap=5.0)
    assert solution.attainable


def test_solve_rescale_level_takes_the_first_grid_point_that_reaches_the_target_inclusively(cfg, phase2_cfg, fitted_population, monkeypatch):
    """A grid point whose ratio exactly equals the target must count as reaching it (`>=`, not `>`)."""
    import src.tier2_module.level_solving as level_solving
    model, population = fitted_population
    grid = np.linspace(1.0, 5.0, level_solving.GRID_POINTS)
    exact_hit = grid[3]
    monkeypatch.setattr(level_solving, "_ratio_at", lambda pop, c, pc, m, col, v: 1.35 if v == exact_hit else 1.0 + 0.01 * list(grid).index(v))
    solution = tier2_solve_rescale_level(population, cfg, phase2_cfg, model, column="f0", target_ratio=1.35, cap=5.0)
    assert solution.attainable
    assert solution.value == pytest.approx(exact_hit)


def test_ratio_at_takes_the_subgroup_ratio_when_it_exceeds_the_pooled_ratio(cfg, phase2_cfg, fitted_population, monkeypatch):
    """The D18 rule is the LARGER of the pooled and subgroup ratios. `rhe_baselines` (called for the baseline
    reference) is defined in `src.sim_module.harm` and looks up `expected_rhe` in its own module namespace, so it
    is mocked directly here to a fixed reference; `expected_rhe` (called directly by `_ratio_at`, for the
    perturbed side only) is mocked in `level_solving`'s own namespace, in call order: pooled, then subgroup."""
    import src.tier2_module.level_solving as level_solving
    model, population = fitted_population
    monkeypatch.setattr(level_solving, "rhe_baselines", lambda batch, cfg_arg: {"pooled": 100.0, "subgroup": 100.0})
    # perturbed pooled=105 (ratio 1.05, mild); perturbed subgroup=500 (ratio 5.0, severe) -- the D18 rule must
    # report 5.0, the larger of the two ratios, not the mild pooled one
    values = iter([105.0, 500.0])
    monkeypatch.setattr(level_solving, "expected_rhe", lambda batch, cfg_arg: next(values))
    ratio = level_solving._ratio_at(population, cfg, phase2_cfg, model, column="f0", level_value=1.0)
    assert ratio == pytest.approx(5.0)


def test_ratio_at_takes_the_pooled_ratio_when_it_exceeds_the_subgroup_ratio(cfg, phase2_cfg, fitted_population, monkeypatch):
    """The mirror of the previous test: when the POOLED ratio is the larger one, that must be what's returned,
    not the subgroup ratio unconditionally."""
    import src.tier2_module.level_solving as level_solving
    model, population = fitted_population
    monkeypatch.setattr(level_solving, "rhe_baselines", lambda batch, cfg_arg: {"pooled": 100.0, "subgroup": 100.0})
    values = iter([500.0, 105.0])   # perturbed pooled=500 (severe); perturbed subgroup=105 (mild)
    monkeypatch.setattr(level_solving, "expected_rhe", lambda batch, cfg_arg: next(values))
    ratio = level_solving._ratio_at(population, cfg, phase2_cfg, model, column="f0", level_value=1.0)
    assert ratio == pytest.approx(5.0)


def test_ratio_at_includes_the_subgroup_at_exactly_the_minimum_share(cfg, phase2_cfg, monkeypatch):
    """D18's rule counts the subgroup once its share is `>= min_subgroup_share`, not strictly greater. Counts how
    many times `expected_rhe` is called directly by `_ratio_at` (the perturbed side only, mocked `rhe_baselines`
    aside): 2 calls (pooled and subgroup) if the subgroup counts, 1 (pooled only) if it does not."""
    import src.tier2_module.level_solving as level_solving
    df = make_synthetic_cohort(n=2_000, feature_columns=FEATURES, rng=np.random.default_rng(60), prevalence=0.3)
    assert phase2_cfg.harm.min_subgroup_share == pytest.approx(0.05)
    df["subgroup"] = 0
    df.loc[df.index[:100], "subgroup"] = 1   # exactly 100/2000 = 0.05, the config's min_subgroup_share, deterministically
    model = RealDeployedModel(cfg).fit(df.iloc[:1000], np.random.default_rng(61))
    population = model.apply(df)
    monkeypatch.setattr(level_solving, "rhe_baselines", lambda batch, cfg_arg: {"pooled": 1.0, "subgroup": 1.0})
    calls = []
    real_expected_rhe = level_solving.expected_rhe

    def counting_expected_rhe(batch, cfg_arg):
        calls.append(batch.y.shape[0])
        return real_expected_rhe(batch, cfg_arg)

    monkeypatch.setattr(level_solving, "expected_rhe", counting_expected_rhe)
    level_solving._ratio_at(population, cfg, phase2_cfg, model, column="f0", level_value=1.0)
    assert len(calls) == 2   # pooled, then subgroup: the subgroup branch ran


def test_tier2_solve_level_works_for_a_non_rescale_perturbation(cfg, phase2_cfg, fitted_population):
    """`tier2_solve_level`/`tier2_ratio_for` must be generic over any single-parameter perturbation, not just
    rescale -- checked here with a version-regression-style `perturb` function (the real motivation: SC07 rescale
    turned out unattainable on real, 115-feature data, but SC17 version regression was not)."""
    from src.tier2_module.level_solving import tier2_ratio_for, tier2_solve_level
    model, population = fitted_population

    def ratio_at(fraction: float) -> float:
        return tier2_ratio_for(population, cfg, phase2_cfg, lambda pop: model.with_regression(fraction, seed=1).apply(pop))

    solution = tier2_solve_level(ratio_at, target_ratio=1.35, neutral=0.0, cap=1.0)
    assert solution.attainable
    assert abs(solution.ratio - 1.35) < 0.02
    assert 0.0 < solution.value < 1.0


def test_solve_rescale_level_is_reproducible():
    """The solver's ratio function must be deterministic given the same population and level -- no hidden randomness."""
    cfg = Tier2StreamConfig(feature_columns=FEATURES, target_sensitivity=0.8)
    df = make_synthetic_cohort(n=5_000, feature_columns=FEATURES, rng=np.random.default_rng(5), prevalence=0.25)
    phase2_cfg = load_config()
    model = RealDeployedModel(cfg).fit(df.iloc[:2500], np.random.default_rng(6))
    population = model.apply(df)
    from src.tier2_module.level_solving import _ratio_at
    a = _ratio_at(population, cfg, phase2_cfg, model, column="f0", level_value=3.0)
    b = _ratio_at(population, cfg, phase2_cfg, model, column="f0", level_value=3.0)
    assert a == b
