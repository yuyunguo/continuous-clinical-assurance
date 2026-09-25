"""D13: the component-level false-alarm budget."""

from dataclasses import replace

import numpy as np

from src.monitor_module import (INDICATOR_ORDER, calibrate_hierarchical, calibrate_threshold, component_groups, episode_starts,
                                null_scores, system_events, system_events_hierarchical)
from src.sim_module.observables import load_indicators


def small_cfg(cfg):
    return replace(cfg, stream=replace(cfg.stream, cases_per_window=250, baseline_windows=16, monitored_windows=40))


def test_component_groups_follow_the_instrument_records():
    states = {r["id"]: r["state_component"] for r in load_indicators()}
    groups = component_groups(INDICATOR_ORDER)
    for component, columns in groups.items():
        assert all(states[INDICATOR_ORDER[c]] == component for c in columns)
    assert sum(len(c) for c in groups.values()) == len(INDICATOR_ORDER) and set(groups) == {"C", "P", "U"}


def test_hierarchical_rule_meets_the_system_budget_on_independent_null_streams(world, model, cfg):
    c = small_cfg(cfg)
    groups = component_groups(INDICATOR_ORDER)
    target, j = 20.0, c.routing.hysteresis_j
    thresholds = calibrate_hierarchical(null_scores(world, model, c, streams=200, seed=1), groups, target, j)
    evaluation = null_scores(world, model, c, streams=200, seed=2)
    starts = episode_starts(system_events_hierarchical(evaluation, groups, thresholds), j)
    windows = starts.size
    rate = starts.sum() / windows * 1000
    se = np.sqrt(max(starts.sum(), 1)) / windows * 1000
    assert abs(rate - target) <= 3 * se + 0.15 * target, (rate, target, thresholds)


def test_an_equal_split_raises_each_component_above_its_standalone_full_budget_threshold(world, model, cfg):
    """A component that gets a third of the budget needs a higher threshold than it would alone at the full budget."""
    c = small_cfg(cfg)
    z = null_scores(world, model, c, streams=200, seed=3)
    groups = component_groups(INDICATOR_ORDER)
    j = c.routing.hysteresis_j
    hierarchical = calibrate_hierarchical(z, groups, 20.0, j)
    for component, columns in groups.items():
        assert hierarchical[component] > calibrate_threshold(z, columns, 20.0, j), component


def test_hierarchical_events_are_the_union_over_components(world, model, cfg):
    c = small_cfg(cfg)
    z = null_scores(world, model, c, streams=20, seed=4)
    groups = component_groups(INDICATOR_ORDER)
    thresholds = {"C": 3.0, "P": 4.0, "U": 5.0}
    union = np.zeros(z.shape[:2], dtype=bool)
    for comp, cols in groups.items():
        union |= system_events(z, cols, thresholds[comp])
    assert np.array_equal(system_events_hierarchical(z, groups, thresholds), union)


def test_more_shares_for_a_component_lower_its_threshold(world, model, cfg):
    c = small_cfg(cfg)
    z = null_scores(world, model, c, streams=200, seed=5)
    groups = component_groups(INDICATOR_ORDER)
    j = c.routing.hysteresis_j
    even = calibrate_hierarchical(z, groups, 20.0, j)
    tilted = calibrate_hierarchical(z, groups, 20.0, j, shares={"C": 0.1, "P": 0.8, "U": 0.1})
    assert tilted["P"] < even["P"] and tilted["C"] > even["C"]
    assert calibrate_threshold(z, groups["P"], 20.0, j) > 0
