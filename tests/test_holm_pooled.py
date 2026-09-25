"""Holm adjustment across the four hypotheses and the scenario-pooled bootstrap (design section 9.1)."""

import numpy as np
import pytest

from src.eval_module.holm import holm
from src.eval_module.pooled import pooled_bootstrap


def test_holm_step_down_adjusted_p_values():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}, alpha=0.05)
    assert out["d"][0] == pytest.approx(0.02) and out["a"][0] == pytest.approx(0.03)
    assert out["c"][0] == pytest.approx(0.06) and out["b"][0] == pytest.approx(0.06)     # 0.04 x 1 is raised to the running maximum
    assert {k for k, (_, reject) in out.items() if reject} == {"a", "d"}


def test_holm_caps_at_one_keeps_order_and_handles_edges():
    out = holm({"x": 0.6, "y": 0.9}, alpha=0.05)
    assert out["x"][0] == pytest.approx(1.0) and out["y"][0] == pytest.approx(1.0)      # 2 x 0.6 is capped at 1, and 0.9 is raised to the running maximum
    assert holm({}, alpha=0.05) == {}
    assert holm({"only": 0.04}, alpha=0.05)["only"] == (pytest.approx(0.04), True)
    tie = holm({"p": 0.02, "q": 0.02}, alpha=0.05)
    assert tie["p"][0] == tie["q"][0] == pytest.approx(0.04)
    nan = holm({"z": float("nan"), "w": 0.001}, alpha=0.05)
    assert nan["z"][1] is False and nan["w"][0] == pytest.approx(0.001)      # a NaN p-value does not count toward the family size


def test_pooled_bootstrap_weights_scenarios_equally_and_averages_cells_within_them():
    a1, a2 = np.full(50, 1.0), np.full(50, 3.0)          # scenario A: cells average to 2
    b = np.full(20, 4.0)                                   # scenario B: one cell
    est, lo, hi, p = pooled_bootstrap({"A": [a1, a2], "B": [b]}, margin=2.5, rng=np.random.default_rng(0), n_boot=200)
    assert est == pytest.approx(3.0) and lo == hi == pytest.approx(3.0) and p == 0.0
    est2, _, _, p2 = pooled_bootstrap({"A": [a1, a2], "B": [b]}, margin=3.5, rng=np.random.default_rng(0), n_boot=200)
    assert est2 == pytest.approx(3.0) and p2 == 1.0


def test_pooled_bootstrap_interval_covers_the_estimate_and_p_reflects_noise():
    rng = np.random.default_rng(1)
    noisy = {"A": [rng.normal(1.0, 3.0, 200)], "B": [rng.normal(1.0, 3.0, 200)]}
    est, lo, hi, p = pooled_bootstrap(noisy, margin=0.0, rng=np.random.default_rng(2), n_boot=1000)
    assert lo < est < hi and hi - lo > 0.3 and 0.0 <= p < 0.2
    _, _, _, p_far = pooled_bootstrap(noisy, margin=est, rng=np.random.default_rng(2), n_boot=1000)
    assert 0.3 < p_far < 0.7                                # a margin at the estimate is a coin flip
    assert np.isnan(pooled_bootstrap({}, margin=0.0, rng=np.random.default_rng(0), n_boot=10)[0])
    assert 0.53 < hi - lo < 0.65                            # a 95 percent interval: pooled standard error about 0.15
    assert pooled_bootstrap({"A": [np.full(30, 3.0)]}, margin=3.0, rng=np.random.default_rng(0), n_boot=50)[3] == 1.0   # a value equal to the margin does not beat it


def test_collect_diffs_groups_cells_by_scenario_and_pairs_replicates():
    from src.eval_module.pooled import collect_diffs
    store = {"SC07|medium|abrupt|2|conventional": np.array([5.0, 6.0]), "SC07|medium|abrupt|2|CCA": np.array([3.0, 2.0]),
             "SC07|high|abrupt|2|conventional": np.array([4.0]), "SC07|high|abrupt|2|CCA": np.array([4.0]),
             "SC08|high|abrupt|2|conventional": np.array([1.0, 1.0]), "SC08|high|abrupt|2|CCA": np.array([2.0, 3.0]),
             "SC07|medium|abrupt|8|conventional": np.array([9.0]), "SC07|medium|abrupt|8|CCA": np.array([1.0])}
    got = collect_diffs(store, ("SC07", "SC08", "SC09"), lag=2, reference="conventional", candidate="CCA")
    assert list(got) == ["SC07", "SC08", "SC09"] and got["SC09"] == []
    assert [d.tolist() for d in got["SC07"]] == [[0.0], [2.0, 4.0]]      # cells in sorted key order and [d.tolist() for d in got["SC08"]] == [[-1.0, -2.0]]
    assert collect_diffs(store, ("SC07",), lag=8, reference="conventional", candidate="CCA")["SC07"][0].tolist() == [8.0]
    only = collect_diffs(store, ("SC07",), lag=2, reference="conventional", candidate=None)   # candidate None: the reference values themselves
    assert [d.tolist() for d in only["SC07"]] == [[4.0], [5.0, 6.0]]
