"""Computational margin analysis: harm-equivalent margins, pass curves, and the resolution floor."""

import numpy as np
import pytest

from src.eval_module.margin import implied_margin, min_detectable, parse_pilot, pass_share, se_from_interval

SAMPLE = """### Label lag 2 windows

| Scenario | Level | Shape | n | A | B | C | D | E | Reduction, split vs conventional [95% CI] | Excess exposure avoided, split vs conventional [95% CI] | Meets | x | y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SC07 | medium | abrupt | 150 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | +7.8 [+7.0, +8.7] | +79 [+70, +87] | yes | +1.0 | +1.0 [+0.0, +2.0] |
| SC08 | high | gradual | 150 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | -0.9 [-1.0, -0.7] | -8 [-10, -7] | no | -1.0 | -1.0 [-2.0, +0.0] |

### Label lag 8 windows

| Scenario | Level | Shape | n | A | B | C | D | E | Reduction, split vs conventional [95% CI] | Excess exposure avoided, split vs conventional [95% CI] | Meets | x | y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SC17 | high | abrupt | 150 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | +7.3 [+7.1, +7.6] | +166 [+160, +172] | yes | +1.0 | +1.0 [+0.0, +2.0] |
"""


def test_implied_margin_converts_delta_baseline_windows_to_windows():
    assert implied_margin(excess_per_window=7.5, baseline_rhe=30.0, delta=1.0) == 4.0
    assert implied_margin(excess_per_window=15.0, baseline_rhe=30.0, delta=0.5) == 1.0
    assert implied_margin(excess_per_window=0.0, baseline_rhe=30.0, delta=1.0) == float("inf")
    assert implied_margin(excess_per_window=-3.0, baseline_rhe=30.0, delta=1.0) == float("inf")


def test_pass_share_decreases_with_the_margin_and_ignores_nan():
    lowers = np.array([0.2, 1.5, 3.0, np.nan, -1.0])
    shares = [pass_share(lowers, m) for m in (0.0, 1.0, 2.0, 4.0)]
    assert shares == [0.75, 0.5, 0.25, 0.0]
    assert np.isnan(pass_share(np.array([np.nan]), 1.0))
    assert pass_share(np.array([1.5, 1.0]), 1.5) == 0.5      # a lower bound equal to the margin passes


def test_min_detectable_uses_one_sided_alpha_and_power():
    assert min_detectable(1.0) == pytest.approx(1.6449 + 0.8416, abs=1e-3)
    assert min_detectable(2.0, alpha=0.025, power=0.9) == pytest.approx(2 * (1.9600 + 1.2816), abs=1e-3)
    assert se_from_interval(-1.0, 3.0) == pytest.approx(4.0 / 3.92, abs=1e-3)


def test_parse_pilot_reads_lag_scenario_and_both_intervals():
    rows = parse_pilot(SAMPLE)
    assert [(r.lag, r.scenario, r.level, r.shape) for r in rows] == [(2, "SC07", "medium", "abrupt"), (2, "SC08", "high", "gradual"), (8, "SC17", "high", "abrupt")]
    a, b, _ = rows
    assert (a.n, a.reduction, a.reduction_lo, a.reduction_hi) == (150, 7.8, 7.0, 8.7)
    assert (a.exposure, a.exposure_lo, a.exposure_hi) == (79.0, 70.0, 87.0)
    assert (b.reduction, b.reduction_lo, b.exposure_hi) == (-0.9, -1.0, -7.0)
