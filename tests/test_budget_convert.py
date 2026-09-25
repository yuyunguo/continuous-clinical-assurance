"""Converting a governance capacity (false-alarm reviews a year) into the study's budget (alert episodes per 1,000 windows)."""

import pytest

from src.eval_module.budget import false_alarms_per_1000_windows, window_days


def test_window_length_follows_from_the_case_volume():
    assert window_days(cases_per_window=500, patients_per_week=500) == pytest.approx(7.0)
    assert window_days(cases_per_window=500, patients_per_week=250) == pytest.approx(14.0)


def test_budget_conversion_for_one_tool():
    assert false_alarms_per_1000_windows(12, window_days=14) == pytest.approx(12 / (365.25 / 14) * 1000)      # about 460
    assert false_alarms_per_1000_windows(1, window_days=7) == pytest.approx(19.16, abs=0.01)
    assert false_alarms_per_1000_windows(0, window_days=7) == 0.0


def test_a_shared_committee_divides_its_capacity_across_the_tools():
    per_tool = false_alarms_per_1000_windows(4, window_days=7, tools_sharing=4)
    assert per_tool == pytest.approx(false_alarms_per_1000_windows(1, window_days=7))
    assert false_alarms_per_1000_windows(4, window_days=7, tools_sharing=1) == pytest.approx(4 * per_tool)


def test_invalid_inputs_are_rejected():
    for kwargs in ({"window_days": 0}, {"window_days": -3}, {"tools_sharing": 0}):
        with pytest.raises(ValueError):
            false_alarms_per_1000_windows(1, **{"window_days": 7, **kwargs})
    with pytest.raises(ValueError):
        window_days(500, 0)
