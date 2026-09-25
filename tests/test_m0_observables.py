"""M0: every data element of every predictive-applicable indicator maps to a simulator output."""

import copy

from src.sim_module.observables import indicator_status, load_indicators, load_observables, validate_observables


def test_every_data_element_is_mapped_and_outputs_exist():
    errors = validate_observables(load_indicators(), load_observables())
    assert errors == []


def test_only_predictive_applicable_indicators_are_covered():
    status = indicator_status(load_indicators(), load_observables())
    assert len(status) == 34 and "L5" in status and "G1" not in status and "L2" not in status


def test_unmapped_element_is_reported():
    obs = copy.deepcopy(load_observables())
    obs["elements"].pop("review logs with timestamps")
    errors = validate_observables(load_indicators(), obs)
    assert any("review logs with timestamps" in e for e in errors)


def test_mapping_to_unknown_output_is_reported():
    obs = copy.deepcopy(load_observables())
    obs["elements"]["scores"] = "no_such_output"
    assert any("no_such_output" in e for e in validate_observables(load_indicators(), obs))


def test_indicator_is_untestable_until_all_its_outputs_are_implemented():
    status = indicator_status(load_indicators(), load_observables())
    assert status["P1"]["testable_now"] is True          # predictions and outcomes exist in the data layer
    assert status["H5"]["testable_now"] is True           # the review layer supplies review logs and an audit sample
    assert status["H7"]["testable_now"] is True           # the workflow layer supplies escalation logs and criteria flags
    assert status["O3"]["testable_now"] is True           # burden and action rate: the workflow layer supplies alert and action logs
    assert status["C1"]["testable_now"] is False         # case-mix drift needs categorical case-mix attributes, which are not simulated
    assert "case_mix_attributes" in status["C1"]["missing_outputs"]
