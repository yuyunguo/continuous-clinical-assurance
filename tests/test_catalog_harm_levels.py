"""The catalog's levels must reproduce their harm targets, so that a change to the harm weights is noticed."""

import numpy as np
import pytest

from src.sim_module import HARM_POPULATION, parse_level, ratio_at

TOLERANCE = 0.07
FAIR_OR_BLIND = ("SC02", "SC04", "SC07", "SC08", "SC09", "SC17", "SC19", "SC25")
HARMLESS = ("SC01", "SC05")


def catalog_ratio(scenarios, sid, level, world, model, cfg, baseline):
    spec = scenarios[sid]["perturbation"]
    return ratio_at(spec["type"], parse_level(spec["levels"][level]), world, model, cfg, HARM_POPULATION, spec.get("mode"), baseline)


@pytest.fixture(scope="module")
def baseline(world, model, cfg):
    from src.sim_module import baselines_for
    return baselines_for(world, model, cfg, HARM_POPULATION)


@pytest.mark.parametrize("sid", FAIR_OR_BLIND)
def test_levels_reproduce_their_harm_targets(sid, scenarios, world, model, cfg, baseline):
    targets = scenarios[sid]["harm_targets"]
    got = [catalog_ratio(scenarios, sid, lv, world, model, cfg, baseline) for lv in ("low", "medium", "high")]
    assert np.allclose(got, targets, atol=TOLERANCE), (sid, got, targets)


@pytest.mark.parametrize("sid", HARMLESS)
def test_harmless_drift_scenarios_stay_below_the_onset_threshold(sid, scenarios, world, model, cfg, baseline):
    limit = 1 + cfg.harm.kappa
    got = [catalog_ratio(scenarios, sid, lv, world, model, cfg, baseline) for lv in ("low", "medium", "high")]
    assert max(got) < limit and scenarios[sid]["purpose"] == "harmless_drift" and scenarios[sid]["primary_tafd"] is False, (sid, got)


def test_every_scenario_has_a_purpose_and_purposes_match_their_flags(scenarios):
    vocabulary = {"control", "fair_test", "blind_spot", "harmless_drift", "process"}
    for sid, s in scenarios.items():
        assert s["purpose"] in vocabulary, sid
        assert (s["purpose"] == "control") == (s["family"] == "control"), sid
        if s["purpose"] == "harmless_drift":
            assert s["primary_tafd"] is False, sid
