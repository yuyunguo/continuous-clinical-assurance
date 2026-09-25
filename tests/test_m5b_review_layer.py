"""Reviewer schedule (SC10-SC13), latency, and the per-window H statistics."""

from dataclasses import replace

import numpy as np
import pytest

from src.monitor_module.review_layer import REVIEW_ORDER, build_review, review_statistics
from src.monitor_module.streams import build_stream, progress_schedule
from src.sim_module import interception_probability
from src.sim_module.reviewer import in_time_probability
from src.sim_module.reviewer_schedule import reviewer_schedule


@pytest.fixture(scope="module")
def stream(cfg, world, model):
    return build_stream(world, model, None, "abrupt", 20, cfg, np.random.default_rng(3))


def test_default_latency_has_no_deadline_so_earlier_results_stand(cfg):
    assert in_time_probability(cfg.reviewer) == 1.0


def test_deadline_lowers_in_time_probability_and_interception(cfg, stream):
    tight = replace(cfg.reviewer, deadline=2.0)
    assert 0.5 < in_time_probability(tight) < 0.99
    b = stream_batch(stream)
    assert interception_probability(b, tight).mean() < interception_probability(b, cfg.reviewer).mean()


def stream_batch(stream):
    from src.sim_module import Batch
    flat = lambda a: a.reshape(-1)  # noqa: E731
    return Batch(x=np.zeros((flat(stream.y).size, 1)), y=flat(stream.y), g=flat(stream.g), severity=flat(stream.severity),
                 p_hat=flat(stream.reported_p), reported_p=flat(stream.reported_p), decision=flat(stream.decision))


def test_schedule_moves_the_named_parameter_only(cfg):
    progress = np.array([0.0, 0.0, 0.5, 1.0])
    bias = reviewer_schedule("reviewer_acceptance_of_ai_errors", 0.4, progress, cfg.reviewer)
    assert np.allclose(bias["automation_bias"], [0, 0, 0.2, 0.4]) and np.allclose(bias["coverage"], cfg.reviewer.coverage)
    cov = reviewer_schedule("review_coverage_drop", 0.4, progress, cfg.reviewer)
    assert np.allclose(cov["coverage"], [1, 1, 0.7, 0.4]) and np.allclose(cov["automation_bias"], 0)
    lat = reviewer_schedule("review_latency_increase", 10.0, progress, cfg.reviewer)
    assert np.allclose(lat["latency_median"], cfg.reviewer.latency_median * np.array([1, 1, 5.5, 10]))
    with pytest.raises(ValueError):
        reviewer_schedule("nonsense", 1.0, progress, cfg.reviewer)


def stats_for(cfg, stream, kind=None, level=0.0, seed=1, audit=0.2):
    reviewer = replace(cfg.reviewer, deadline=5.0)
    monitored = stream.y.shape[0] - stream.n_baseline
    progress = progress_schedule("abrupt", 20, monitored, cfg.stream.ramp_windows) if kind else np.zeros(monitored)
    rev = build_review(stream, replace(cfg, reviewer=reviewer), kind, level, progress, np.random.default_rng(seed))
    return review_statistics(rev, audit, np.random.default_rng(seed + 1)), stream.n_baseline


def mean_shift(stats, n_base, name):
    x = stats[name]
    return np.nanmean(x[n_base + 25:]) - np.nanmean(x[:n_base])   # after the scenario has reached full effect


def test_null_review_statistics_are_stable(cfg, stream):
    stats, nb = stats_for(cfg, stream)
    assert set(stats) == set(REVIEW_ORDER)
    assert abs(mean_shift(stats, nb, "H1")) < 0.03
    assert 0.9 < np.nanmean(stats["H1"]) <= 1.0


def test_automation_bias_moves_h4_up_h2_down_and_leaves_h1(cfg, stream):
    stats, nb = stats_for(cfg, stream, "reviewer_acceptance_of_ai_errors", 0.4)
    assert mean_shift(stats, nb, "H4") > 0.2
    assert mean_shift(stats, nb, "H2") < -0.02
    assert mean_shift(stats, nb, "H5") < -0.1
    assert abs(mean_shift(stats, nb, "H1")) < 0.03


def test_coverage_drop_moves_h1_and_h5(cfg, stream):
    stats, nb = stats_for(cfg, stream, "review_coverage_drop", 0.4)
    assert mean_shift(stats, nb, "H1") < -0.4
    assert mean_shift(stats, nb, "H5") < -0.3


def test_queue_delay_moves_latency_and_ehor_but_not_documented_review(cfg, stream):
    stats, nb = stats_for(cfg, stream, "review_latency_increase", 10.0)
    assert mean_shift(stats, nb, "H6") > 5.0
    assert mean_shift(stats, nb, "H5") < -0.3
    assert abs(mean_shift(stats, nb, "H1")) < 0.03 and abs(mean_shift(stats, nb, "H4")) < 0.05


def test_audit_h5_is_nan_without_required_cases_and_bounded(cfg, stream):
    stats, _ = stats_for(cfg, stream, audit=0.01)
    valid = stats["H5"][~np.isnan(stats["H5"])]
    assert ((valid >= 0) & (valid <= 1)).all()


def test_only_examined_cases_can_be_overridden_so_coverage_drop_leaves_rates_among_reviewed_unchanged(cfg, stream):
    stats, nb = stats_for(cfg, stream, "review_coverage_drop", 0.4)
    for name in ("H2", "H3", "H4"):
        assert abs(mean_shift(stats, nb, name)) < 0.05, name


def test_appropriate_override_rate_is_the_share_of_overrides_that_correct_an_error(cfg, stream):
    stats, _ = stats_for(cfg, stream)
    assert 0.5 < np.nanmean(stats["H3"]) < 0.98
    noisy = stats_for(replace(cfg, reviewer=replace(cfg.reviewer, false_reject=0.5)), stream)[0]
    assert np.nanmean(noisy["H3"]) < np.nanmean(stats["H3"]) - 0.3
    assert np.nanmean(noisy["H2"]) > np.nanmean(stats["H2"]) + 0.2


def test_review_scores_apply_the_label_lag_only_to_adjudicated_indicators(cfg, stream):
    from src.monitor_module.review_layer import REVIEW_LABEL_DEPENDENT, review_scores_multi
    monitored = stream.y.shape[0] - stream.n_baseline
    progress = np.zeros(monitored)
    rev = build_review(stream, cfg, None, 0.0, progress, np.random.default_rng(5))
    z = review_scores_multi(stream, rev, 0.2, (0, 8), np.random.default_rng(6))
    assert z[0].shape == (monitored, len(REVIEW_ORDER))
    for k, name in enumerate(REVIEW_ORDER):
        if name in REVIEW_LABEL_DEPENDENT:
            assert np.allclose(z[8][8:, k], z[0][:-8, k])
        else:
            assert np.allclose(z[8][:, k], z[0][:, k])


def test_h2_is_two_sided_through_an_upward_arm(cfg, stream):
    """The indicator record says the override rate is charted in both directions. Blanket rejection raises it, so H2u must respond to a rise."""
    from src.monitor_module.review_layer import REVIEW_DIRECTION
    assert "H2u" in REVIEW_ORDER and REVIEW_DIRECTION["H2"] == -1 and REVIEW_DIRECTION["H2u"] == 1
    monitored = stream.y.shape[0] - stream.n_baseline
    progress = progress_schedule("abrupt", 20, monitored, cfg.stream.ramp_windows)
    from src.monitor_module.review_layer import review_scores_multi
    rev = build_review(stream, replace(cfg, reviewer=replace(cfg.reviewer, deadline=5.0)), "blanket_rejection", 0.5, progress, np.random.default_rng(1))
    z = review_scores_multi(stream, rev, 0.2, (0,), np.random.default_rng(2))[0]
    up, down = z[30:, REVIEW_ORDER.index("H2u")].mean(), z[30:, REVIEW_ORDER.index("H2")].mean()
    assert up > 5.0 and down < -5.0                 # a rise in overrides is a large positive score on the upward arm and a large negative one on H2
    rev0 = build_review(stream, replace(cfg, reviewer=replace(cfg.reviewer, deadline=5.0)), "review_coverage_drop", 0.4, progress, np.random.default_rng(1))
    stats = review_statistics(rev0, 0.2, np.random.default_rng(2))
    assert np.allclose(stats["H2u"], stats["H2"], equal_nan=True)     # the same statistic, read in the other direction
