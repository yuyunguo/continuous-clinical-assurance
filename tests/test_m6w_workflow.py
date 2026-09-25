"""Workflow layer: schedule (SC14, SC15, SC16, SC22), per-window O1-O6 and H7, and the harm-defined onset."""


from dataclasses import replace

import numpy as np
import pytest

from src.monitor_module.review_layer import build_review
from src.monitor_module.streams import build_stream, progress_schedule
from src.monitor_module.workflow_layer import WORKFLOW_ORDER, build_workflow, workflow_scores_multi, workflow_statistics
from src.sim_module.workflow_harm import WorkflowRatio
from src.sim_module.workflow_schedule import workflow_schedule


@pytest.fixture(scope="module")
def stream(cfg, world, model):
    return build_stream(world, model, None, "abrupt", 20, cfg, np.random.default_rng(3))


def test_workflow_config_defaults(cfg):
    w = cfg.workflow
    assert 0.9 < w.p_deliver < 1.0 and 0 < w.time_critical_share < 1 and w.tat_limit > 1 and 0 <= w.verify < 1


def test_schedule_moves_only_the_named_parameter(cfg):
    progress = np.array([0.0, 0.0, 0.5, 1.0])
    base = cfg.workflow
    esc = workflow_schedule("escalation_delivery_failure", 0.4, progress, base)
    assert np.allclose(esc["p_deliver"], base.p_deliver * np.array([1, 1, 0.8, 0.6])) and np.allclose(esc["tat_mult"], 1.0)
    tat = workflow_schedule("turnaround_time_increase", 8.0, progress, base)
    assert np.allclose(tat["tat_mult"], [1, 1, 4.5, 8]) and np.allclose(tat["p_deliver"], base.p_deliver)
    dup = workflow_schedule("alert_duplication", 6.0, progress, base)
    assert np.allclose(dup["alert_dup"], [1, 1, 3.5, 6])
    ver = workflow_schedule("error_propagation_unchecked", 0.5, progress, base)
    assert np.allclose(ver["verify"], base.verify - 0.5 * np.array([0, 0, 0.5, 1]))
    assert (workflow_schedule("error_propagation_unchecked", 9.0, progress, base)["verify"] >= 0).all()
    with pytest.raises(ValueError):
        workflow_schedule("nonsense", 1.0, progress, base)


def stats_for(cfg, stream, kind=None, level=0.0, seed=1):
    monitored = stream.y.shape[0] - stream.n_baseline
    progress = progress_schedule("abrupt", 20, monitored, cfg.stream.ramp_windows) if kind else np.zeros(monitored)
    full = np.concatenate([np.zeros(stream.n_baseline), progress])
    review = build_review(stream, cfg, None, 0.0, progress, np.random.default_rng(seed))
    par = workflow_schedule(kind, level, full, cfg.workflow) if kind else workflow_schedule("alert_duplication", 1.0, full, cfg.workflow)
    return workflow_statistics(build_workflow(stream, review, cfg, par, np.random.default_rng(seed + 1))), stream.n_baseline


def shift(stats, nb, name):
    x = stats[name]
    return np.nanmean(x[nb + 25:]) - np.nanmean(x[:nb])


def test_null_statistics_are_stable_and_named(cfg, stream):
    stats, nb = stats_for(cfg, stream)
    assert set(stats) == set(WORKFLOW_ORDER)
    assert 0.95 < np.nanmean(stats["H7"]) <= 1.0 and 0.9 < np.nanmean(stats["O1"]) <= 1.0
    for name in ("H7", "O1", "O3", "O5"):
        assert abs(shift(stats, nb, name)) < 0.02, name


def test_escalation_failure_moves_h7_o2_o1_only(cfg, stream):
    stats, nb = stats_for(cfg, stream, "escalation_delivery_failure", 0.4)
    assert shift(stats, nb, "H7") < -0.25 and shift(stats, nb, "O2") > 0.25 and shift(stats, nb, "O1") < -0.03
    assert abs(shift(stats, nb, "O3")) < 0.02 and abs(shift(stats, nb, "O4")) < 0.3


def test_alert_duplication_moves_o3_and_leaves_escalation_alone(cfg, stream):
    stats, nb = stats_for(cfg, stream, "alert_duplication", 6.0)
    base = np.nanmean(stats["O3"][:nb])
    assert shift(stats, nb, "O3") > 3 * base
    assert abs(shift(stats, nb, "H7")) < 0.03


def test_turnaround_increase_moves_o4_and_late_escalations(cfg, stream):
    stats, nb = stats_for(cfg, stream, "turnaround_time_increase", 8.0)
    assert shift(stats, nb, "O4") > 3.0 and shift(stats, nb, "O2") > 0.1
    assert abs(shift(stats, nb, "O3")) < 0.02


def test_unchecked_propagation_moves_o5_and_o6(cfg, stream):
    stats, nb = stats_for(cfg, stream, "error_propagation_unchecked", 0.5)
    assert shift(stats, nb, "O5") > 0.5 * np.nanmean(stats["O5"][:nb])
    assert np.nanmean(stats["O6"][nb + 25:]) > np.nanmean(stats["O6"][:nb])
    assert abs(shift(stats, nb, "H7")) < 0.03


def test_scores_lag_only_the_adjudicated_indicators(cfg, stream):
    from src.monitor_module.workflow_layer import WORKFLOW_LABEL_DEPENDENT
    monitored = stream.y.shape[0] - stream.n_baseline
    review = build_review(stream, cfg, None, 0.0, np.zeros(monitored), np.random.default_rng(5))
    par = workflow_schedule("alert_duplication", 1.0, np.zeros(stream.y.shape[0]), cfg.workflow)
    z = workflow_scores_multi(stream, build_workflow(stream, review, cfg, par, np.random.default_rng(6)), (0, 8))
    assert z[0].shape == (monitored, len(WORKFLOW_ORDER))
    for k, name in enumerate(WORKFLOW_ORDER):
        if name in WORKFLOW_LABEL_DEPENDENT:
            assert np.allclose(z[8][8:, k], z[0][:-8, k])
        else:
            assert np.allclose(z[8][:, k], z[0][:, k])


def test_workflow_ratio_is_one_at_baseline_and_rises_with_each_failure(cfg, world, model):
    ratio = WorkflowRatio(cfg, model.apply(world.sample(60_000, np.random.default_rng(1))))
    base = {"coverage": cfg.reviewer.coverage, "automation_bias": 0.0, "false_reject": cfg.reviewer.false_reject, "latency_median": cfg.reviewer.latency_median,
            "p_deliver": cfg.workflow.p_deliver, "tat_mult": 1.0, "alert_dup": 1.0, "verify": cfg.workflow.verify}
    assert abs(ratio(base) - 1.0) < 1e-9
    assert ratio({**base, "p_deliver": 0.6 * cfg.workflow.p_deliver}) > ratio({**base, "p_deliver": 0.9 * cfg.workflow.p_deliver}) > 1.0
    assert ratio({**base, "tat_mult": 8.0}) > ratio({**base, "tat_mult": 2.0}) > 1.0
    assert ratio({**base, "verify": 0.0}) > ratio({**base, "verify": 0.25}) > 1.0
    assert abs(ratio({**base, "alert_dup": 6.0}) - 1.0) < 1e-9             # duplication has no direct harm (design SC15)
    assert ratio({**base, "coverage": 0.4}) > 1.0                           # reviewer failures raise the extended harm too


def test_downstream_errors_follow_reviewer_misses_and_verification(cfg, stream):
    monitored = stream.y.shape[0] - stream.n_baseline
    review = build_review(stream, cfg, None, 0.0, np.zeros(monitored), np.random.default_rng(7))
    par = workflow_schedule("alert_duplication", 1.0, np.zeros(stream.y.shape[0]), cfg.workflow)
    wf = build_workflow(stream, review, cfg, par, np.random.default_rng(8))
    expected = np.mean(review.error & ~(review.in_time & review.recognized)) * (1 - cfg.workflow.verify)
    assert abs(wf.downstream.mean() - expected) < 0.004
    sentinel_share = wf.sentinel.sum() / max(wf.downstream.sum(), 1)
    assert 0.0 < sentinel_share < 0.5                                       # only the sentinel severity class is counted by O6
    # a failing reviewer (SC11) raises downstream errors even though the workflow itself is unchanged
    bad = build_review(stream, cfg, "review_coverage_drop", 0.4, progress_schedule("abrupt", 20, monitored, cfg.stream.ramp_windows), np.random.default_rng(7))
    worse = build_workflow(stream, bad, cfg, par, np.random.default_rng(8))
    assert worse.downstream[stream.n_baseline + 25:].mean() > 1.5 * wf.downstream[stream.n_baseline + 25:].mean()


def test_turnaround_statistic_is_the_95th_percentile(cfg, stream):
    stats, nb = stats_for(cfg, stream)
    p95 = cfg.workflow.tat_median * np.exp(cfg.workflow.tat_sigma * 1.645)
    assert abs(np.nanmean(stats["O4"][:nb]) - p95) < 0.1


def test_lag_set_and_orientation_of_the_scores(cfg, stream):
    from src.monitor_module.workflow_layer import WORKFLOW_LABEL_DEPENDENT
    assert set(WORKFLOW_LABEL_DEPENDENT) == {"O5", "O6"}
    monitored = stream.y.shape[0] - stream.n_baseline
    progress = progress_schedule("abrupt", 20, monitored, cfg.stream.ramp_windows)
    full = np.concatenate([np.zeros(stream.n_baseline), progress])
    review = build_review(stream, cfg, None, 0.0, progress, np.random.default_rng(9))
    wf = build_workflow(stream, review, cfg, workflow_schedule("escalation_delivery_failure", 0.4, full, cfg.workflow), np.random.default_rng(10))
    z = workflow_scores_multi(stream, wf, (0,))[0]
    assert z[25:, WORKFLOW_ORDER.index("H7")].mean() > 3.0 and z[25:, WORKFLOW_ORDER.index("O2")].mean() > 3.0    # both oriented so that larger is worse


def test_joint_schedule_couples_review_delay_into_turnaround_and_keeps_the_rest_at_baseline(cfg):
    from src.sim_module.joint_schedule import joint_schedule
    progress = np.array([0.0, 0.5, 1.0])
    q = joint_schedule("review_latency_increase", 10.0, progress, cfg)
    assert np.allclose(q["latency_median"], cfg.reviewer.latency_median * np.array([1, 5.5, 10]))
    assert np.allclose(q["tat_mult"], [1, 5.5, 10])                     # a slower review lengthens the turnaround it is part of
    assert np.allclose(q["p_deliver"], cfg.workflow.p_deliver) and np.allclose(q["coverage"], cfg.reviewer.coverage) and np.allclose(q["alert_dup"], 1.0)
    e = joint_schedule("escalation_delivery_failure", 0.4, progress, cfg)
    assert np.allclose(e["p_deliver"], cfg.workflow.p_deliver * np.array([1, 0.8, 0.6])) and np.allclose(e["latency_median"], cfg.reviewer.latency_median)
    assert np.allclose(e["tat_mult"], 1.0) and np.allclose(e["automation_bias"], cfg.reviewer.automation_bias) and np.allclose(e["coverage"], cfg.reviewer.coverage)
    assert np.allclose(e["alert_dup"], 1.0) and np.allclose(e["verify"], cfg.workflow.verify)
    assert set(e) == {"coverage", "automation_bias", "false_reject", "latency_median", "p_deliver", "tat_mult", "alert_dup", "verify"}
    with pytest.raises(ValueError):
        joint_schedule("nonsense", 1.0, progress, cfg)


def test_failed_escalations_carry_the_false_negative_factor(cfg, world, model):
    pop = model.apply(world.sample(60_000, np.random.default_rng(1)))
    high, low = replace(cfg, harm=replace(cfg.harm, fn_factor=3.0)), replace(cfg, harm=replace(cfg.harm, fn_factor=1.0))
    params = {"coverage": cfg.reviewer.coverage, "automation_bias": 0.0, "false_reject": cfg.reviewer.false_reject, "latency_median": cfg.reviewer.latency_median,
              "p_deliver": 0.5 * cfg.workflow.p_deliver, "tat_mult": 1.0, "alert_dup": 1.0, "verify": cfg.workflow.verify}
    base = {**params, "p_deliver": cfg.workflow.p_deliver}
    # an escalation that never arrives on a true alert is harm of the same kind as a missed deterioration, so it scales with the factor
    r_hi, r_lo = WorkflowRatio(high, pop), WorkflowRatio(low, pop)
    assert r_hi._total(params) - r_hi._total(base) > 2.0 * (r_lo._total(params) - r_lo._total(base))


def test_action_rate_o3a_falls_with_duplication_and_is_label_free(cfg, stream):
    from src.monitor_module.workflow_layer import WORKFLOW_DIRECTION, WORKFLOW_LABEL_DEPENDENT
    assert "O3a" in WORKFLOW_ORDER and WORKFLOW_DIRECTION["O3a"] == -1 and "O3a" not in WORKFLOW_LABEL_DEPENDENT
    stats, nb = stats_for(cfg, stream)
    assert abs(np.nanmean(stats["O3a"][:nb]) - cfg.workflow.action_base) < 0.02
    dup, nb = stats_for(cfg, stream, "alert_duplication", 6.0)
    expected = cfg.workflow.action_base / (1 + cfg.workflow.fatigue * 5.0)
    assert abs(np.nanmean(dup["O3a"][nb + 25:]) - expected) < 0.02
    assert shift(dup, nb, "O3a") < -0.2 and abs(shift(dup, nb, "H7")) < 0.03      # fatigue lowers the action rate and leaves escalation alone


def test_fatigue_coupling_raises_automation_bias_only_when_switched_on(cfg):
    from src.sim_module.joint_schedule import joint_schedule
    progress = np.array([0.0, 1.0])
    off = joint_schedule("alert_duplication", 6.0, progress, cfg)
    assert cfg.workflow.fatigue_bias == 0.0 and np.allclose(off["automation_bias"], cfg.reviewer.automation_bias)
    on_cfg = replace(cfg, workflow=replace(cfg.workflow, fatigue_bias=0.3))
    on = joint_schedule("alert_duplication", 6.0, progress, on_cfg)
    assert np.allclose(on["automation_bias"], cfg.reviewer.automation_bias + 0.3 * np.array([0.0, 1.0 - 1.0 / 6.0]))
    other = joint_schedule("escalation_delivery_failure", 0.4, progress, on_cfg)
    assert np.allclose(other["automation_bias"], cfg.reviewer.automation_bias)    # only duplicated alerts cause fatigue
