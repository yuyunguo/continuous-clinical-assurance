"""M4: TAFD classification and matched false-alarm calibration."""

from dataclasses import replace

import numpy as np

from src.eval_module.tafd import bootstrap_rho, classify_replicate, relative_reduction
from src.monitor_module import calibrate_threshold, episode_starts, null_scores, system_events


def test_classification_covers_every_case():
    t = 40
    none = np.zeros(t, dtype=bool)
    assert classify_replicate(none, s0=10, t0=12, horizon=20, j=2) == {"tafd": 20, "censored": True, "false_alarms": 0, "anticipatory": False, "carry_in": False, "lead_time": 0}
    hit = none.copy()
    hit[15] = True
    assert classify_replicate(hit, 10, 12, 20, 2)["tafd"] == 3
    early = none.copy()
    early[11] = True
    assert classify_replicate(early, 10, 12, 20, 2)["tafd"] == 0 and classify_replicate(early, 10, 12, 20, 2)["anticipatory"]
    late = none.copy()
    late[40 - 1] = True
    assert classify_replicate(late, 10, 12, 20, 2)["tafd"] == 20          # beyond the horizon counts as not detected
    fa = none.copy()
    fa[[3, 9, 20]] = True
    out = classify_replicate(fa, 10, 12, 20, 2)
    assert out["false_alarms"] == 2 and out["carry_in"] is True and out["tafd"] == 8   # starts at windows 3 and 9, both before s0


def test_relative_reduction_and_bootstrap_are_paired():
    rng = np.random.default_rng(0)
    conv = rng.integers(10, 40, 200).astype(float)
    cca = np.maximum(conv - 8, 0)
    assert abs(relative_reduction(conv, cca) - (conv.mean() - cca.mean()) / conv.mean()) < 1e-12
    lo, hi = bootstrap_rho(conv, cca, np.random.default_rng(1), n_boot=400)
    assert lo < relative_reduction(conv, cca) < hi and lo > 0


def test_calibrated_threshold_meets_the_budget_on_independent_null_streams(world, model, cfg):
    c = replace(cfg, stream=replace(cfg.stream, cases_per_window=250, baseline_windows=16, monitored_windows=40))
    calib = null_scores(world, model, c, streams=160, seed=101)
    evaluation = null_scores(world, model, c, streams=160, seed=202)
    target, j = 20.0, c.routing.hysteresis_j
    h = calibrate_threshold(calib, list(range(calib.shape[2])), target, j)
    events = episode_starts(system_events(evaluation, list(range(evaluation.shape[2])), h), j)
    windows = events.shape[0] * events.shape[1]
    rate = events.sum() / windows * 1000
    se = np.sqrt(max(events.sum(), 1)) / windows * 1000
    assert abs(rate - target) <= 3 * se + 0.15 * target, (rate, target, se, h)


def test_more_indicators_need_a_higher_threshold_at_the_same_budget(world, model, cfg):
    c = replace(cfg, stream=replace(cfg.stream, cases_per_window=250, baseline_windows=16, monitored_windows=40))
    scores = null_scores(world, model, c, streams=160, seed=303)
    j = c.routing.hysteresis_j
    h_few = calibrate_threshold(scores, [0, 1, 3], 20.0, j)
    h_all = calibrate_threshold(scores, list(range(scores.shape[2])), 20.0, j)
    assert h_all > h_few
