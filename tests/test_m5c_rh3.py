"""RH3 mixture streams: random reviewer failures, harm-defined window labels, and the AUROC gain (design section 9.1)."""

from dataclasses import replace

import numpy as np

from src.monitor_module.mixture import WindowRatio, auroc_gain, mixture_schedule
from src.monitor_module.review_layer import REVIEW_ORDER, build_review, build_review_params
from src.monitor_module.streams import build_stream, progress_schedule
from src.sim_module.reviewer_schedule import PARAM_OF, reviewer_schedule

LEVELS = {"reviewer_acceptance_of_ai_errors": [0.1, 0.25, 0.4], "review_coverage_drop": [0.9, 0.7, 0.4], "review_latency_increase": [2.0, 4.0, 10.0]}


def test_build_review_is_the_special_case_of_build_review_params(cfg, world, model):
    stream = build_stream(world, model, None, "abrupt", 20, cfg, np.random.default_rng(3))
    progress = progress_schedule("abrupt", 20, cfg.stream.monitored_windows, cfg.stream.ramp_windows)
    a = build_review(stream, cfg, "review_coverage_drop", 0.7, progress, np.random.default_rng(4))
    par = reviewer_schedule("review_coverage_drop", 0.7, np.concatenate([np.zeros(stream.n_baseline), progress]), cfg.reviewer)
    b = build_review_params(stream, cfg, par, np.random.default_rng(4))
    assert (a.examined == b.examined).all() and np.allclose(a.latency, b.latency)


def test_mixture_schedule_has_one_or_two_distinct_events(cfg):
    counts = set()
    for seed in range(60):
        par, events = mixture_schedule(np.random.default_rng(seed), 52, 104, cfg.reviewer, LEVELS, 10, 40)
        counts.add(len(events))
        assert len({e.kind for e in events}) == len(events)
        for e in events:
            assert 10 <= e.start <= 40 and e.level in LEVELS[e.kind]
            name = PARAM_OF[e.kind]
            assert np.allclose(par[name][:52 + e.start], getattr(cfg.reviewer, name))      # nothing moves before the start
            assert not np.isclose(par[name][-1], getattr(cfg.reviewer, name))               # and it has moved by the end
    assert counts == {1, 2}


def test_window_ratio_is_one_at_baseline_and_rises_with_harsher_failure(cfg, world, model):
    ratio = WindowRatio(cfg, model.apply(world.sample(60_000, np.random.default_rng(1))))
    base = {"coverage": cfg.reviewer.coverage, "automation_bias": 0.0, "false_reject": cfg.reviewer.false_reject, "latency_median": cfg.reviewer.latency_median}
    assert abs(ratio(base) - 1.0) < 1e-9
    mild, harsh = ratio({**base, "coverage": 0.9}), ratio({**base, "coverage": 0.4})
    assert 1.0 < mild < harsh
    assert ratio({**base, "coverage": 0.9}) == mild          # cached: identical parameters give the identical value


def test_auroc_gain_positive_for_an_informative_extra_feature_and_null_for_a_useless_one():
    rng = np.random.default_rng(0)
    n_streams, w = 400, 20
    y = (rng.random((n_streams, w)) < 0.4).astype(int)
    f1 = y + rng.normal(0, 1.5, y.shape)                  # weak
    f2 = y + rng.normal(0, 0.5, y.shape)                  # strong
    junk = rng.normal(0, 1, y.shape)
    half = n_streams // 2
    good = auroc_gain(y[:half], f1[:half, :, None], np.stack([f1, f2], axis=2)[:half], y[half:], f1[half:, :, None],
                      np.stack([f1, f2], axis=2)[half:], np.random.default_rng(1), n_boot=300)
    assert good.gain > 0.1 and good.lo > 0 and good.p < 0.01
    bad = auroc_gain(y[:half], f1[:half, :, None], np.stack([f1, junk], axis=2)[:half], y[half:], f1[half:, :, None],
                     np.stack([f1, junk], axis=2)[half:], np.random.default_rng(1), n_boot=300)
    assert abs(bad.gain) < 0.02 and bad.p > 0.05


def test_mixture_weights_change_how_often_each_failure_is_drawn(cfg):
    kinds = list(LEVELS)
    counts = dict.fromkeys(kinds, 0)
    for seed in range(300):
        _, events = mixture_schedule(np.random.default_rng(seed), 52, 104, cfg.reviewer, LEVELS, 10, 40, weights=[0.1, 0.8, 0.1])
        for e in events:
            counts[e.kind] += 1
    assert counts["review_coverage_drop"] > 2 * counts["reviewer_acceptance_of_ai_errors"]
    same = mixture_schedule(np.random.default_rng(4), 52, 104, cfg.reviewer, LEVELS, 10, 40)[1]
    assert same == mixture_schedule(np.random.default_rng(4), 52, 104, cfg.reviewer, LEVELS, 10, 40, weights=None)[1]


def test_mixture_features_shapes_labels_and_determinism(cfg, world, model):
    from src.monitor_module.mixture import mixture_features
    c = replace(cfg, reviewer=replace(cfg.reviewer, deadline=5.0, coverage=0.95))
    ratio = WindowRatio(c, model.apply(world.sample(40_000, np.random.default_rng(1))))
    feats, labels, counts = mixture_features(4, 7, c, world, model, LEVELS, ratio, (0, 8), 0.2)
    t = c.stream.monitored_windows
    assert feats[0].shape == (4, t, len(REVIEW_ORDER)) and feats[8].shape == (4, t, len(REVIEW_ORDER)) and labels.shape == (4, t) and len(counts) == 4
    assert set(np.unique(labels)) <= {0, 1} and labels.max() == 1
    again = mixture_features(4, 7, c, world, model, LEVELS, ratio, (0, 8), 0.2)
    assert np.allclose(feats[0], again[0][0]) and np.allclose(feats[8], again[0][8]) and (labels == again[1]).all()
    assert labels[:, :20].sum() == 0          # nothing is positive before the earliest possible start (monitored window 20)
    assert not np.allclose(feats[0][0], feats[0][1])                    # each stream has its own seed
    heavy = mixture_features(4, 7, c, world, model, LEVELS, ratio, (0, 8), 0.2, weights=[0.0, 1.0, 0.0])
    assert not np.allclose(heavy[0][0], feats[0])                       # the failure mix is passed through
    audited = mixture_features(4, 7, c, world, model, LEVELS, ratio, (0, 8), 0.9)
    assert not np.allclose(audited[0][0][:, :, 4], feats[0][:, :, 4]) and np.allclose(audited[0][0][:, :, 0], feats[0][:, :, 0])   # only H5 uses the audit
