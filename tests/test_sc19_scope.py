"""SC19 scope creep and the C3 intended-use conformance indicator."""

import numpy as np
import pytest

from src.monitor_module import INDICATOR_ORDER, LABEL_DEPENDENT, build_stream, progress_schedule, stream_scores
from src.monitor_module.statistics import window_statistics
from src.sim_module import Batch, PerturbationConfig, PerturbationFactory


def make(cfg, world, model, level, progress, n=200_000, seed=1):
    pert = PerturbationFactory("out_of_scope_use_share")(PerturbationConfig(level_value=level))
    return pert.generate(world, model, n, np.random.default_rng(seed), progress)


def test_batch_carries_the_scope_flag_through_slice_select_and_concat():
    flags = np.array([1, 0, 0, 1], dtype=bool)
    a = Batch(x=np.zeros((4, 2)), y=np.array([0, 1, 0, 1]), g=np.zeros(4, dtype=int), severity=np.array([1, 2, 1, 3]), out_of_scope=flags)
    b = Batch(x=np.zeros((2, 2)), y=np.array([1, 1]), g=np.zeros(2, dtype=int), severity=np.array([1, 1]))          # no flag: in scope
    assert a.slice(1, 3).out_of_scope.tolist() == [False, False] and a.select(a.y == 1).out_of_scope.tolist() == [False, True]
    both = Batch.concat([a, b])
    assert both.out_of_scope.tolist() == [True, False, False, True, False, False] and both.n == 6
    assert Batch.concat([b, b]).out_of_scope.tolist() == [False] * 4


def test_out_of_scope_share_grows_with_level_and_progress_and_the_cases_are_shuffled(cfg, world, model):
    none = make(cfg, world, model, 0.3, 0.0)
    half = make(cfg, world, model, 0.3, 0.5)
    full = make(cfg, world, model, 0.3, 1.0)
    assert none.out_of_scope.sum() == 0
    assert abs(half.out_of_scope.mean() - 0.15) < 0.01 and abs(full.out_of_scope.mean() - 0.30) < 0.01
    first, second = full.out_of_scope[: full.n // 2].mean(), full.out_of_scope[full.n // 2:].mean()
    assert abs(first - second) < 0.01                        # spread over the batch, so every window carries its share


def test_model_performs_worse_outside_its_scope(cfg, world, model):
    b = make(cfg, world, model, 0.5, 1.0)
    out, inside = b.select(b.out_of_scope), b.select(~b.out_of_scope)
    se = lambda x: float(((x.decision == 1) & (x.y == 1)).sum() / (x.y == 1).sum())  # noqa: E731
    assert se(out) < se(inside) - 0.10


def test_c3_is_a_label_free_component_c_indicator_with_a_realistic_baseline(cfg, world, model):
    assert "C3" in INDICATOR_ORDER and "C3" not in LABEL_DEPENDENT
    stream = build_stream(world, model, None, "abrupt", 20, cfg, np.random.default_rng(3))
    assert stream.out_of_scope is not None and stream.out_of_scope.shape == stream.y.shape
    stats = window_statistics(stream, cfg)
    assert abs(stats["C3"].mean() - (1 - cfg.context.baseline_out_of_scope)) < 0.005
    assert stats["C3"][: stream.n_baseline].std() > 0.002    # noise, so a drop is measured against something


def test_scope_creep_lowers_c3_and_raises_its_score(cfg, world, model):
    pert = PerturbationFactory("out_of_scope_use_share")(PerturbationConfig(level_value=0.3))
    stream = build_stream(world, model, pert, "gradual", 20, cfg, np.random.default_rng(4))
    stats = window_statistics(stream, cfg)
    b = stream.n_baseline
    assert stats["C3"][b + 60:].mean() < stats["C3"][:b].mean() - 0.2
    z = stream_scores(stream, cfg, lag=0)
    assert z[70:, INDICATOR_ORDER.index("C3")].mean() > 5.0
    assert progress_schedule("gradual", 20, cfg.stream.monitored_windows, cfg.stream.ramp_windows)[-1] == 1.0
    assert set(pytest.importorskip("numpy").unique(stream.out_of_scope)) <= {False, True}
