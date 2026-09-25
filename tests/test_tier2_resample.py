"""Block bootstrap over real windows, for null-stream calibration on a single real dataset (`Tier2-Readiness.md` §3:
"the same false-alarm calibration on real null streams, since real windows are not exchangeable" and "replicates
are resamples of one dataset").
"""

import numpy as np
import pytest

from src.sim_module.config import HarmConfig
from src.tier2_module.config import Tier2StreamConfig
from src.tier2_module.real_model import RealDeployedModel
from src.tier2_module.resample import block_bootstrap_stream, block_bootstrap_windows
from src.tier2_module.synthetic_stand_in import make_synthetic_cohort
from src.tier2_module.windowing import assign_windows

FEATURES = ("f0", "f1", "f2")


@pytest.fixture
def harm():
    return HarmConfig(severity_cutoff=2, weights=(1.0, 7.1, 100.0), severity_probs_positive=(0.5, 0.35, 0.15),
                      severity_probs_negative=(0.85, 0.12, 0.03), kappa=0.25, fn_factor=2.4)


@pytest.fixture
def cfg():
    return Tier2StreamConfig(feature_columns=FEATURES, window_size=20, target_sensitivity=0.8)


@pytest.fixture
def windowed(cfg):
    df = make_synthetic_cohort(n=2_000, feature_columns=FEATURES, rng=np.random.default_rng(0), prevalence=0.25)
    model = RealDeployedModel(cfg).fit(df.iloc[:1000], np.random.default_rng(1))
    return assign_windows(model.apply(df), cfg)   # 100 windows of 20 cases


def test_block_bootstrap_windows_returns_the_requested_window_count(cfg, windowed):
    out = block_bootstrap_windows(windowed, cfg, n_windows_out=30, block_size=5, rng=np.random.default_rng(2))
    assert out["window"].nunique() == 30
    assert len(out) == 30 * cfg.window_size


def test_block_bootstrap_windows_preserves_contiguous_blocks_from_the_real_sequence(cfg, windowed):
    """Every 5-window block in the output must equal some 5-window block that actually occurs in the real windows, in order."""
    out = block_bootstrap_windows(windowed, cfg, n_windows_out=15, block_size=5, rng=np.random.default_rng(3))
    real_first_order_keys = windowed.groupby("window")["order_key"].apply(lambda s: tuple(s.to_numpy()))
    for block_start in (0, 5, 10):
        block = out[(out["window"] >= block_start) & (out["window"] < block_start + 5)]
        block_order_keys = [tuple(block[block["window"] == w]["order_key"].to_numpy()) for w in range(block_start, block_start + 5)]
        # this block's 5 consecutive order-key tuples must match SOME 5 consecutive real windows, in the same relative order
        found = False
        n_real_windows = windowed["window"].nunique()
        for real_start in range(n_real_windows - 4):
            real_block = [real_first_order_keys[real_start + i] for i in range(5)]
            if real_block == block_order_keys:
                found = True
                break
        assert found, f"block at output window {block_start} does not match any contiguous 5-window run of the real data"


def test_block_bootstrap_windows_with_block_size_one_can_mix_non_adjacent_real_windows(cfg, windowed):
    """A degenerate block size of 1 should be able to draw non-adjacent real windows into consecutive output windows
    (i.i.d. resampling), unlike a larger block size which always preserves a contiguous real run."""
    out = block_bootstrap_windows(windowed, cfg, n_windows_out=200, block_size=1, rng=np.random.default_rng(4))
    drawn_order = out.groupby("window")["order_key"].first().to_numpy()
    # with 200 draws from 100 real windows' start order-keys, consecutive non-adjacency must occur somewhere
    real_starts = windowed.groupby("window")["order_key"].first().to_numpy()
    gaps = np.diff(np.searchsorted(np.sort(real_starts), drawn_order))
    assert not np.all(np.abs(gaps) == 1)


def test_block_bootstrap_windows_accepts_a_block_size_equal_to_the_real_window_count(cfg, windowed):
    """block_size == the real window count is the boundary case: only one valid block start (0) exists, so the
    resampled windows must be the whole real sequence, in its original order, every time."""
    n_real = windowed["window"].nunique()
    out = block_bootstrap_windows(windowed, cfg, n_windows_out=n_real, block_size=n_real, rng=np.random.default_rng(8))
    expected = windowed["order_key"].tolist()
    assert out["order_key"].tolist() == expected


def test_block_bootstrap_windows_handles_a_window_count_that_is_not_a_multiple_of_block_size(cfg, windowed):
    out = block_bootstrap_windows(windowed, cfg, n_windows_out=22, block_size=5, rng=np.random.default_rng(9))
    assert out["window"].nunique() == 22
    assert len(out) == 22 * cfg.window_size


def test_block_bootstrap_windows_rejects_a_block_size_larger_than_the_real_window_count(cfg, windowed):
    n_real = windowed["window"].nunique()
    with pytest.raises(ValueError, match="block_size"):
        block_bootstrap_windows(windowed, cfg, n_windows_out=10, block_size=n_real + 1, rng=np.random.default_rng(5))


def test_block_bootstrap_windows_is_reproducible_given_the_same_seed(cfg, windowed):
    a = block_bootstrap_windows(windowed, cfg, n_windows_out=25, block_size=4, rng=np.random.default_rng(6))
    b = block_bootstrap_windows(windowed, cfg, n_windows_out=25, block_size=4, rng=np.random.default_rng(6))
    assert a["order_key"].tolist() == b["order_key"].tolist()


def test_block_bootstrap_stream_builds_a_streamdata_with_the_requested_shape_and_zero_baseline(cfg, harm, windowed):
    stream = block_bootstrap_stream(windowed, cfg, harm, n_windows_out=40, block_size=5, rng=np.random.default_rng(7))
    assert stream.y.shape == (40, cfg.window_size)
    assert stream.n_baseline == 0
    assert stream.s0 is None
    assert stream.progress.shape == (40,)
    assert (stream.progress == 0.0).all()


def test_block_bootstrap_stream_accepts_a_nonzero_baseline(cfg, harm, windowed):
    """A calibration null stream needs its own standardization baseline (a subset of windows to compute the
    reference mean/SD from), distinct from the pure-null default of zero baseline windows."""
    stream = block_bootstrap_stream(windowed, cfg, harm, n_windows_out=40, block_size=5, rng=np.random.default_rng(7), n_baseline=10)
    assert stream.n_baseline == 10
    assert stream.progress.shape == (30,)
