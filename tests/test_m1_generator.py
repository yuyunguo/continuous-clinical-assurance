"""M1: Tier 1 generator and baseline model."""

import numpy as np

from src.eval_module.metrics import auroc, ece, sensitivity
from src.sim_module import DeployedModel, SyntheticWorld


def test_same_seed_gives_identical_batches(world):
    a = world.sample(500, np.random.default_rng(7))
    b = world.sample(500, np.random.default_rng(7))
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y) and np.array_equal(a.severity, b.severity)


def test_batch_shapes_and_ranges(world, cfg):
    batch = world.sample(2000, np.random.default_rng(1))
    assert batch.x.shape == (2000, cfg.generator.n_features)
    assert set(np.unique(batch.severity)) <= {1, 2, 3}
    assert set(np.unique(batch.y)) <= {0, 1} and set(np.unique(batch.g)) <= {0, 1}


def test_baseline_prevalence_and_subgroup_share(world, cfg):
    batch = world.sample(400_000, np.random.default_rng(2))
    assert abs(batch.y.mean() - cfg.generator.prevalence) < 0.005
    assert abs(batch.g.mean() - cfg.generator.subgroup_share) < 0.005


def test_baseline_bands_hold_on_every_seed(cfg):
    """Acceptance for M1: bands fixed in the protocol before this test was first run."""
    bands = cfg.m1_bands
    n_eval = cfg.stream.baseline_windows * cfg.stream.cases_per_window
    failures = []
    for seed in range(bands.seeds):
        world = SyntheticWorld(cfg.generator)
        model = DeployedModel(cfg.generator).fit(world, np.random.default_rng(seed))
        batch = model.apply(world.sample(n_eval, np.random.default_rng(10_000 + seed)))
        stats = {"ece": ece(batch.y, batch.reported_p), "auroc": auroc(batch.y, batch.reported_p),
                 "se": sensitivity(batch.y, batch.decision)}
        if not (stats["ece"] <= bands.ece_max and bands.auroc_min <= stats["auroc"] <= bands.auroc_max
                and abs(stats["se"] - cfg.generator.target_sensitivity) <= bands.sensitivity_tolerance):
            failures.append((seed, {k: round(v, 4) for k, v in stats.items()}))
    assert not failures, f"{len(failures)} of {bands.seeds} seeds outside the bands: {failures[:5]}"
