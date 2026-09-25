"""Shared fixtures for the Phase II tests."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.sim_module import DeployedModel, SyntheticWorld, load_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def cfg():
    """Protocol configuration (run/conf/experiment/phase2_defaults.yaml)."""
    return load_config()


@pytest.fixture(scope="session")
def world(cfg):
    """Baseline synthetic world."""
    return SyntheticWorld(cfg.generator)


@pytest.fixture(scope="session")
def model(cfg, world):
    """Deployed model fit on a training sample that is disjoint from every evaluation sample."""
    return DeployedModel(cfg.generator).fit(world, np.random.default_rng(cfg.seed))


@pytest.fixture(scope="session")
def scenarios():
    """The pre-specified scenario catalog."""
    return {s["id"]: s for s in yaml.safe_load((ROOT / "phase2" / "scenarios.yaml").read_text(encoding="utf-8"))}
