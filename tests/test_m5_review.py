"""Reviewer sampling and the EHOR estimators (design sections 10 and 11)."""

from dataclasses import replace

import numpy as np
import pytest

from src.eval_module.ehor import audit_sample, ehor_truth, iir_ht, ehor_ht, naive_ehor
from src.sim_module import interception_probability
from src.sim_module.review import sample_review


@pytest.fixture(scope="module")
def batch(cfg, world, model):
    b = model.apply(world.sample(40_000, np.random.default_rng(11)))
    return b


def with_reviewer(cfg, **kw):
    return replace(cfg, reviewer=replace(cfg.reviewer, **kw))


def test_defaults_leave_interception_unchanged(cfg, batch):
    """Automation bias defaults to zero, so every earlier result stands."""
    assert cfg.reviewer.automation_bias == 0.0
    base = cfg.reviewer
    assert np.allclose(interception_probability(batch, base), base.coverage * (1 / (1 + np.exp(-(base.beta0 + base.beta_confidence * (1 - np.where(batch.decision == 1, batch.reported_p, 1 - batch.reported_p)) + base.beta_severity * (batch.severity - 1))))))


def test_automation_bias_lowers_interception(cfg, batch):
    low = interception_probability(batch, cfg.reviewer).mean()
    high = interception_probability(batch, replace(cfg.reviewer, automation_bias=0.4)).mean()
    assert abs(high - 0.6 * low) < 1e-9


def test_requires_detection_definition(cfg, batch):
    out = sample_review(batch, cfg, np.random.default_rng(1))
    expected = (batch.decision != batch.y) & (batch.severity >= cfg.harm.severity_cutoff)
    assert (out.R == expected).all()
    assert not (out.I & ~out.R).any()          # only errors requiring detection are intercepted
    assert not (out.Z & (batch.decision != batch.y)).any()   # wrongly intervened outputs were correct


def test_sampled_interception_matches_expectation(cfg, batch):
    out = sample_review(batch, cfg, np.random.default_rng(2))
    err = batch.decision != batch.y
    assert abs(out.intercepted_any[err].mean() - interception_probability(batch, cfg.reviewer)[err].mean()) < 0.01


def test_identity_cov_times_conditional(cfg, batch):
    out = sample_review(batch, with_reviewer(cfg, coverage=0.7), np.random.default_rng(3))
    truth = ehor_truth(out)
    assert abs(truth.ehor - truth.coverage * truth.conditional) < 1e-12
    assert abs(truth.coverage - 0.7) < 0.03


def test_no_coverage_means_zero_ehor(cfg, batch):
    out = sample_review(batch, with_reviewer(cfg, coverage=0.0), np.random.default_rng(4))
    truth = ehor_truth(out)
    assert truth.ehor == 0.0 and not out.I.any() and not out.Z.any()


def test_false_reject_rate(cfg, batch):
    out = sample_review(batch, with_reviewer(cfg, false_reject=0.1), np.random.default_rng(5))
    ok = batch.decision == batch.y
    assert abs(out.Z[ok].mean() - 0.1) < 0.01
    assert abs(iir_ht(out, np.ones(batch.n)) - out.Z[ok].mean()) < 1e-12


def test_full_audit_recovers_truth_exactly(cfg, batch):
    out = sample_review(batch, cfg, np.random.default_rng(6))
    assert abs(ehor_ht(out, np.ones(batch.n)) - ehor_truth(out).ehor) < 1e-12


def test_audit_weights_are_inverse_inclusion(cfg, batch):
    out = sample_review(batch, cfg, np.random.default_rng(7))
    sampled, w = audit_sample(out, batch, fraction=0.05, stratified=True, rng=np.random.default_rng(8))
    assert sampled.sum() > 0 and (w[sampled] > 0).all() and (w[~sampled] == 0).all()
    # weights sum to about the population size (Horvitz-Thompson)
    assert abs(w.sum() / batch.n - 1.0) < 0.05


def test_ht_is_nearly_unbiased_and_naive_is_inflated(cfg, batch):
    out = sample_review(batch, cfg, np.random.default_rng(9))
    truth = ehor_truth(out).ehor
    est = []
    for s in range(60):
        sampled, w = audit_sample(out, batch, fraction=0.2, stratified=True, rng=np.random.default_rng(100 + s))
        est.append(ehor_ht(out, w))
    assert abs(np.mean(est) - truth) < 0.02
    assert naive_ehor(out, discovery=0.3) > truth + 0.05
    assert abs(naive_ehor(out, discovery=1.0) - truth) < 1e-12


def test_audit_size_and_oversampling_of_low_confidence(cfg, batch):
    out = sample_review(batch, cfg, np.random.default_rng(12))
    sampled, w = audit_sample(out, batch, fraction=0.05, stratified=True, rng=np.random.default_rng(13))
    assert abs(sampled.mean() - 0.05) < 0.005
    confidence = np.where(batch.decision == 1, batch.reported_p, 1.0 - batch.reported_p)
    low = confidence <= np.quantile(confidence, 1 / 3)
    high = confidence >= np.quantile(confidence, 2 / 3)
    assert sampled[low].mean() > 2 * sampled[high].mean()
    flat, _ = audit_sample(out, batch, fraction=0.05, stratified=False, rng=np.random.default_rng(14))
    assert abs(flat.mean() - 0.05) < 0.005


def test_ht_standard_error_zero_for_full_audit_and_covers(cfg, batch):
    from src.eval_module.ehor import ehor_ht_se
    out = sample_review(batch, cfg, np.random.default_rng(15))
    assert ehor_ht_se(out, np.ones(batch.n)) == 0.0
    truth = ehor_truth(out).ehor
    hits = 0
    for s in range(200):
        _, w = audit_sample(out, batch, fraction=0.2, stratified=True, rng=np.random.default_rng(500 + s))
        est, se = ehor_ht(out, w), ehor_ht_se(out, w)
        hits += abs(est - truth) <= 1.96 * se
    assert 0.90 <= hits / 200 <= 0.99


def test_sampled_review_honours_the_deadline(cfg, batch):
    tight = replace(cfg, reviewer=replace(cfg.reviewer, deadline=2.0))
    out = sample_review(batch, tight, np.random.default_rng(16))
    err = batch.decision != batch.y
    expected = interception_probability(batch, tight.reviewer)[err].mean()
    assert abs(out.intercepted_any[err].mean() - expected) < 0.01
    loose = sample_review(batch, cfg, np.random.default_rng(16))
    assert out.reviewed.mean() < loose.reviewed.mean() - 0.03
    truth = ehor_truth(out)
    assert abs(truth.ehor - truth.coverage * truth.conditional) < 1e-12
