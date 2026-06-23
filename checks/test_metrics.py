from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from pneumapost.tallies.diagnostics import auroc, specificity_at_sensitivity
from pneumapost.tallies.inference_stats import (
    benjamini_hochberg,
    bootstrap_ci,
    cohens_d,
    delong_test,
    holm_bonferroni,
)


def test_auroc_matches_reference() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    score = rng.normal(y * 0.8, 1.0)
    assert abs(auroc(y, score) - roc_auc_score(y, score)) < 1e-9


def test_specificity_at_high_sensitivity_is_monotone() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    score = rng.normal(y * 1.2, 1.0)
    loose = specificity_at_sensitivity(y, score, 0.80)
    strict = specificity_at_sensitivity(y, score, 0.99)
    assert loose >= strict


def test_delong_identical_scores_is_not_significant() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    score = rng.normal(y, 1.0)
    z, p = delong_test(y, score, score)
    assert abs(z) < 1e-6
    assert p > 0.99


def test_delong_flags_clear_separation_gap() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 800)
    weak = rng.normal(y * 0.3, 1.0)
    strong = rng.normal(y * 1.5, 1.0)
    _, p = delong_test(y, strong, weak)
    assert p < 0.05


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 500)
    score = rng.normal(y * 1.0, 1.0)
    lo, hi = bootstrap_ci(y, score, 300, 0.05, seed=7)
    point = auroc(y, score)
    assert lo <= point <= hi


def test_cohens_d_sign_and_scale() -> None:
    a = np.full(50, 2.0) + np.random.default_rng(0).normal(0, 0.1, 50)
    b = np.full(50, 0.0) + np.random.default_rng(1).normal(0, 0.1, 50)
    assert cohens_d(a, b) > 5.0


def test_multiple_comparison_corrections() -> None:
    pvals = [0.001, 0.02, 0.2, 0.7]
    holm = holm_bonferroni(pvals, alpha=0.05)
    bh = benjamini_hochberg(pvals, alpha=0.10)
    assert holm[0] and not holm[3]
    assert bh[0] and not bh[3]
    assert sum(bh) >= sum(holm)
