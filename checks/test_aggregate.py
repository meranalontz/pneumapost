from __future__ import annotations

import torch

from pneumapost.apparatus.prototypes import shrink_to_marginal
from pneumapost.dispatch.aggregate import confidence_weighted
from pneumapost.fittings.geometry import chordal_norm, to_hypersphere


def test_identical_sites_pool_to_themselves() -> None:
    dim = 12
    means = to_hypersphere(torch.randn(2, 2, dim))
    counts = torch.full((2, 2), 100.0)
    pooled, pops = confidence_weighted(
        [means, means, means], [counts, counts, counts], 24.0, 1 << 30
    )
    expected = to_hypersphere(shrink_to_marginal(means, 3 * counts, 24.0))
    assert torch.allclose(pooled, expected, atol=1e-4)
    assert torch.allclose(pops, 3 * counts)


def test_low_count_cell_is_shrunk_toward_marginal() -> None:
    dim = 10
    means = to_hypersphere(torch.randn(2, 2, dim))
    counts = torch.tensor([[400.0, 2.0], [380.0, 3.0]])
    shrunk = shrink_to_marginal(means, counts, kappa=24.0)
    male_move = float(chordal_norm(shrunk[0, 1], means[0, 1]))
    female_move = float(chordal_norm(shrunk[0, 0], means[0, 0]))
    assert male_move > female_move


def test_pooled_prototypes_live_on_sphere() -> None:
    dim = 16
    means = to_hypersphere(torch.randn(2, 2, dim))
    counts = torch.full((2, 2), 50.0)
    pooled, _ = confidence_weighted([means], [counts], 24.0, 1 << 30)
    norms = pooled.reshape(-1, dim).norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
