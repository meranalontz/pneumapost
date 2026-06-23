from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from pneumapost.fittings.geometry import pairwise_squared, squared_distance, to_hypersphere


@given(
    rows=st.integers(min_value=1, max_value=8),
    dim=st.integers(min_value=2, max_value=16),
    seed=st.integers(min_value=0, max_value=2**16),
)
@settings(max_examples=40, deadline=None)
def test_projection_is_unit_norm(rows: int, dim: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, dim, generator=g) * 7.0
    norms = to_hypersphere(x).norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@given(seed=st.integers(min_value=0, max_value=2**16))
@settings(max_examples=30, deadline=None)
def test_distance_is_self_zero_and_symmetric(seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(5, 9, generator=g)
    b = torch.randn(5, 9, generator=g)
    assert torch.allclose(squared_distance(a, a), torch.zeros(5), atol=1e-5)
    assert torch.allclose(squared_distance(a, b), squared_distance(b, a), atol=1e-5)


def test_pairwise_matches_loop() -> None:
    g = torch.Generator().manual_seed(3)
    z = torch.randn(6, 4, generator=g)
    anchors = torch.randn(3, 4, generator=g)
    table = pairwise_squared(z, anchors)
    for i in range(6):
        for j in range(3):
            assert torch.allclose(table[i, j], squared_distance(z[i], anchors[j]), atol=1e-5)
