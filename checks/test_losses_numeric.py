from __future__ import annotations

import torch

from pneumapost.apparatus.fusion import FusionState
from pneumapost.fittings.geometry import to_hypersphere
from pneumapost.fittings.glossary import N_MODALITIES, FrankingSpec
from pneumapost.franking.objective import _confidence, _sex_repulsion, evaluate
from pneumapost.intake.modalities import Batch


def _toy_batch(z: torch.Tensor, verdict: torch.Tensor, sex: torch.Tensor) -> Batch:
    n = z.size(0)
    return Batch(
        breath=torch.zeros(n, 1),
        microbiome=torch.zeros(n, 1, 1),
        omics=torch.zeros(n, 1),
        presence=torch.ones(n, N_MODALITIES, dtype=torch.bool),
        verdict=verdict,
        sex=sex,
        never_smoker=torch.zeros(n, dtype=torch.bool),
        driver_negative=torch.zeros(n, dtype=torch.bool),
        arm=torch.zeros(n, dtype=torch.long),
        site=torch.zeros(n, dtype=torch.long),
    )


def test_sex_repulsion_activates_only_when_prototypes_too_close() -> None:
    dim = 8
    means = torch.zeros(2, 2, dim)
    means[0, 0, 0] = 1.0
    means[0, 1, 0] = 1.0
    means[1, 0, 0] = 1.0
    means[1, 1, 0] = -1.0
    counts = torch.ones(2, 2)
    near = _sex_repulsion(means, counts, margin=0.75)
    means[0, 1, 0] = -1.0
    far = _sex_repulsion(means, counts, margin=0.75)
    assert float(near) > float(far)
    assert float(far) == 0.0


def test_confidence_pull_grows_for_small_cells() -> None:
    dim = 6
    means = to_hypersphere(torch.randn(2, 2, dim))
    big = torch.full((2, 2), 500.0)
    small = torch.full((2, 2), 2.0)
    assert float(_confidence(means, small, kappa=24.0)) > float(_confidence(means, big, kappa=24.0))


def test_contrastive_drops_as_embeddings_approach_target() -> None:
    dim = 16
    bank = to_hypersphere(torch.randn(4, dim))
    targets = torch.tensor([3, 0, 1, 2])
    verdict = torch.tensor([1, 0, 0, 1])
    sex = torch.tensor([1, 0, 1, 0])
    spec = FrankingSpec()
    state = FusionState(
        fused=torch.zeros(4, dim),
        weights=torch.ones(4, 3) / 3,
        null_embed=torch.zeros(N_MODALITIES, dim),
        marginal=torch.zeros(N_MODALITIES, dim),
    )
    far = to_hypersphere(-bank[targets] + 0.05 * torch.randn(4, dim))
    near = to_hypersphere(bank[targets] + 0.05 * torch.randn(4, dim))
    loss_far = evaluate(far, state, _toy_batch(far, verdict, sex), bank, targets, spec, tau=0.5)
    loss_near = evaluate(near, state, _toy_batch(near, verdict, sex), bank, targets, spec, tau=0.5)
    assert float(loss_near.contrastive) < float(loss_far.contrastive)


def test_distance_gradient_matches_finite_difference() -> None:
    torch.manual_seed(0)
    z = torch.randn(4, dtype=torch.float64, requires_grad=True)
    anchor = torch.randn(4, dtype=torch.float64)
    loss = torch.sum((z - anchor) ** 2)
    loss.backward()
    assert z.grad is not None
    analytic = z.grad.clone()
    eps = 1e-5
    numeric = torch.zeros(4, dtype=torch.float64)
    base = z.detach()
    for i in range(4):
        bump = base.clone()
        bump[i] += eps
        up = torch.sum((bump - anchor) ** 2)
        bump[i] -= 2 * eps
        down = torch.sum((bump - anchor) ** 2)
        numeric[i] = (up - down) / (2 * eps)
    assert torch.allclose(analytic, numeric, atol=1e-6)
