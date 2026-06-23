from __future__ import annotations

import torch

from pneumapost.apparatus.network import build_network
from pneumapost.fittings.glossary import N_MODALITIES, ApparatusSpec, IntakeSpec
from pneumapost.intake.cohort import build_cohort
from pneumapost.intake.modalities import collate


def _count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_encoder_and_projection_shapes() -> None:
    spec = ApparatusSpec()
    isp = IntakeSpec()
    net = build_network(spec, isp.omics_dim).eval()
    batch = collate(build_cohort(isp, 0)[0][:5])
    stack = net.encode(batch)
    assert stack.shape == (5, N_MODALITIES, spec.d_enc)
    z, state = net(batch)
    assert z.shape == (5, spec.d_proj)
    assert torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(state.weights.sum(dim=1), torch.ones(5), atol=1e-5)


def test_branch_parameter_budgets() -> None:
    net = build_network(ApparatusSpec(), IntakeSpec().omics_dim)
    omics = _count(net.omics) / 1e6
    micro = _count(net.microbiome) / 1e6
    assert 2.7 <= omics <= 3.3
    assert 1.7 <= micro <= 2.1
    assert _count(net.breath) < _count(net.microbiome)
