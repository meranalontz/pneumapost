from __future__ import annotations

from typing import Sequence

import torch

from ..apparatus.prototypes import sex_marginal, shrinkage_omega
from ..fittings.geometry import to_hypersphere
from ..fittings.glossary import N_SEXES, N_VERDICTS


def _quantize(prototypes: torch.Tensor, budget_bytes: int) -> torch.Tensor:
    elements = prototypes.numel()
    bytes_per = budget_bytes / max(elements, 1)
    if bytes_per >= 4:
        return prototypes
    quantized = prototypes.to(torch.bfloat16).to(prototypes.dtype)
    return quantized


def confidence_weighted(
    local_means: list[torch.Tensor],
    local_counts: list[torch.Tensor],
    kappa: float,
    budget_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = local_means[0].size(-1)
    acc = torch.zeros(N_VERDICTS, N_SEXES, dim)
    weight = torch.zeros(N_VERDICTS, N_SEXES)
    populations = torch.zeros(N_VERDICTS, N_SEXES)
    for means, counts in zip(local_means, local_counts):
        omega = shrinkage_omega(counts, kappa)
        w = (omega * counts).unsqueeze(-1)
        acc += w * means
        weight += omega * counts
        populations += counts
    pooled = acc / weight.unsqueeze(-1).clamp_min(1e-6)
    global_omega = shrinkage_omega(populations, kappa).unsqueeze(-1)
    marginal = sex_marginal(pooled, populations).unsqueeze(1)
    shrunk = global_omega * pooled + (1.0 - global_omega) * marginal
    shrunk = _quantize(shrunk, budget_bytes)
    return to_hypersphere(shrunk), populations


def proximal_term(
    params: Sequence[torch.Tensor], anchors: Sequence[torch.Tensor], mu: float
) -> torch.Tensor:
    total = params[0].new_zeros(())
    for p, a in zip(params, anchors):
        total = total + torch.sum((p - a) ** 2)
    return 0.5 * mu * total
