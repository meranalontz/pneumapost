from __future__ import annotations

import torch
import torch.nn as nn

from ..fittings.geometry import to_hypersphere
from ..fittings.glossary import N_SEXES, N_VERDICTS


def cell_means(
    z: torch.Tensor, verdict: torch.Tensor, sex: torch.Tensor, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.zeros(N_VERDICTS, N_SEXES, dim, device=z.device)
    counts = torch.zeros(N_VERDICTS, N_SEXES, device=z.device)
    for c in range(N_VERDICTS):
        for s in range(N_SEXES):
            picked = (verdict == c) & (sex == s)
            n = int(picked.sum())
            counts[c, s] = float(n)
            if n > 0:
                means[c, s] = z[picked].mean(dim=0)
    return means, counts


def shrinkage_omega(counts: torch.Tensor, kappa: float) -> torch.Tensor:
    return counts / (counts + kappa)


def sex_marginal(means: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    weight = counts.unsqueeze(-1).clamp_min(1e-6)
    return (means * weight).sum(dim=1) / weight.sum(dim=1)


def shrink_to_marginal(means: torch.Tensor, counts: torch.Tensor, kappa: float) -> torch.Tensor:
    omega = shrinkage_omega(counts, kappa).unsqueeze(-1)
    marginal = sex_marginal(means, counts).unsqueeze(1)
    blended = omega * means + (1.0 - omega) * marginal
    return to_hypersphere(blended)


class PrototypeBank(nn.Module):
    bank: torch.Tensor
    populations: torch.Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        start = to_hypersphere(torch.randn(N_VERDICTS, N_SEXES, dim))
        self.register_buffer("bank", start)
        self.register_buffer("populations", torch.zeros(N_VERDICTS, N_SEXES))

    def install(self, prototypes: torch.Tensor, populations: torch.Tensor) -> None:
        self.bank.copy_(to_hypersphere(prototypes))
        self.populations.copy_(populations)

    def flat(self) -> torch.Tensor:
        return self.bank.reshape(N_VERDICTS * N_SEXES, -1)
