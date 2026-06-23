from __future__ import annotations

import torch

from ..apparatus.network import PneumaNet
from ..fittings.geometry import squared_distance
from ..fittings.glossary import Sex, Verdict
from ..intake.modalities import Batch


def sex_matched_probability(
    z: torch.Tensor, bank: torch.Tensor, sex: torch.Tensor, tau: float
) -> torch.Tensor:
    benign = bank[int(Verdict.BENIGN), sex]
    luad = bank[int(Verdict.LUAD), sex]
    d_benign = squared_distance(z, benign)
    d_luad = squared_distance(z, luad)
    logits = torch.stack([-d_benign / tau, -d_luad / tau], dim=1)
    return torch.softmax(logits, dim=1)[:, 1]


def infer(net: PneumaNet, batch: Batch, tau: float) -> torch.Tensor:
    net.eval()
    with torch.no_grad():
        z, _ = net(batch)
        return sex_matched_probability(z, net.prototypes.bank, batch["sex"], tau)


def decide(probability: torch.Tensor, threshold: float) -> torch.Tensor:
    return probability >= threshold


def default_sex(n: int) -> torch.Tensor:
    return torch.full((n,), int(Sex.FEMALE), dtype=torch.long)
