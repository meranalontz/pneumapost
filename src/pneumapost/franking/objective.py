from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..apparatus.fusion import FusionState
from ..apparatus.prototypes import cell_means, sex_marginal, shrinkage_omega
from ..fittings.geometry import pairwise_squared
from ..fittings.glossary import N_SEXES, N_VERDICTS, FrankingSpec, Sex
from ..intake.modalities import Batch


@dataclass
class FrankingTerms:
    contrastive: torch.Tensor
    sex_repulsion: torch.Tensor
    null_anchor: torch.Tensor
    confidence: torch.Tensor
    total: torch.Tensor


def _sex_repulsion(means: torch.Tensor, counts: torch.Tensor, margin: float) -> torch.Tensor:
    total = means.new_zeros(())
    for c in range(N_VERDICTS):
        if counts[c, int(Sex.FEMALE)] > 0 and counts[c, int(Sex.MALE)] > 0:
            gap = torch.linalg.vector_norm(means[c, int(Sex.FEMALE)] - means[c, int(Sex.MALE)])
            total = total + F.relu(margin - gap)
    return total


def _confidence(means: torch.Tensor, counts: torch.Tensor, kappa: float) -> torch.Tensor:
    omega = shrinkage_omega(counts, kappa)
    marginal = sex_marginal(means, counts)
    total = means.new_zeros(())
    for c in range(N_VERDICTS):
        for s in range(N_SEXES):
            if counts[c, s] > 0:
                pull = torch.sum((means[c, s] - marginal[c]) ** 2)
                total = total + (1.0 - omega[c, s]) * pull
    return total


def evaluate(
    z: torch.Tensor,
    state: FusionState,
    batch: Batch,
    bank_flat: torch.Tensor,
    cell_targets: torch.Tensor,
    spec: FrankingSpec,
    tau: float,
) -> FrankingTerms:
    logits = -pairwise_squared(z, bank_flat) / tau
    contrastive = F.cross_entropy(logits, cell_targets)
    means, counts = cell_means(z, batch["verdict"], batch["sex"], z.size(-1))
    sex_repulsion = _sex_repulsion(means, counts, spec.sex_margin)
    null_anchor = torch.sum((state.null_embed - state.marginal.detach()) ** 2)
    confidence = _confidence(means, counts, spec.kappa)
    total = (
        contrastive + spec.gamma * sex_repulsion + spec.lam * null_anchor + spec.beta * confidence
    )
    return FrankingTerms(
        contrastive=contrastive,
        sex_repulsion=sex_repulsion,
        null_anchor=null_anchor,
        confidence=confidence,
        total=total,
    )
