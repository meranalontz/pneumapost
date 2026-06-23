from __future__ import annotations

import torch
import torch.nn as nn

from ..fittings.geometry import to_hypersphere
from ..fittings.glossary import ApparatusSpec


class SphereProjection(nn.Module):
    def __init__(self, spec: ApparatusSpec) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(spec.d_enc, spec.d_enc),
            nn.GELU(),
            nn.LayerNorm(spec.d_enc),
            nn.Linear(spec.d_enc, spec.d_proj),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return to_hypersphere(self.net(fused))
