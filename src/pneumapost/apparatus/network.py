from __future__ import annotations

import torch
import torch.nn as nn

from ..fittings.glossary import ApparatusSpec
from ..intake.modalities import Batch
from .breath_cnn import BreathTrace
from .fusion import FusionState, GatedNullFusion
from .microbiome_set import MicrobiomeSet
from .omics_gat import OmicsGraphAttention
from .projection import SphereProjection
from .prototypes import PrototypeBank


class PneumaNet(nn.Module):
    def __init__(self, spec: ApparatusSpec, omics_features: int) -> None:
        super().__init__()
        self.breath = BreathTrace(spec)
        self.microbiome = MicrobiomeSet(spec)
        self.omics = OmicsGraphAttention(spec, omics_features)
        self.fusion = GatedNullFusion(spec)
        self.projection = SphereProjection(spec)
        self.prototypes = PrototypeBank(spec.d_proj)

    def encode(self, batch: Batch) -> torch.Tensor:
        e_breath = self.breath(batch["breath"])
        e_micro = self.microbiome(batch["microbiome"])
        e_omics = self.omics(batch["omics"])
        return torch.stack([e_breath, e_micro, e_omics], dim=1)

    def forward(self, batch: Batch) -> tuple[torch.Tensor, FusionState]:
        stack = self.encode(batch)
        state = self.fusion(stack, batch["presence"])
        z = self.projection(state.fused)
        return z, state


def build_network(spec: ApparatusSpec, omics_features: int) -> PneumaNet:
    return PneumaNet(spec, omics_features)
