from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fittings.glossary import N_MODALITIES, ApparatusSpec, Modality


@dataclass
class FusionState:
    fused: torch.Tensor
    weights: torch.Tensor
    null_embed: torch.Tensor
    marginal: torch.Tensor


class GatedNullFusion(nn.Module):
    marginal: torch.Tensor
    seen: torch.Tensor

    def __init__(self, spec: ApparatusSpec) -> None:
        super().__init__()
        dim = spec.d_enc
        self.null_embed = nn.Parameter(torch.randn(N_MODALITIES, dim) * 0.02)
        self.modality_pos = nn.Parameter(torch.randn(N_MODALITIES, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, spec.fusion_heads, dim * 2, batch_first=True, activation="gelu"
        )
        self.mixer = nn.TransformerEncoder(layer, spec.fusion_depth)
        self.gate = nn.Linear(dim, 1)
        self.presence_bias = nn.Parameter(torch.tensor(2.0))
        self.transfer_beta = spec.transfer_gate_beta
        self.register_buffer("marginal", torch.zeros(N_MODALITIES, dim))
        self.register_buffer("seen", torch.zeros(N_MODALITIES))

    def _update_marginal(self, embeddings: torch.Tensor, presence: torch.Tensor) -> None:
        with torch.no_grad():
            for m in range(N_MODALITIES):
                picked = presence[:, m]
                if bool(picked.any()):
                    mean = embeddings[picked, m].mean(dim=0)
                    self.marginal[m].mul_(0.99).add_(mean, alpha=0.01)
                    self.seen[m] += 1.0

    def _transfer_gate(self, embeddings: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        breath = embeddings[:, int(Modality.BREATH)]
        others = presence.clone()
        others[:, int(Modality.BREATH)] = False
        weight = others.float().unsqueeze(-1)
        denom = weight.sum(dim=1).clamp_min(1.0)
        context = (embeddings * weight).sum(dim=1) / denom
        consistency = F.cosine_similarity(breath, context, dim=-1)
        consistency = torch.where(others.any(dim=1), consistency, torch.zeros_like(consistency))
        return torch.sigmoid(self.transfer_beta * consistency)

    def forward(self, embeddings: torch.Tensor, presence: torch.Tensor) -> FusionState:
        if self.training:
            self._update_marginal(embeddings, presence)
        null = self.null_embed.unsqueeze(0).expand(embeddings.size(0), -1, -1)
        slots = torch.where(presence.unsqueeze(-1), embeddings, null)
        mixed = self.mixer(slots + self.modality_pos.unsqueeze(0))
        logits = self.gate(mixed).squeeze(-1) + self.presence_bias * presence.float()
        breath_idx = int(Modality.BREATH)
        a_breath = self._transfer_gate(embeddings, presence)
        gated = torch.log(a_breath.clamp_min(1e-6))
        logits = logits.clone()
        logits[:, breath_idx] = logits[:, breath_idx] + gated
        weights = F.softmax(logits, dim=1)
        fused = torch.sum(weights.unsqueeze(-1) * mixed, dim=1)
        return FusionState(
            fused=fused, weights=weights, null_embed=self.null_embed, marginal=self.marginal
        )
