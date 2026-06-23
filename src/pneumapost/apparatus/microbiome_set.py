from __future__ import annotations

import torch
import torch.nn as nn

from ..fittings.glossary import ApparatusSpec
from ..intake.modalities import LATENT


class MultiheadBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(query, key, key, need_weights=False)
        h = self.norm_a(query + attended)
        return self.norm_b(h + self.ff(h))


class InducedBlock(nn.Module):
    def __init__(self, dim: int, heads: int, inducing: int) -> None:
        super().__init__()
        self.points = nn.Parameter(torch.randn(1, inducing, dim) * 0.1)
        self.project = MultiheadBlock(dim, heads)
        self.distribute = MultiheadBlock(dim, heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchors = self.points.expand(x.size(0), -1, -1)
        summary = self.project(anchors, x)
        return self.distribute(x, summary)


class Pooling(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.1)
        self.block = MultiheadBlock(dim, heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.seed.expand(x.size(0), -1, -1)
        return self.block(query, x).squeeze(1)


class MicrobiomeSet(nn.Module):
    def __init__(self, spec: ApparatusSpec) -> None:
        super().__init__()
        dim = spec.set_hidden
        self.embed = nn.Sequential(nn.Linear(LATENT, dim), nn.GELU(), nn.Linear(dim, dim))
        self.encoder = nn.ModuleList(
            [InducedBlock(dim, spec.set_heads, spec.set_inducing) for _ in range(2)]
        )
        self.pool = Pooling(dim, spec.set_heads)
        self.head = nn.Sequential(
            nn.Linear(dim, spec.d_enc), nn.GELU(), nn.Linear(spec.d_enc, spec.d_enc)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for block in self.encoder:
            h = block(h)
        return self.head(self.pool(h))
