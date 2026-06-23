from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fittings.glossary import ApparatusSpec


def _knn_adjacency(n_nodes: int, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    template = torch.randn(n_nodes, 16, generator=gen)
    dist = torch.cdist(template, template)
    dist.fill_diagonal_(float("inf"))
    neighbors = torch.topk(dist, k, dim=1, largest=False).indices
    adj = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
    rows = torch.arange(n_nodes).unsqueeze(1).expand(-1, k)
    adj[rows.reshape(-1), neighbors.reshape(-1)] = True
    adj = adj | adj.t()
    adj.fill_diagonal_(True)
    return adj


class GraphAttentionLayer(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.merge = nn.Linear(dim, dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        pooled = torch.matmul(weights, v).transpose(1, 2).reshape(b, n, -1)
        h = self.norm_a(x + self.merge(pooled))
        return self.norm_b(h + self.ff(h))


class OmicsGraphAttention(nn.Module):
    adjacency: torch.Tensor

    def __init__(self, spec: ApparatusSpec, n_features: int) -> None:
        super().__init__()
        dim = spec.gat_hidden
        self.node_embed = nn.Parameter(torch.randn(n_features, dim) * 0.02)
        self.node_scale = nn.Parameter(torch.randn(n_features, dim) * 0.02)
        self.layers = nn.ModuleList(
            [GraphAttentionLayer(dim, spec.gat_heads) for _ in range(spec.gat_layers)]
        )
        self.readout_query = nn.Parameter(torch.randn(1, 1, dim) * 0.1)
        self.readout = nn.MultiheadAttention(dim, spec.gat_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(dim, spec.d_enc), nn.GELU(), nn.Linear(spec.d_enc, spec.d_enc)
        )
        adj = _knn_adjacency(n_features, spec.gat_graph_k, seed=11)
        self.register_buffer("adjacency", adj.view(1, 1, n_features, n_features))

    def forward(self, omics: torch.Tensor) -> torch.Tensor:
        h = self.node_embed.unsqueeze(0) + omics.unsqueeze(-1) * self.node_scale.unsqueeze(0)
        for layer in self.layers:
            h = layer(h, self.adjacency)
        query = self.readout_query.expand(h.size(0), -1, -1)
        pooled, _ = self.readout(query, h, h, need_weights=False)
        return self.head(pooled.squeeze(1))
