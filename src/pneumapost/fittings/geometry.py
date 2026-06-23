from __future__ import annotations

import torch
import torch.nn.functional as F


def to_hypersphere(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2.0, dim=-1, eps=eps)


def squared_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum((a - b) ** 2, dim=-1)


def pairwise_squared(z: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    diff = z.unsqueeze(1) - anchors.unsqueeze(0)
    return torch.sum(diff**2, dim=-1)


def chordal_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(a - b, dim=-1)
