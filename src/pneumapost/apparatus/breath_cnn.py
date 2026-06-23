from __future__ import annotations

import torch
import torch.nn as nn

from ..fittings.glossary import ApparatusSpec


class BreathTrace(nn.Module):
    def __init__(self, spec: ApparatusSpec) -> None:
        super().__init__()
        chans = (1,) + spec.cnn_channels
        pad = spec.cnn_kernel // 2
        blocks: list[nn.Module] = []
        for c_in, c_out in zip(chans[:-1], chans[1:]):
            blocks.append(nn.Conv1d(c_in, c_out, spec.cnn_kernel, padding=pad))
            blocks.append(nn.BatchNorm1d(c_out))
            blocks.append(nn.GELU())
            blocks.append(nn.MaxPool1d(2))
        self.stem = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(spec.cnn_channels[-1], spec.d_enc),
            nn.GELU(),
            nn.Linear(spec.d_enc, spec.d_enc),
        )

    def forward(self, breath: torch.Tensor) -> torch.Tensor:
        h = self.stem(breath.unsqueeze(1))
        h = self.pool(h).squeeze(-1)
        return self.head(h)
