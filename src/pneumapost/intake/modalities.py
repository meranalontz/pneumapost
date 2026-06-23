from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, cast

import torch

from ..fittings.glossary import N_MODALITIES

LATENT = 24


@dataclass(frozen=True)
class Specimen:
    breath: torch.Tensor
    microbiome: torch.Tensor
    omics: torch.Tensor
    presence: torch.Tensor
    verdict: int
    sex: int
    never_smoker: bool
    driver_negative: bool
    arm: int
    site: int


class Batch(TypedDict):
    breath: torch.Tensor
    microbiome: torch.Tensor
    omics: torch.Tensor
    presence: torch.Tensor
    verdict: torch.Tensor
    sex: torch.Tensor
    never_smoker: torch.Tensor
    driver_negative: torch.Tensor
    arm: torch.Tensor
    site: torch.Tensor


def collate(specimens: list[Specimen]) -> Batch:
    return Batch(
        breath=torch.stack([s.breath for s in specimens]),
        microbiome=torch.stack([s.microbiome for s in specimens]),
        omics=torch.stack([s.omics for s in specimens]),
        presence=torch.stack([s.presence for s in specimens]),
        verdict=torch.tensor([s.verdict for s in specimens], dtype=torch.long),
        sex=torch.tensor([s.sex for s in specimens], dtype=torch.long),
        never_smoker=torch.tensor([s.never_smoker for s in specimens], dtype=torch.bool),
        driver_negative=torch.tensor([s.driver_negative for s in specimens], dtype=torch.bool),
        arm=torch.tensor([s.arm for s in specimens], dtype=torch.long),
        site=torch.tensor([s.site for s in specimens], dtype=torch.long),
    )


def to_device(batch: Batch, device: str) -> Batch:
    moved = {key: cast(torch.Tensor, value).to(device) for key, value in batch.items()}
    return cast(Batch, moved)


def empty_presence() -> torch.Tensor:
    return torch.zeros(N_MODALITIES, dtype=torch.bool)
