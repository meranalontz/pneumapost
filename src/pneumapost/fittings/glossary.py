from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, TypedDict

import torch


class Modality(IntEnum):
    BREATH = 0
    MICROBIOME = 1
    OMICS = 2


class Sex(IntEnum):
    FEMALE = 0
    MALE = 1


class Verdict(IntEnum):
    BENIGN = 0
    LUAD = 1


MODALITIES: tuple[Modality, ...] = (Modality.BREATH, Modality.MICROBIOME, Modality.OMICS)
SEXES: tuple[Sex, ...] = (Sex.FEMALE, Sex.MALE)
VERDICTS: tuple[Verdict, ...] = (Verdict.BENIGN, Verdict.LUAD)
N_MODALITIES = len(MODALITIES)
N_SEXES = len(SEXES)
N_VERDICTS = len(VERDICTS)


class Carrier(TypedDict):
    prototypes: torch.Tensor
    omegas: torch.Tensor
    counts: torch.Tensor
    site: int
    payload_bytes: int


@dataclass(frozen=True)
class SignalSpec:
    sex_axis_gap: float = 1.35
    female_omics_coupling: float = 0.61
    male_omics_coupling: float = 0.06
    breath_ceiling: float = 0.80
    never_smoker_lift: float = 0.18
    driver_negative_share: float = 0.46
    noise: float = 1.0


@dataclass(frozen=True)
class IntakeSpec:
    n_sites: int = 4
    prospective_female: tuple[int, ...] = (742, 638, 521, 417)
    prospective_male: tuple[int, ...] = (118, 104, 92, 78)
    retro_female: tuple[int, ...] = (651, 564, 489, 339)
    retro_male: tuple[int, ...] = (96, 84, 76, 62)
    luad_prevalence: float = 0.5
    never_smoker_share: float = 0.64
    omics_dim: int = 256
    n_taxa: int = 320
    taxa_set_size: int = 96
    breath_len: int = 512
    missing_breath: float = 0.18
    missing_microbiome: float = 0.34
    missing_omics: float = 0.41
    signal: SignalSpec = field(default_factory=SignalSpec)


@dataclass(frozen=True)
class ApparatusSpec:
    d_enc: int = 256
    d_proj: int = 128
    gat_hidden: int = 260
    gat_layers: int = 3
    gat_heads: int = 4
    gat_graph_k: int = 12
    set_hidden: int = 208
    set_inducing: int = 24
    set_heads: int = 4
    cnn_channels: tuple[int, ...] = (48, 96, 160)
    cnn_kernel: int = 7
    fusion_heads: int = 4
    fusion_depth: int = 2
    tau: float = 0.5
    transfer_gate_beta: float = 4.0


@dataclass(frozen=True)
class FrankingSpec:
    gamma: float = 0.5
    lam: float = 0.1
    beta: float = 0.2
    sex_margin: float = 0.75
    kappa: float = 24.0
    mu_prox: float = 0.05


@dataclass(frozen=True)
class DispatchSpec:
    strategy: str = "sexcond_fedprotocon"
    rounds: int = 20
    local_epochs: int = 400
    batch_size: int = 512
    grad_accum: int = 4
    world_size: int = 8
    lr: float = 5e-7
    warmup_steps: int = 5000
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    precision: str = "fp32"
    seeds: int = 20
    budget_bytes: int = 1_887_436


@dataclass(frozen=True)
class DeliverySpec:
    target_sensitivity: float = 0.95
    temperature: float = 0.5
    bootstrap: int = 2000
    fdr_alpha: float = 0.10


@dataclass(frozen=True)
class RunSpec:
    seed: int = 0
    device: str = "cpu"
    intake: IntakeSpec = field(default_factory=IntakeSpec)
    apparatus: ApparatusSpec = field(default_factory=ApparatusSpec)
    franking: FrankingSpec = field(default_factory=FrankingSpec)
    dispatch: DispatchSpec = field(default_factory=DispatchSpec)
    delivery: DeliverySpec = field(default_factory=DeliverySpec)


def cell_index(verdict: Verdict, sex: Sex) -> int:
    return int(verdict) * N_SEXES + int(sex)


def effective_batch(spec: DispatchSpec) -> int:
    return spec.batch_size * spec.grad_accum * spec.world_size


def with_overrides(spec: RunSpec, **changes: Any) -> RunSpec:
    return replace(spec, **changes)
