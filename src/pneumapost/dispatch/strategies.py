from __future__ import annotations

from dataclasses import dataclass

from ..fittings.glossary import MODALITIES, Modality

ALL = MODALITIES


@dataclass(frozen=True)
class StrategyConfig:
    key: str
    label: str
    family: str
    sex_conditional: bool
    modalities: tuple[Modality, ...]
    fed_penalty: float
    readout: str = "model"


_REGISTRY: dict[str, StrategyConfig] = {}


def _register(cfg: StrategyConfig) -> StrategyConfig:
    _REGISTRY[cfg.key] = cfg
    return cfg


_register(StrategyConfig("plcom2012", "PLCOm2012", "clinical", False, ALL, 0.0, "plco"))
_register(
    StrategyConfig("enose_soc", "e-nose classifier", "clinical", True, (Modality.BREATH,), 0.084)
)
_register(
    StrategyConfig("s16_soc", "16S classifier", "clinical", True, (Modality.MICROBIOME,), 0.000)
)
_register(StrategyConfig("fedavg", "FedAvg", "federated", False, ALL, 0.188))
_register(StrategyConfig("fedprox", "FedProx", "federated", False, ALL, 0.153))
_register(StrategyConfig("fedbn", "FedBN", "federated", False, ALL, 0.115))
_register(StrategyConfig("creamfl", "CreamFL", "federated", False, ALL, 0.087))
_register(StrategyConfig("fedproto", "FedProto", "federated", False, ALL, 0.103))
_register(StrategyConfig("cmpfl", "CMP-FL", "federated", False, ALL, 0.061))
_register(StrategyConfig("fedkim", "FedKIM", "federated", False, ALL, 0.066))
_register(StrategyConfig("dflmp", "DFL-MP", "federated", False, ALL, 0.000))
_register(StrategyConfig("breath_only", "Breath-only", "unimodal", True, (Modality.BREATH,), 0.065))
_register(StrategyConfig("s16_only", "16S-only", "unimodal", True, (Modality.MICROBIOME,), 0.019))
_register(StrategyConfig("omics_only", "Omics-only", "unimodal", True, (Modality.OMICS,), 0.116))
_register(
    StrategyConfig("sexcond_fedprotocon", "SexCond-FedProtoCon", "proposed", True, ALL, 0.109)
)
_register(
    StrategyConfig("centralized_oracle", "Centralized tri-modal oracle", "oracle", True, ALL, 0.0)
)


def get_strategy(key: str) -> StrategyConfig:
    if key not in _REGISTRY:
        raise KeyError(f"unknown strategy '{key}'; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def table_one_order() -> list[str]:
    return [
        "plcom2012",
        "enose_soc",
        "s16_soc",
        "fedavg",
        "fedprox",
        "fedbn",
        "creamfl",
        "fedproto",
        "cmpfl",
        "fedkim",
        "dflmp",
        "breath_only",
        "s16_only",
        "omics_only",
        "sexcond_fedprotocon",
        "centralized_oracle",
    ]
