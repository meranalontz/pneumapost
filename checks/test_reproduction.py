from __future__ import annotations

from pneumapost.delivery.harness import never_smoker_card, table_one
from pneumapost.dispatch.strategies import get_strategy
from pneumapost.fittings.glossary import RunSpec

_FEDERATED = ("fedavg", "fedprox", "fedbn", "creamfl", "fedproto", "cmpfl", "fedkim", "dflmp")


def test_headline_prospective_auroc(main_spec: RunSpec) -> None:
    table = table_one(main_spec, main_spec.seed)
    proposed = table["sexcond_fedprotocon"]["prospective"].auroc
    assert 0.881 <= proposed <= 0.905


def test_never_smoker_subgroup(main_spec: RunSpec) -> None:
    card = never_smoker_card(main_spec, main_spec.seed)
    assert 0.885 <= card.auroc <= 0.915


def test_proposed_beats_every_federated_baseline(main_spec: RunSpec) -> None:
    table = table_one(main_spec, main_spec.seed)
    proposed = table["sexcond_fedprotocon"]["prospective"].auroc
    best_baseline = max(table[k]["prospective"].auroc for k in _FEDERATED)
    assert proposed - best_baseline >= 0.025


def test_oracle_is_upper_bound(main_spec: RunSpec) -> None:
    table = table_one(main_spec, main_spec.seed)
    proposed = table["sexcond_fedprotocon"]["prospective"].auroc
    oracle = table["centralized_oracle"]["prospective"].auroc
    assert oracle >= proposed - 1e-6


def test_retrospective_arm_supports_prospective(main_spec: RunSpec) -> None:
    table = table_one(main_spec, main_spec.seed)
    retro = table["sexcond_fedprotocon"]["retrospective"].auroc
    assert 0.895 <= retro <= 0.925


def test_unimodal_below_multimodal(main_spec: RunSpec) -> None:
    table = table_one(main_spec, main_spec.seed)
    proposed = table["sexcond_fedprotocon"]["prospective"].auroc
    for key in ("breath_only", "s16_only", "omics_only"):
        assert table[key]["prospective"].auroc < proposed
    assert get_strategy("sexcond_fedprotocon").sex_conditional
