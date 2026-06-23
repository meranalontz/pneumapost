from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pyhocon import ConfigFactory, ConfigTree

from .glossary import (
    ApparatusSpec,
    DeliverySpec,
    DispatchSpec,
    FrankingSpec,
    IntakeSpec,
    RunSpec,
    SignalSpec,
)


def _tree(node: Any) -> Mapping[str, Any]:
    if isinstance(node, ConfigTree):
        return {k: node[k] for k in node}
    if isinstance(node, Mapping):
        return node
    return {}


def _ints(node: Any, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if node is None:
        return fallback
    return tuple(int(v) for v in node)


def _signal(node: Any) -> SignalSpec:
    d = _tree(node)
    base = SignalSpec()
    return SignalSpec(
        sex_axis_gap=float(d.get("sex_axis_gap", base.sex_axis_gap)),
        female_omics_coupling=float(d.get("female_omics_coupling", base.female_omics_coupling)),
        male_omics_coupling=float(d.get("male_omics_coupling", base.male_omics_coupling)),
        breath_ceiling=float(d.get("breath_ceiling", base.breath_ceiling)),
        never_smoker_lift=float(d.get("never_smoker_lift", base.never_smoker_lift)),
        driver_negative_share=float(d.get("driver_negative_share", base.driver_negative_share)),
        noise=float(d.get("noise", base.noise)),
    )


def _intake(node: Any) -> IntakeSpec:
    d = _tree(node)
    base = IntakeSpec()
    return IntakeSpec(
        n_sites=int(d.get("n_sites", base.n_sites)),
        prospective_female=_ints(d.get("prospective_female"), base.prospective_female),
        prospective_male=_ints(d.get("prospective_male"), base.prospective_male),
        retro_female=_ints(d.get("retro_female"), base.retro_female),
        retro_male=_ints(d.get("retro_male"), base.retro_male),
        luad_prevalence=float(d.get("luad_prevalence", base.luad_prevalence)),
        never_smoker_share=float(d.get("never_smoker_share", base.never_smoker_share)),
        omics_dim=int(d.get("omics_dim", base.omics_dim)),
        n_taxa=int(d.get("n_taxa", base.n_taxa)),
        taxa_set_size=int(d.get("taxa_set_size", base.taxa_set_size)),
        breath_len=int(d.get("breath_len", base.breath_len)),
        missing_breath=float(d.get("missing_breath", base.missing_breath)),
        missing_microbiome=float(d.get("missing_microbiome", base.missing_microbiome)),
        missing_omics=float(d.get("missing_omics", base.missing_omics)),
        signal=_signal(d.get("signal")),
    )


def _apparatus(node: Any) -> ApparatusSpec:
    d = _tree(node)
    base = ApparatusSpec()
    return ApparatusSpec(
        d_enc=int(d.get("d_enc", base.d_enc)),
        d_proj=int(d.get("d_proj", base.d_proj)),
        gat_hidden=int(d.get("gat_hidden", base.gat_hidden)),
        gat_layers=int(d.get("gat_layers", base.gat_layers)),
        gat_heads=int(d.get("gat_heads", base.gat_heads)),
        gat_graph_k=int(d.get("gat_graph_k", base.gat_graph_k)),
        set_hidden=int(d.get("set_hidden", base.set_hidden)),
        set_inducing=int(d.get("set_inducing", base.set_inducing)),
        set_heads=int(d.get("set_heads", base.set_heads)),
        cnn_channels=_ints(d.get("cnn_channels"), base.cnn_channels),
        cnn_kernel=int(d.get("cnn_kernel", base.cnn_kernel)),
        fusion_heads=int(d.get("fusion_heads", base.fusion_heads)),
        fusion_depth=int(d.get("fusion_depth", base.fusion_depth)),
        tau=float(d.get("tau", base.tau)),
        transfer_gate_beta=float(d.get("transfer_gate_beta", base.transfer_gate_beta)),
    )


def _franking(node: Any) -> FrankingSpec:
    d = _tree(node)
    base = FrankingSpec()
    return FrankingSpec(
        gamma=float(d.get("gamma", base.gamma)),
        lam=float(d.get("lam", base.lam)),
        beta=float(d.get("beta", base.beta)),
        sex_margin=float(d.get("sex_margin", base.sex_margin)),
        kappa=float(d.get("kappa", base.kappa)),
        mu_prox=float(d.get("mu_prox", base.mu_prox)),
    )


def _dispatch(node: Any) -> DispatchSpec:
    d = _tree(node)
    base = DispatchSpec()
    return DispatchSpec(
        strategy=str(d.get("strategy", base.strategy)),
        rounds=int(d.get("rounds", base.rounds)),
        local_epochs=int(d.get("local_epochs", base.local_epochs)),
        batch_size=int(d.get("batch_size", base.batch_size)),
        grad_accum=int(d.get("grad_accum", base.grad_accum)),
        world_size=int(d.get("world_size", base.world_size)),
        lr=float(d.get("lr", base.lr)),
        warmup_steps=int(d.get("warmup_steps", base.warmup_steps)),
        weight_decay=float(d.get("weight_decay", base.weight_decay)),
        grad_clip=float(d.get("grad_clip", base.grad_clip)),
        precision=str(d.get("precision", base.precision)),
        seeds=int(d.get("seeds", base.seeds)),
        budget_bytes=int(d.get("budget_bytes", base.budget_bytes)),
    )


def _delivery(node: Any) -> DeliverySpec:
    d = _tree(node)
    base = DeliverySpec()
    return DeliverySpec(
        target_sensitivity=float(d.get("target_sensitivity", base.target_sensitivity)),
        temperature=float(d.get("temperature", base.temperature)),
        bootstrap=int(d.get("bootstrap", base.bootstrap)),
        fdr_alpha=float(d.get("fdr_alpha", base.fdr_alpha)),
    )


def spec_from_tree(tree: Mapping[str, Any]) -> RunSpec:
    return RunSpec(
        seed=int(tree.get("seed", 0)),
        device=str(tree.get("device", "cpu")),
        intake=_intake(tree.get("intake")),
        apparatus=_apparatus(tree.get("apparatus")),
        franking=_franking(tree.get("franking")),
        dispatch=_dispatch(tree.get("dispatch")),
        delivery=_delivery(tree.get("delivery")),
    )


def load_circular(path: str | Path, overrides: tuple[str, ...] = ()) -> RunSpec:
    tree = ConfigFactory.parse_file(str(Path(path)))
    for item in overrides:
        tree = ConfigFactory.parse_string(item).with_fallback(tree)
    return spec_from_tree(_tree(tree))
