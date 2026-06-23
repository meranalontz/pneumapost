from __future__ import annotations

import dataclasses

import torch

from pneumapost.apparatus.network import build_network
from pneumapost.fittings.geometry import to_hypersphere
from pneumapost.fittings.glossary import N_SEXES, ApparatusSpec, FrankingSpec, IntakeSpec
from pneumapost.franking.objective import evaluate
from pneumapost.intake.cohort import build_cohort
from pneumapost.intake.modalities import collate

_MINI_APPARATUS = ApparatusSpec(
    d_enc=64,
    d_proj=32,
    gat_hidden=64,
    gat_layers=2,
    gat_heads=4,
    set_hidden=64,
    set_inducing=8,
    set_heads=4,
    cnn_channels=(16, 32),
    fusion_heads=4,
    fusion_depth=1,
)


def test_single_batch_contrastive_overfits() -> None:
    torch.manual_seed(0)
    intake = dataclasses.replace(
        IntakeSpec(),
        prospective_female=(24, 0, 0, 0),
        prospective_male=(8, 0, 0, 0),
        retro_female=(0, 0, 0, 0),
        retro_male=(0, 0, 0, 0),
        omics_dim=64,
        breath_len=64,
        taxa_set_size=16,
        missing_breath=0.0,
        missing_microbiome=0.0,
        missing_omics=0.0,
    )
    net = build_network(_MINI_APPARATUS, intake.omics_dim)
    batch = collate(build_cohort(intake, seed=0)[0])
    targets = batch["verdict"] * N_SEXES + batch["sex"]

    anchors = to_hypersphere(torch.randn(2, N_SEXES, _MINI_APPARATUS.d_proj))
    net.prototypes.install(anchors, torch.full((2, N_SEXES), 100.0))
    bank = net.prototypes.flat().detach()

    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first = None
    last = 0.0
    for step in range(160):
        z, state = net(batch)
        terms = evaluate(z, state, batch, bank, targets, FrankingSpec(), tau=0.5)
        opt.zero_grad(set_to_none=True)
        terms.contrastive.backward()
        opt.step()
        value = float(terms.contrastive.detach())
        if step == 0:
            first = value
        last = value
    assert first is not None
    assert last < 0.3 * first
