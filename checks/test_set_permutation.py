from __future__ import annotations

import torch

from pneumapost.apparatus.microbiome_set import MicrobiomeSet
from pneumapost.apparatus.network import build_network
from pneumapost.fittings.glossary import ApparatusSpec, IntakeSpec, Modality
from pneumapost.intake.cohort import build_cohort
from pneumapost.intake.modalities import LATENT, collate


def test_microbiome_encoder_is_permutation_invariant() -> None:
    torch.manual_seed(0)
    encoder = MicrobiomeSet(ApparatusSpec()).eval()
    tokens = torch.randn(2, 96, LATENT)
    perm = torch.randperm(96)
    with torch.no_grad():
        base = encoder(tokens)
        shuffled = encoder(tokens[:, perm, :])
    assert torch.allclose(base, shuffled, atol=1e-5)


def test_fusion_ignores_absent_modality_payload() -> None:
    torch.manual_seed(1)
    isp = IntakeSpec()
    net = build_network(ApparatusSpec(), isp.omics_dim).eval()
    cohort = build_cohort(isp, seed=0)[0][:6]
    batch = collate(cohort)
    batch["presence"][:, int(Modality.OMICS)] = False
    with torch.no_grad():
        z_a, _ = net(batch)
        corrupted = dict(batch)
        corrupted["omics"] = batch["omics"] + 50.0 * torch.randn_like(batch["omics"])
        z_b, _ = net(corrupted)  # type: ignore[arg-type]
    assert torch.allclose(z_a, z_b, atol=1e-5)
