from __future__ import annotations

import dataclasses

import torch

from pneumapost.dispatch.trainer import FederatedTrainer
from pneumapost.fittings.glossary import RunSpec


def test_one_federated_round_updates_bank(tiny_spec: RunSpec) -> None:
    dispatch = dataclasses.replace(
        tiny_spec.dispatch,
        rounds=1,
        local_epochs=1,
        batch_size=32,
        grad_accum=1,
        world_size=1,
        warmup_steps=1,
        lr=1e-3,
    )
    spec = dataclasses.replace(tiny_spec, dispatch=dispatch)
    trainer = FederatedTrainer(spec)
    before = trainer.net.prototypes.bank.clone()
    history = trainer.run()

    assert len(history.round_loss) == 1
    assert trainer.last_step() > 0
    bank = trainer.net.prototypes.bank
    assert torch.isfinite(bank).all()
    norms = bank.reshape(-1, bank.size(-1)).norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    assert not torch.allclose(before, bank)
