from __future__ import annotations

from ..dispatch.checkpoint import save_checkpoint
from ..dispatch.trainer import FederatedTrainer
from ..fittings.clerking import get_logger
from ..fittings.glossary import effective_batch, with_overrides
from ..fittings.hocon_io import load_circular

_log = get_logger("wicket.train")


def train(
    config: str, *override: str, device: str = "cpu", out: str = "runs/pneumapost.pt"
) -> None:
    spec = with_overrides(load_circular(config, override), device=device)
    _log.info(
        "rounds=%d local_epochs=%d effective_batch=%d lr=%.2e world_size=%d",
        spec.dispatch.rounds,
        spec.dispatch.local_epochs,
        effective_batch(spec.dispatch),
        spec.dispatch.lr,
        spec.dispatch.world_size,
    )
    trainer = FederatedTrainer(spec)
    history = trainer.run()
    save_checkpoint(out, trainer.net.state_dict(), spec, trainer.last_step())
    _log.info("final round loss %.4f -> %s", history.round_loss[-1], out)
