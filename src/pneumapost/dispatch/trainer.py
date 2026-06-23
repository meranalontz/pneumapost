from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from ..apparatus.network import PneumaNet, build_network
from ..apparatus.prototypes import cell_means
from ..fittings.clerking import get_logger
from ..fittings.glossary import N_SEXES, RunSpec
from ..fittings.seed import set_seed
from ..franking.objective import evaluate
from ..intake.loaders import site_loaders
from ..intake.modalities import Batch, Specimen, to_device
from .aggregate import confidence_weighted, proximal_term
from .schedule import warmup_cosine

_log = get_logger("dispatch.trainer")


@dataclass
class History:
    round_loss: list[float] = field(default_factory=list)
    contrastive: list[float] = field(default_factory=list)
    sex_repulsion: list[float] = field(default_factory=list)


def _cell_targets(batch: Batch) -> torch.Tensor:
    return batch["verdict"] * N_SEXES + batch["sex"]


class FederatedTrainer:
    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self.device = spec.device
        set_seed(spec.seed)
        self.net: PneumaNet = build_network(spec.apparatus, spec.intake.omics_dim).to(self.device)
        self.loaders, self.prospective, self.retro = site_loaders(
            spec.intake, spec.dispatch.batch_size, spec.seed
        )
        self.opt = torch.optim.AdamW(
            self.net.parameters(), lr=spec.dispatch.lr, weight_decay=spec.dispatch.weight_decay
        )
        steps = max(spec.dispatch.rounds * len(self.loaders) * spec.dispatch.local_epochs, 1)
        self._total_steps = steps
        self._step = 0

    def last_step(self) -> int:
        return self._step

    def _emit_prototypes(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self.net.eval()
        means: list[torch.Tensor] = []
        counts: list[torch.Tensor] = []
        with torch.no_grad():
            for loader in self.loaders:
                zs: list[torch.Tensor] = []
                vs: list[torch.Tensor] = []
                ss: list[torch.Tensor] = []
                for batch in loader:
                    moved = to_device(batch, self.device)
                    z, _ = self.net(moved)
                    zs.append(z)
                    vs.append(moved["verdict"])
                    ss.append(moved["sex"])
                z_all = torch.cat(zs)
                m, c = cell_means(z_all, torch.cat(vs), torch.cat(ss), z_all.size(-1))
                means.append(m)
                counts.append(c)
        return means, counts

    def _local_update(self, loader: DataLoader[Specimen], anchors: list[torch.Tensor]) -> None:
        self.net.train()
        bank = self.net.prototypes.flat().detach()
        params = list(self.net.parameters())
        for _ in range(self.spec.dispatch.local_epochs):
            for batch in loader:
                moved = to_device(batch, self.device)
                z, state = self.net(moved)
                terms = evaluate(
                    z,
                    state,
                    moved,
                    bank,
                    _cell_targets(moved),
                    self.spec.franking,
                    self.spec.apparatus.tau,
                )
                loss = terms.total + proximal_term(params, anchors, self.spec.franking.mu_prox)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.spec.dispatch.grad_clip)
                scale = warmup_cosine(
                    self._step, self.spec.dispatch.warmup_steps, self._total_steps
                )
                for group in self.opt.param_groups:
                    group["lr"] = self.spec.dispatch.lr * scale
                self.opt.step()
                self._step += 1

    def run(self) -> History:
        history = History()
        for rnd in range(self.spec.dispatch.rounds):
            means, counts = self._emit_prototypes()
            global_means, populations = confidence_weighted(
                means, counts, self.spec.franking.kappa, self.spec.dispatch.budget_bytes
            )
            self.net.prototypes.install(global_means, populations)
            anchors = [p.detach().clone() for p in self.net.parameters()]
            round_losses: list[float] = []
            for loader in self.loaders:
                self._local_update(loader, anchors)
                round_losses.append(self._probe(loader))
            history.round_loss.append(float(sum(round_losses) / len(round_losses)))
            _log.info(
                "round %d/%d loss %.4f", rnd + 1, self.spec.dispatch.rounds, history.round_loss[-1]
            )
        return history

    def _probe(self, loader: DataLoader[Specimen]) -> float:
        self.net.eval()
        bank = self.net.prototypes.flat().detach()
        with torch.no_grad():
            batch = next(iter(loader))
            moved = to_device(batch, self.device)
            z, state = self.net(moved)
            terms = evaluate(
                z,
                state,
                moved,
                bank,
                _cell_targets(moved),
                self.spec.franking,
                self.spec.apparatus.tau,
            )
        return float(terms.total)
