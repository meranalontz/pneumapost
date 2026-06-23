from __future__ import annotations

import torch
import torch.nn as nn

from ..apparatus.network import PneumaNet, build_network
from ..dispatch.checkpoint import load_checkpoint
from ..fittings.clerking import get_logger
from ..fittings.hocon_io import spec_from_tree
from ..intake.modalities import LATENT, Batch

_log = get_logger("wicket.export")


class EdgeReader(nn.Module):
    def __init__(self, net: PneumaNet, tau: float) -> None:
        super().__init__()
        self.net = net
        self.tau = tau

    def forward(
        self,
        breath: torch.Tensor,
        microbiome: torch.Tensor,
        omics: torch.Tensor,
        presence: torch.Tensor,
        sex: torch.Tensor,
    ) -> torch.Tensor:
        batch = Batch(
            breath=breath,
            microbiome=microbiome,
            omics=omics,
            presence=presence,
            verdict=torch.zeros_like(sex),
            sex=sex,
            never_smoker=torch.zeros_like(sex, dtype=torch.bool),
            driver_negative=torch.zeros_like(sex, dtype=torch.bool),
            arm=torch.zeros_like(sex),
            site=torch.zeros_like(sex),
        )
        z, _ = self.net(batch)
        bank = self.net.prototypes.bank
        benign = bank[0, sex]
        luad = bank[1, sex]
        d0 = torch.sum((z - benign) ** 2, dim=-1)
        d1 = torch.sum((z - luad) ** 2, dim=-1)
        logits = torch.stack([-d0 / self.tau, -d1 / self.tau], dim=1)
        return torch.softmax(logits, dim=1)[:, 1]


def export(checkpoint: str, *, out: str = "runs/pneumapost.onnx") -> None:
    payload = load_checkpoint(checkpoint)
    spec = spec_from_tree(payload["spec"])
    net = build_network(spec.apparatus, spec.intake.omics_dim)
    net.load_state_dict(payload["model"])
    reader = EdgeReader(net, spec.apparatus.tau).eval()
    isp = spec.intake
    args = (
        torch.randn(1, isp.breath_len),
        torch.randn(1, isp.taxa_set_size, LATENT),
        torch.randn(1, isp.omics_dim),
        torch.ones(1, 3, dtype=torch.bool),
        torch.zeros(1, dtype=torch.long),
    )
    torch.onnx.export(
        reader,
        args,
        out,
        input_names=["breath", "microbiome", "omics", "presence", "sex"],
        output_names=["malignancy"],
        opset_version=17,
        dynamo=False,
    )
    _log.info("exported ONNX edge reader -> %s", out)
