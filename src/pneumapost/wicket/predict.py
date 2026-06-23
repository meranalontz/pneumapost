from __future__ import annotations

from ..apparatus.network import build_network
from ..delivery.calibrate import calibrate_threshold
from ..delivery.inference import decide, infer
from ..dispatch.checkpoint import load_checkpoint
from ..fittings.clerking import get_logger
from ..fittings.glossary import Sex, with_overrides
from ..fittings.hocon_io import spec_from_tree
from ..intake.cohort import build_cohort
from ..intake.modalities import collate, to_device

_log = get_logger("wicket.predict")


def predict(checkpoint: str, *, limit: int = 16, device: str = "cpu") -> None:
    payload = load_checkpoint(checkpoint)
    spec = with_overrides(spec_from_tree(payload["spec"]), device=device)
    net = build_network(spec.apparatus, spec.intake.omics_dim).to(device)
    net.load_state_dict(payload["model"])
    cohorts = build_cohort(spec.intake, spec.seed)
    sample = cohorts[0][:limit]
    batch = to_device(collate(sample), device)
    probability = infer(net, batch, spec.apparatus.tau)
    fem = batch["sex"] == int(Sex.FEMALE)
    threshold = calibrate_threshold(
        batch["verdict"][fem].cpu().numpy(),
        probability[fem].cpu().numpy(),
        spec.delivery.target_sensitivity,
    )
    flags = decide(probability, threshold)
    for i in range(len(sample)):
        _log.info(
            "patient %02d  p_malignant=%.3f  decision=%s  truth=%d",
            i,
            float(probability[i]),
            "refer" if bool(flags[i]) else "clear",
            int(batch["verdict"][i]),
        )
