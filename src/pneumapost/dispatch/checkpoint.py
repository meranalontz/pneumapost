from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..fittings.glossary import RunSpec


def save_checkpoint(
    path: str | Path, model_state: dict[str, Any], spec: RunSpec, step: int
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_state,
        "spec": asdict(spec),
        "seed": spec.seed,
        "step": step,
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=False)
    return payload
