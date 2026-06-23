from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..fittings.clerking import get_logger

_log = get_logger("intake.adapters")

_TCGA_HINT = (
    "Expected a processed TCGA-LUAD feature matrix (samples x features, tab-separated, "
    "header row of feature ids, first column sample id) under the given path. "
    "Obtain the open processed levels from the GDC Data Portal listed in datasets.txt."
)
_SRA_HINT = (
    "Expected a genus-level 16S abundance table (samples x taxa, tab-separated) under the "
    "given path. Obtain the amplicon runs from the ENA/SRA study listed in datasets.txt."
)


def _read_matrix(path: Path, hint: str) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    rows: list[list[float]] = []
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"{path} is empty. {hint}")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[0])
            rows.append([float(v) for v in parts[1:]])
    return ids, np.asarray(rows, dtype=np.float32)


def load_omics_matrix(path: str | Path, omics_dim: int) -> torch.Tensor:
    ids, mat = _read_matrix(Path(path), _TCGA_HINT)
    _log.info("loaded %d TCGA-LUAD omics rows with %d features", len(ids), mat.shape[1])
    mat = _fit_width(mat, omics_dim)
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True) + 1e-6
    return torch.from_numpy((mat - mu) / sd)


def load_taxa_table(path: str | Path, set_size: int, latent: int) -> torch.Tensor:
    ids, mat = _read_matrix(Path(path), _SRA_HINT)
    _log.info("loaded %d 16S rows with %d taxa", len(ids), mat.shape[1])
    rel = mat / (mat.sum(axis=1, keepdims=True) + 1e-8)
    top = np.argsort(-rel, axis=1)[:, :set_size]
    tokens = np.zeros((mat.shape[0], set_size, latent), dtype=np.float32)
    for i in range(mat.shape[0]):
        chosen = top[i]
        tokens[i, : chosen.shape[0], 0] = rel[i, chosen]
        tokens[i, : chosen.shape[0], 1] = np.log1p(mat[i, chosen])
    return torch.from_numpy(tokens)


def _fit_width(mat: np.ndarray, width: int) -> np.ndarray:
    have = mat.shape[1]
    if have == width:
        return mat
    if have > width:
        keep = np.argsort(-mat.var(axis=0))[:width]
        return mat[:, np.sort(keep)]
    pad = np.zeros((mat.shape[0], width - have), dtype=np.float32)
    return np.concatenate([mat, pad], axis=1)
