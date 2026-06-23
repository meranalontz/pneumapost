from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _arrays(
    y: np.ndarray | list[int], score: np.ndarray | list[float]
) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y, dtype=np.int64), np.asarray(score, dtype=np.float64)


def auroc(y: np.ndarray | list[int], score: np.ndarray | list[float]) -> float:
    yy, ss = _arrays(y, score)
    pos = yy == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(ss)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def operating_threshold(
    y: np.ndarray | list[int], score: np.ndarray | list[float], sensitivity: float
) -> float:
    yy, ss = _arrays(y, score)
    pos = ss[yy == 1]
    if pos.size == 0:
        return float("inf")
    quantile = np.quantile(pos, 1.0 - sensitivity, method="lower")
    return float(quantile)


def specificity_at_sensitivity(
    y: np.ndarray | list[int], score: np.ndarray | list[float], sensitivity: float
) -> float:
    yy, ss = _arrays(y, score)
    thr = operating_threshold(yy, ss, sensitivity)
    neg = ss[yy == 0]
    if neg.size == 0:
        return float("nan")
    return float((neg < thr).mean())


def balanced_accuracy(
    y: np.ndarray | list[int], score: np.ndarray | list[float], threshold: float
) -> float:
    yy, ss = _arrays(y, score)
    pred = (ss >= threshold).astype(np.int64)
    tpr = float((pred[yy == 1] == 1).mean()) if (yy == 1).any() else 0.0
    tnr = float((pred[yy == 0] == 0).mean()) if (yy == 0).any() else 0.0
    return 0.5 * (tpr + tnr)


def f1_score(y: np.ndarray | list[int], score: np.ndarray | list[float], threshold: float) -> float:
    yy, ss = _arrays(y, score)
    pred = (ss >= threshold).astype(np.int64)
    tp = float(((pred == 1) & (yy == 1)).sum())
    fp = float(((pred == 1) & (yy == 0)).sum())
    fn = float(((pred == 0) & (yy == 1)).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0
