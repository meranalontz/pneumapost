from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

from .diagnostics import auroc


def _midrank(x: np.ndarray) -> np.ndarray:
    return rankdata(x, method="average")


def _fast_delong(scores: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    m = n_pos
    n = scores.shape[1] - m
    pos = scores[:, :m]
    neg = scores[:, m:]
    k = scores.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(scores[r])
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_test(
    y: np.ndarray | list[int],
    score_a: np.ndarray | list[float],
    score_b: np.ndarray | list[float],
) -> tuple[float, float]:
    yy = np.asarray(y, dtype=np.int64)
    order = np.argsort(-yy, kind="stable")
    n_pos = int((yy == 1).sum())
    stacked = np.vstack(
        [np.asarray(score_a, dtype=np.float64)[order], np.asarray(score_b, dtype=np.float64)[order]]
    )
    aucs, cov = _fast_delong(stacked, n_pos)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return 0.0, 1.0
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    p = float(2 * norm.sf(abs(z)))
    return z, p


def bootstrap_ci(
    y: np.ndarray | list[int],
    score: np.ndarray | list[float],
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    yy = np.asarray(y, dtype=np.int64)
    ss = np.asarray(score, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = yy.shape[0]
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        samples[b] = auroc(yy[idx], ss[idx])
    lo = float(np.nanpercentile(samples, 100 * alpha / 2))
    hi = float(np.nanpercentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def cohens_d(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    na, nb = aa.size, bb.size
    pooled = ((na - 1) * aa.var(ddof=1) + (nb - 1) * bb.var(ddof=1)) / max(na + nb - 2, 1)
    spread = float(np.sqrt(pooled)) if pooled > 0 else 0.0
    if spread == 0.0:
        return 0.0
    return float((aa.mean() - bb.mean()) / spread)


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    reject = [False] * m
    for rank, idx in enumerate(order):
        if pvalues[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def benjamini_hochberg(pvalues: list[float], alpha: float) -> list[bool]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    reject = [False] * m
    threshold_rank = -1
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= alpha * rank / m:
            threshold_rank = rank
    if threshold_rank > 0:
        for rank, idx in enumerate(order, start=1):
            if rank <= threshold_rank:
                reject[idx] = True
    return reject
